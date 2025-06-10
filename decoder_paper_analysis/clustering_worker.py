#!/usr/bin/env python3
"""
Subprocess worker for UMAP/HDBSCAN clustering operations.
This isolates the dangerous GPU memory operations that can cause OOM crashes.
"""

import sys
import json
import time
import pathlib
import traceback
import numpy as np
import cupy as cp
import cuml
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import v_measure_score
from typing import Dict, Any, Optional
from collections import Counter

# Constants that need to match the main script
SILENCE_LABEL_VALUE = 0
RANDOM_STATE = 42

class ClusteringMetrics:
    """Evaluate clustering vs. ground‑truth phrase labels."""
    def __init__(self, gt: np.ndarray, pred: np.ndarray, silence: int = 0):
        gt = np.asarray(gt).ravel(); pred = np.asarray(pred).ravel()
        if gt.shape != pred.shape:
            min_len = min(len(gt), len(pred))
            if abs(len(gt) - len(pred)) > 100: print(f"Warning: GT/Pred shapes differ ({gt.shape} vs {pred.shape}). Truncating to {min_len}")
            gt = gt[:min_len]; pred = pred[:min_len]
        self.gt_raw = gt.astype(int); self.pred = pred.astype(int)
        self.gt = self._merge_silence(self.gt_raw, silence_label=silence) 
        self.gt_types = np.unique(self.gt); self.pred_types = np.unique(self.pred)
        self._build_confusion(); self.mapping = self._hungarian()

    @staticmethod
    def _merge_silence(arr: np.ndarray, silence_label: int) -> np.ndarray:
        if arr.size == 0: return arr
        out = arr.copy(); i = 0
        while i < len(out):
            if out[i] != silence_label: i += 1; continue
            j = i
            while j < len(out) and out[j] == silence_label: j += 1
            left_val = out[i-1] if i > 0 else None; right_val = out[j] if j < len(out) else None
            fill_value = silence_label 
            if left_val is not None and left_val != silence_label: fill_value = left_val
            elif right_val is not None and right_val != silence_label: fill_value = right_val
            elif (left_val is None or left_val == silence_label) and (right_val is None or right_val == silence_label): pass
            out[i:j] = fill_value; i = j
        return out

    def _build_confusion(self) -> None:
        if not self.gt_types.size or not self.pred_types.size: self.C=np.array([],dtype=int).reshape(0,0); self.C_norm=np.array([],dtype=float).reshape(0,0); return
        gt_idx={l:i for i,l in enumerate(self.gt_types)}; pr_idx={l:i for i,l in enumerate(self.pred_types)}
        self.C=np.zeros((len(self.gt_types),len(self.pred_types)),dtype=int)
        for g,p in zip(self.gt,self.pred):
            if g in gt_idx and p in pr_idx: np.add.at(self.C,(gt_idx[g],pr_idx[p]),1)
        cs=self.C.sum(axis=0,keepdims=True)
        with np.errstate(divide='ignore',invalid='ignore'): self.C_norm=np.divide(self.C,cs,where=cs!=0,out=np.zeros_like(self.C,dtype=float))

    def _hungarian(self) -> Dict[int,int]:
        if not hasattr(self,'C_norm') or self.C_norm.size==0: return {}
        cost_matrix=-self.C_norm; 
        try: r_ind,c_ind=linear_sum_assignment(cost_matrix)
        except ValueError: return {}
        mp={}
        for r,c in zip(r_ind,c_ind):
            if r<len(self.gt_types) and c<len(self.pred_types): mp[self.gt_types[r]]=self.pred_types[c]
        return mp

    def v_measure(self) -> float:
        if self.gt.size==0 or self.pred.size==0 or len(np.unique(self.gt))==0 or len(np.unique(self.pred))==0: return 0.0
        try:
            mgt=np.min(self.gt); mpr=np.min(self.pred)
            gt_a=self.gt-mgt if mgt<0 else self.gt; pr_a=self.pred-mpr if mpr<0 else self.pred
            return v_measure_score(gt_a,pr_a)
        except ValueError: return 1.0 if len(self.gt_types)<=1 and len(self.pred_types)<=1 and (len(self.gt_types)==0 or len(self.pred_types)==0 or self.gt_types[0]==self.pred_types[0]) else 0.0

    def _fer_generic(self,use_mask:Optional[np.ndarray]=None)->float:
        if self.gt.size==0: return 100.0
        eff_gt,eff_pred=(self.gt[use_mask],self.pred[use_mask]) if use_mask is not None else (self.gt,self.pred)
        if eff_gt.size==0: return 0.0
        corr=sum(1 for g,p in zip(eff_gt,eff_pred) if g in self.mapping and self.mapping[g]==p)
        return 100.0*(1.0-corr/eff_gt.size) if eff_gt.size > 0 else 0.0
        
    def total_fer(self)->float: return self._fer_generic()
    def matched_fer(self)->float: return self._fer_generic(np.isin(self.gt,list(self.mapping.keys()))) if self.mapping else 0.0
    frame_error_rate=total_fer

    def macro_fer(self)->float:
        if not self.gt_types.size: return 100.0
        p_fer=[]
        for gt_t in self.gt_types:
            msk=(self.gt==gt_t)
            if not np.any(msk): continue
            if gt_t not in self.mapping: p_fer.append(1.0); continue
            errs=np.sum(self.pred[msk]!=self.mapping[gt_t]); tot=np.sum(msk)
            p_fer.append(errs/tot if tot>0 else 0.0)
        return 100.0*np.mean(p_fer) if p_fer else 100.0

    def stats(self)->Dict[str,Any]:
        if self.gt.size==0: return {"pct_types_mapped":0,"pct_frames_mapped":0,"mapped_counts":{},"unmapped_counts":{},"n_gt_types":0,"n_pred_types":0}
        cnts=Counter(self.gt); m_gt_t=set(self.mapping.keys())
        m_frms=sum(cnts[gtl] for gtl in m_gt_t if gtl in cnts)
        n_gt,n_pr=len(self.gt_types),len(self.pred_types)
        return {"pct_types_mapped":100*len(m_gt_t)/n_gt if n_gt else 0,
                "pct_frames_mapped":100*m_frms/self.gt.size if self.gt.size else 0,
                "mapped_counts":{k:v for k,v in cnts.items() if k in m_gt_t},
                "unmapped_counts":{k:v for k,v in cnts.items() if k not in m_gt_t},
                "n_gt_types":n_gt,"n_pred_types":n_pr}

def basic_majority_vote(labels:np.ndarray,window_size:int) -> np.ndarray:
    if window_size<=1 or len(labels)==0: return labels.copy()
    n=len(labels); smoothed=np.copy(labels); h_win=window_size//2
    for i in range(n):
        s,e=max(0,i-h_win),min(n,i+h_win+1); win=labels[s:e]
        if len(win)>0:
            cnts=Counter(win); top2=cnts.most_common(2)
            if len(top2)==1 or top2[0][1]>top2[1][1]: smoothed[i]=top2[0][0]
            else:
                tied=[item[0] for item in top2 if item[1]==top2[0][1]]
                smoothed[i]=labels[i] if labels[i] in tied else sorted(tied)[0]
    return smoothed

def run_clustering_config(config_dict: Dict[str, Any], fold_path: str, n_data_points: int, status_file: str):
    """
    Run a single clustering configuration in isolation.
    This function performs UMAP + HDBSCAN + smoothing + evaluation.
    """
    try:
        # Load data
        data = np.load(fold_path)
        X_full = data["predictions"]
        gt_full = data["ground_truth_labels"]
        
        n_pts = min(n_data_points, X_full.shape[0])
        X_hd = X_full[:n_pts].astype(np.float32, copy=False)
        gt_eval = gt_full[:n_pts]
        
        # Move to GPU
        X_gpu = cp.asarray(X_hd)
        
        # Extract parameters
        umap_params = {k: v for k, v in config_dict.items() if k in ['n_components', 'n_neighbors', 'min_dist', 'metric']}
        hdbscan_params = {k: v for k, v in config_dict.items() if k in ['min_cluster_size', 'min_samples']}
        smoothing_window = config_dict.get('smoothing_window', 0)
        
        # Initialize timing
        t_umap = t_hdbscan = t_eval = float('nan')
        
        # UMAP
        t_start = time.time()
        umap_model = cuml.UMAP(**umap_params, init="spectral", random_state=RANDOM_STATE, n_epochs=200)
        emb_gpu = umap_model.fit_transform(X_gpu)
        t_umap = time.time() - t_start
        
        # HDBSCAN
        t_start = time.time()
        hdb_model = cuml.HDBSCAN(**hdbscan_params, metric='euclidean', prediction_data=False)
        hdb_labels_gpu = hdb_model.fit_predict(emb_gpu)
        hdb_labels_np = cp.asnumpy(hdb_labels_gpu)
        t_hdbscan = time.time() - t_start
        
        # Clean up GPU memory
        del X_gpu, emb_gpu, hdb_labels_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        # Smoothing and evaluation
        t_start = time.time()
        smoothed_preds = basic_majority_vote(hdb_labels_np, smoothing_window) if smoothing_window > 0 else hdb_labels_np.copy()
        cm = ClusteringMetrics(gt=gt_eval, pred=smoothed_preds, silence=SILENCE_LABEL_VALUE)
        stats = cm.stats()
        t_eval = time.time() - t_start
        
        # Prepare results
        results = {
            "fold_path_str": fold_path,
            **config_dict,
            "total_fer": cm.total_fer(),
            "v_measure": cm.v_measure(),
            "matched_fer": cm.matched_fer(),
            "macro_fer": cm.macro_fer(),
            "n_gt_types": stats['n_gt_types'],
            "n_pred_clusters": stats['n_pred_types'],
            "pct_types_mapped": stats['pct_types_mapped'],
            "pct_frames_mapped": stats['pct_frames_mapped'],
            "time_umap": t_umap,
            "time_hdbscan": t_hdbscan,
            "time_eval_block_all_smoothing": t_eval,
            "oom_flag_umap": False,
            "oom_flag_hdbscan": False,
            "error_message": None
        }
        
        # Write success status
        pathlib.Path(status_file).write_text(json.dumps({"status": "OK", "results": results}))
        
    except Exception as e:
        # Clean up GPU memory on error
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
        
        error_str = str(e)
        oom_flag = "out_of_memory" in error_str.lower() or "bad_alloc" in error_str.lower()
        
        error_results = {
            "fold_path_str": fold_path,
            **config_dict,
            "total_fer": float('inf'),
            "v_measure": float('nan'),
            "matched_fer": float('nan'),
            "macro_fer": float('nan'),
            "n_gt_types": float('nan'),
            "n_pred_clusters": float('nan'),
            "pct_types_mapped": float('nan'),
            "pct_frames_mapped": float('nan'),
            "time_umap": float('nan'),
            "time_hdbscan": float('nan'),
            "time_eval_block_all_smoothing": float('nan'),
            "oom_flag_umap": oom_flag,
            "oom_flag_hdbscan": oom_flag,
            "error_message": f"{'OOM ' if oom_flag else ''}Error: {error_str}"
        }
        
        # Write error status
        pathlib.Path(status_file).write_text(json.dumps({
            "status": "ERROR", 
            "error": error_str,
            "oom": oom_flag,
            "traceback": traceback.format_exc(),
            "results": error_results
        }))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: clustering_worker.py <config_json> <fold_path> <n_data_points> <status_file>")
        sys.exit(1)
    
    config_json = sys.argv[1]
    fold_path = sys.argv[2]
    n_data_points = int(sys.argv[3])
    status_file = sys.argv[4]
    
    config_dict = json.loads(config_json)
    run_clustering_config(config_dict, fold_path, n_data_points, status_file) 