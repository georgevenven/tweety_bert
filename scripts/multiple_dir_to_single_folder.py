import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool

def copy_file(args):
    item_src_path, item_dst_path = args
    if not os.path.exists(item_dst_path):
        shutil.copy(item_src_path, item_dst_path)
    else:
        print(f"File {item_dst_path} already exists. Skipping.")

def process_directory(args):
    src_dir, dst_dir = args
    if not os.path.isdir(src_dir):
        print(f"Skipping non-directory: {src_dir}")
        return []

    file_pairs = []
    for item in os.listdir(src_dir):
        item_src_path = os.path.join(src_dir, item)
        item_dst_path = os.path.join(dst_dir, item)

        if os.path.isfile(item_src_path):
            file_pairs.append((item_src_path, item_dst_path))
        elif os.path.isdir(item_src_path):
            if not os.path.exists(item_dst_path):
                shutil.copytree(item_src_path, item_dst_path)
            else:
                print(f"Directory {item_dst_path} already exists. Skipping.")

    return file_pairs

def combine_directories(src_dirs, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    with Pool() as pool:
        directory_args = [(src_dir, dst_dir) for src_dir in src_dirs]
        file_pairs = []
        for result in pool.imap_unordered(process_directory, directory_args):
            file_pairs.extend(result)

        total_files = len(file_pairs)
        with tqdm(total=total_files, desc="Processing files") as pbar:
            for _ in pool.imap_unordered(copy_file, file_pairs):
                pbar.update()

if __name__ == "__main__":
    src_dirs = [
        "/media/george-vengrovski/disk2/zebra_finch/combined_specs",
        "/media/george-vengrovski/disk2/canary_yarden/combined_no_clip_specs",
        "/media/george-vengrovski/disk2/canary/sorted_2/combined_specs",
        "/media/george-vengrovski/disk2/canary/sorted_1/combined_spec",
        "/media/george-vengrovski/disk2/budgie/warble_spec",
        "/media/george-vengrovski/disk2/budgie/T5_ssd_combined_specs",
        "/media/george-vengrovski/disk2/budgie/pair_spec",
        "/media/george-vengrovski/disk2/brown_thrasher/brown_thrasher_specs",
        "/media/george-vengrovski/disk2/bengalese-finch/bengalese-finch_nickle_dave/combined_specs",
        "/media/george-vengrovski/disk2/bengalese-finch/3470165/combined_specs",

    ]
    dst_dir = "/media/george-vengrovski/disk1/multispecies_data_set"

    combine_directories(src_dirs, dst_dir)
    