import numpy as np

# same as analysis.py, later migrate to a single file for post-processing

def syllable_to_phrase_labels(arr, silence=-1):
    """
    Convert a sequence of syllable labels into a sequence of phrase labels,
    merging silence bins with their nearest adjacent syllables.


    For each contiguous block of silence:
    - If it's bounded by the same label on both sides, assign that label to all.
    - If it's bounded by two different labels, assign each time-bin to the closer label;
    ties go to the left.
    - If it's at the beginning or end (missing one side), assign to the available side.


    Parameters
    ----------
    arr : np.ndarray
        Array of integer labels, where `silence` frames are indicated by `silence`.
    silence : int, optional
        Integer value representing silence, by default -1.


    Returns
    -------
    np.ndarray
        Array of phrase-level labels with silence frames appropriately merged.
    """
    new_arr = np.array(arr, dtype=int)
    length = len(new_arr)
    if length == 0:
        return new_arr  # Edge case: empty input


    # Helper function to find contiguous regions of silence
    def find_silence_runs(labels):
        runs = []
        in_silence = False
        start = None


        for i, val in enumerate(labels):
            if val == silence and not in_silence:
                in_silence = True
                start = i
            elif val != silence and in_silence:
                runs.append((start, i - 1))
                in_silence = False
        # If ended in silence
        if in_silence:
            runs.append((start, length - 1))
        return runs


    # Identify contiguous silence regions
    silence_runs = find_silence_runs(new_arr)


    for start_idx, end_idx in silence_runs:
        # Check left and right labels
        left_label = new_arr[start_idx - 1] if start_idx > 0 else None
        right_label = new_arr[end_idx + 1] if end_idx < length - 1 else None


        if left_label is None and right_label is None:
            # Entire array is silence or single region with no bounding labels
            # Do nothing or choose some default strategy
            continue
        elif left_label is None:
            # Leading silence; merge with right label
            new_arr[start_idx:end_idx+1] = right_label
        elif right_label is None:
            # Trailing silence; merge with left label
            new_arr[start_idx:end_idx+1] = left_label
        elif left_label == right_label:
            # Same label on both sides
            new_arr[start_idx:end_idx+1] = left_label
        else:
            # Different labels on both sides
            # Assign each bin to whichever side is closer; ties go left
            left_distances = np.arange(start_idx, end_idx + 1) - (start_idx - 1)
            right_distances = (end_idx + 1) - np.arange(start_idx, end_idx + 1)


            for i in range(start_idx, end_idx + 1):
                # Distance from left non-silence is (i - (start_idx - 1))
                dist_left = i - (start_idx - 1)
                # Distance from right non-silence is ((end_idx + 1) - i)
                dist_right = (end_idx + 1) - i


                if dist_left < dist_right:
                    new_arr[i] = left_label
                elif dist_right < dist_left:
                    new_arr[i] = right_label
                else:
                    # Tie -> go left
                    new_arr[i] = left_label


    return new_arr

def majority_vote(data, window_size=1):
    """
    Return an array of the same length as 'data',
    where each index i is replaced by the majority over
    a window around i. No padding is added.
    """
    from collections import Counter
    data = np.asarray(data)
    n = len(data)
    
    # If window_size=1, no smoothing
    if window_size <= 1 or n == 0:
        return data.copy()
    
    half_w = window_size // 2
    output = np.zeros_like(data)
    
    for i in range(n):
        # define start/end, clamped
        start = max(0, i - half_w)
        end   = min(n, i + half_w + 1)
        window = data[start:end]
        
        # majority
        c = Counter(window)
        major_label = max(c, key=c.get)  # picks the label with highest freq
        output[i] = major_label
    
    return output

def fill_noise_with_nearest_label(labels):
    """
    For each noise point (labeled -1), find the nearest non-noise
    label to the left or right and assign it to this point. If no
    non-noise label is found, it remains -1.
    
    Parameters:
    - labels: np.ndarray
        Array of cluster labels where -1 indicates noise.


    Returns:
    - labels: np.ndarray
        Array with noise points replaced by the nearest non-noise labels,
        when possible.
    """
    noise_indices = np.where(labels == -1)[0]
    for idx in noise_indices:
        # Search left
        left_idx = idx - 1
        while left_idx >= 0 and labels[left_idx] == -1:
            left_idx -= 1
        
        # Search right
        right_idx = idx + 1
        while right_idx < len(labels) and labels[right_idx] == -1:
            right_idx += 1
        
        # Compute distances if valid
        left_dist = (idx - left_idx) if left_idx >= 0 else np.inf
        right_dist = (right_idx - idx) if right_idx < len(labels) else np.inf

        # Assign based on nearest non-noise label
        if left_dist == np.inf and right_dist == np.inf:
            # No non-noise neighbors found, remain -1
            continue
        elif left_dist <= right_dist:
            labels[idx] = labels[left_idx]
        else:
            labels[idx] = labels[right_idx]

    return labels