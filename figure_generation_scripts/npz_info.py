import numpy as np

class npz_info:
    def __init__(self, file_path, max_length=500):
        """
        Initialize npz_info to analyze contents of a .npz file
        
        Args:
            file_path (str): Path to the .npz file
            max_length (int): Maximum length for truncating output displays
        """
        try:
            self.data = np.load(file_path, allow_pickle=True)
            self.file_path = file_path
            self.max_length = max_length
            
            # Get list of all arrays stored in the npz file
            self.keys = list(self.data.keys())
            
            print(f"NPZ file: {file_path}")
            print(f"Contains {len(self.keys)} arrays:")
            for key in self.keys:
                array = self.data[key]
                print(f"- {key}: shape={array.shape}, dtype={array.dtype}")
                
                # Add check for file_indices
                if key == 'file_indices':
                    unique_files = len(np.unique(array))
                    print(f"  Number of unique file indices: {unique_files}")
                
        except Exception as e:
            print(f"Error loading NPZ file: {str(e)}")
            raise

info = npz_info('/media/george-vengrovski/66AA-9C0A/yarden_umaps_for_paper/llb16_for_paper.npz')



