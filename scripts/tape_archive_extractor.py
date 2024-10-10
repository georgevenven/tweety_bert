import struct
import os
import wave
import numpy as np
from tqdm import tqdm

"""
Tape Data Storage Structure:

/tape_data_set
    /data_set_1
    /data_set_2
        /tape_name_1
            /experiment_name
                /recording1
                    filename.raw
                /recording2
                    filename.raw
"""


def read_32bit_integers_from_file(filename):
    integers = []
    
    # Check if the file exists and print its size
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return integers
    else:
        file_size = os.path.getsize(filename)
        print(f"File size: {file_size} bytes")
    
    # Open the binary file in read-binary mode
    with open(filename, 'rb') as file:
        index = 0
        while True:
            # Read 4 bytes (32 bits)
            bytes_data = file.read(4)
            
            # If we've reached the end of the file, break the loop
            if not bytes_data:
                break
            
            if len(bytes_data) != 4:
                print(f"Incomplete data at byte {index}: {bytes_data}")
                break
            
            # Unpack the 4 bytes as a signed 32-bit integer ('i' format in struct for little-endian)
            try:
                integer = struct.unpack('i', bytes_data)[0]
                integers.append(integer)
            except struct.error as e:
                print(f"Error unpacking data at byte {index}: {e}")
                break

            index += 4
    
    return integers


def convert_raw_to_wav(input_directory, output_directory):
    # Traverse the directory structure
    raw_files = []
    for root, dirs, files in os.walk(input_directory):
        raw_files.extend([os.path.join(root, file) for file in files if file.endswith('.raw')])
    
    for raw_file_path in tqdm(raw_files, desc="Converting files"):
        print(f"Processing file: {raw_file_path}")
        
        # Read integers from the .raw file
        integers = read_32bit_integers_from_file(raw_file_path)
        
        # Convert integers to numpy array with dtype int32
        audio_data = np.array(integers, dtype=np.int32)
        
        # Create the mirrored directory structure in the output directory
        relative_path = os.path.relpath(os.path.dirname(raw_file_path), input_directory)
        output_path = os.path.join(output_directory, relative_path)
        os.makedirs(output_path, exist_ok=True)
        
        # Determine the name for the .wav file
        wav_file_name = f"{os.path.splitext(os.path.basename(raw_file_path))[0]}.wav"
        wav_file_path = os.path.join(output_path, wav_file_name)
        
        # Save the data as a .wav file
        with wave.open(wav_file_path, 'w') as wav_file:
            # Set parameters: 1 channel, 4 bytes per sample, 48000 samples per second
            wav_file.setnchannels(1)
            wav_file.setsampwidth(4)  # 4 bytes for 32-bit
            wav_file.setframerate(48000)
            wav_file.writeframes(audio_data.tobytes())
        
        print(f"Converted {raw_file_path} to {wav_file_path}")

# Example usage
input_directory = '/media/george-vengrovski/disk1/tape_data_set'
output_directory = '/media/george-vengrovski/disk1/tape_data_set_wav'
convert_raw_to_wav(input_directory, output_directory)