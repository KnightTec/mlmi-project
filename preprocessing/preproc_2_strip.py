import subprocess
import os
import sys
from tqdm import tqdm

def main():
    in_dataset_path = sys.argv[1]
    out_dataset_path = sys.argv[2]

    # Replace 'script.py' with the path to the script you want to call
    freesurfer_home = os.getenv("FREESURFER_HOME")
    script_path = os.path.join(freesurfer_home, "python/scripts/mri_synthstrip")

    os.makedirs(out_dataset_path, exist_ok=True)

    subdirs = []
    for dir in os.listdir(in_dataset_path):
        if os.path.isdir(os.path.join(in_dataset_path, dir)):
            subdirs.append(dir)

    subdirs = sorted(subdirs)

    for dir in tqdm(subdirs):
        in_dir_path = os.path.join(in_dataset_path, dir)
        for file in os.listdir(in_dir_path):
            if not file.lower().endswith(('.nii', '.nii.gz')):
                continue
            
            # Run synthstrip
            in_file = os.path.join(in_dir_path, file)
            out_dir = os.path.join(out_dataset_path, dir)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, file)
            arguments = ['-i', in_file, '-o', out_file, "-g"] 
            print(arguments)
            try:
                subprocess.run(['python', script_path] + arguments, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()