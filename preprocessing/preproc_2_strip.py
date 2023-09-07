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
            out_file = os.path.join(os.path.join(out_dataset_path, dir), file)
            arguments = ['-i', in_file, '-o', out_file, "-g"] 

            try:
                result = subprocess.run(['python', script_path] + arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                # Access the output and error messages if needed
                output = result.stdout
                error = result.stderr
                print(f"Output: {output}")
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()