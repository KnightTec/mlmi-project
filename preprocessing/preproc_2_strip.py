import subprocess
import os
import sys
from tqdm import tqdm
import argparse
import csv

def main():
    parser = argparse.ArgumentParser(description="Dataset skull stripping via SynthStrip")
    parser.add_argument("-i", "--input", type=str, help="input dataset")
    parser.add_argument("-o", "--output", type=str, help="output dataset")
    parser.add_argument("--csv", default="", type=str, help="subdirectories to strip")
    args = parser.parse_args()

    in_dataset_path = args.input
    out_dataset_path = args.output

    # Replace 'script.py' with the path to the script you want to call
    freesurfer_home = os.getenv("FREESURFER_HOME")
    script_path = os.path.join(freesurfer_home, "python/scripts/mri_synthstrip")

    os.makedirs(out_dataset_path, exist_ok=True)

    subdirs = []
    if args.csv:
        with open(args.csv, newline='') as csvfile:
            csv_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in csv_content:
                subdirs.append(row[0])
    else:
        for dir in os.listdir(in_dataset_path):
            if os.path.isdir(os.path.join(in_dataset_path, dir)):
                subdirs.append(dir)
        subdirs = sorted(subdirs)

    for dir in tqdm(subdirs):
        in_dir_path = os.path.join(in_dataset_path, dir)
        if not os.path.exists(in_dir_path):
            print(f"Skipping {in_dir_path} because it does not exists...")
            continue
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