import subprocess
import os
import sys
from tqdm import tqdm
import argparse
import csv
import shutil

def main():
    parser = argparse.ArgumentParser(description="Dataset skull stripping via SynthStrip")
    parser.add_argument("-i", "--input", type=str, help="input dataset")
    parser.add_argument("-o", "--output", type=str, help="output dataset")
    parser.add_argument("--checkpoint", type=str, help=".csv file with subdirs that have already been processed")
    args = parser.parse_args()

    in_dataset_path = args.input
    out_dataset_path = args.output

    script_path = "/freesurfer/mri_synthmorph"

    os.makedirs(out_dataset_path, exist_ok=True)

    finished_subdirs = set()
    if args.csv:
        if os.path.exists(args.checkpoint):
            with open(args.checkpoint, newline='') as csvfile:
                csv_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in csv_content:
                    finished_subdirs.add(row[0])

    for dir in os.listdir(in_dataset_path):
        if os.path.isdir(os.path.join(in_dataset_path, dir)) and not dir in finished_subdirs:
            subdirs.append(dir)
    subdirs = sorted(subdirs)

    for dir in tqdm(subdirs):
        in_dir_path = os.path.join(in_dataset_path, dir)
        if not os.path.exists(in_dir_path):
            print(f"Skipping {in_dir_path} because it does not exists...")
            continue

        t1_scans = []
        for file in os.listdir(in_dir_path):
            if not file.lower().endswith(('.nii', '.nii.gz')):
                continue
            if "T1w" in file:
                t1_scans.append(file)
        
        # select the T1 scan with the lexicographically highest filename (e.g. run-02 over run-01)
        t1_ref_file = sorted(t1_scans)[-1]
        t1_ref_file_path = os.path.join(in_dir_path, t1_ref_file)

        for file in os.listdir(in_dir_path):
            if not file.lower().endswith(('.nii', '.nii.gz')):
                continue

            in_file = os.path.join(in_dir_path, file)
            out_dir = os.path.join(out_dataset_path, dir)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, file)

            if file == t1_ref_file:
                shutil.copyfile(in_file, out_file)
                print(f"Copying T1 reference file {t1_ref_file}...")
            else:
                # Run synthmorph
                arguments = ['-m', "rigid", '-g', "-o", out_file, in_file, t1_ref_file_path]
                print(f"Running mri_synthmorph {' '.join(arguments)}")
                try:
                    subprocess.run(['python', script_path] + arguments, text=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e}")
        
        with open(args.checkpoint, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([dir])

if __name__ == "__main__":
    main()