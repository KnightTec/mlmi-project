import os
import sys
from tqdm import tqdm
import csv

def session_has_t1w(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.nii', '.nii.gz')):
                if "T1w" in file:
                    return True
    return False

def main():
    mr_sessions_path = sys.argv[1]

    for item in tqdm(os.listdir(mr_sessions_path)):
        session_path = os.path.join(mr_sessions_path, item)
        if not session_has_t1w(session_path):
            print(f"{item} has no T1-weighted scans!")

if __name__ == "__main__":
    main()