import nibabel as nib
import os
import sys
from tqdm import tqdm
import csv

def check_orientation_metadata(nifti_file_path):
    try:
        # Load the NIfTI file
        img = nib.load(nifti_file_path)
    except:
        print(f"Failed to load {nifti_file_path}. Skipping...")
        return True  # Skip this file but don't count it as missing metadata
    
    # Extract the header
    header = img.header
    
    # Check for orientation information in the header
    qform_code = int(header['qform_code'])
    sform_code = int(header['sform_code'])
    
    return qform_code > 0 or sform_code > 0

modalities = ["T1w", "T2w", "T2star", "FLAIR", "angio"]

modality_counts_total = {
    "T1w": 0,
    "T2w": 0,
    "T2star": 0,
    "FLAIR": 0,
    "angio": 0
}

modality_counts_with_meta = {
    "T1w": 0,
    "T2w": 0,
    "T2star": 0,
    "FLAIR": 0,
    "angio": 0
}

def check_all_nifti_files(directory):
    all_files_count = 0
    missing_metadata_count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.nii', '.nii.gz')):
                all_files_count += 1
                file_path = os.path.join(root, file)
                
                has_metadata = check_orientation_metadata(file_path)

                current_modality = ""
                for modality in modalities:
                    if modality in file:
                        current_modality = modality
                        break
                modality_counts_total[current_modality] += 1
                if has_metadata:
                    modality_counts_with_meta[current_modality] += 1
                
                if not has_metadata:
                    missing_metadata_count += 1
    
    all_have_meta = missing_metadata_count == 0

    with_meta_count = all_files_count - missing_metadata_count

    if not all_have_meta and all_files_count != missing_metadata_count:
        # just check that we can drop the full MR session
        session_id = os.path.basename(os.path.normpath(directory))
        print(f"{session_id} is only missing metadata partially")

    return all_have_meta, with_meta_count, all_files_count

def main():
    mr_sessions_path = sys.argv[1]

    sessions_with_meta = []

    total_session_count = 0
    total_file_count = 0
    session_count_with_meta = 0
    total_file_count_with_meta = 0
    for item in tqdm(os.listdir(mr_sessions_path)):
        session_path = os.path.join(mr_sessions_path, item)
        total_session_count += 1
        all_have_meta, with_meta_count, all_files_count = check_all_nifti_files(session_path)
        total_file_count += all_files_count
        total_file_count_with_meta += with_meta_count
        if all_have_meta:
            session_count_with_meta += 1
            sessions_with_meta.append(item)

    print("-----------------------------------------------------------")
    print(f"{total_file_count_with_meta} out of {total_file_count} scans contain full metadata")
    print(f"{session_count_with_meta} out of {total_session_count} MR sessions contain full metadata")
    print("Total modality counts:")
    print(modality_counts_total)
    print("Modality counts with meta:")
    print(modality_counts_with_meta)

    with open('oasis_3_mr_session_ids_clean.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        for id in sessions_with_meta:
            csvwriter.writerow([id])


if __name__ == "__main__":
    main()