import os
import sys
from tqdm import tqdm
import csv

from monai.transforms import Compose, LoadImaged, Orientationd, SaveImage, EnsureChannelFirstd


def process_mr_session(input_session_path, output_session_path):
    # Define keys for dictionary
    keys = ["image"]

    images = []
    for root, _, files in os.walk(input_session_path):
        for file in files:
            if file.lower().endswith(('.nii', '.nii.gz')):
                file_path = os.path.join(root, file)
                images.append(file_path)

    # Create data list as a list of dictionaries
    data = [{"image": img,} for img in images]

    # Create a MONAI transform pipeline
    transform = Compose([
        LoadImaged(keys),  # Load NIfTI images
        EnsureChannelFirstd(keys),
        Orientationd(keys, axcodes="LPS"),  # Reorient to RAS orientation
    ])

    image_saver = SaveImage(output_dir=output_session_path, output_postfix="LPS",
                             output_ext=".nii.gz", resample=False, print_log=False, separate_folder=False)

    # Apply transformations
    for item in data:
        transformed = transform(item)

        if (len(transformed["image"].shape) != 4):
            print(f'Skipping scan {transformed["image_meta_dict"]["filename_or_obj"]} because it does not have 3 dimensions')
            continue
        
        # Save transformed image and label back to NIfTI format
        image_saver(transformed["image"][0, :, :, :], meta_data=transformed["image_meta_dict"]) 

def main():
    mr_sessions_path = sys.argv[1]
    mr_session_ids = sys.argv[2]
    out_path = sys.argv[3]

    session_ids = []
    with open(mr_session_ids, newline='') as csvfile:
        csv_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csv_content:
            session_ids.append(row[0])

    for item in tqdm(session_ids):
        session_path = os.path.join(mr_sessions_path, item)
        out_session_path = os.path.join(out_path, item)
        os.makedirs(out_session_path, exist_ok=True)
        process_mr_session(input_session_path=session_path, output_session_path=out_session_path)

if __name__ == "__main__":
    main()