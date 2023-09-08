import os
import sys
import csv
from tqdm import tqdm
from monai.transforms import ( 
    Compose, 
    LoadImaged, 
    SaveImaged, 
    EnsureChannelFirstd,
    CropForegroundd,
    ScaleIntensityd,
    CenterSpatialCropd,
    SpatialPadd,
)
import time
   
def process_subject(input_subject_path, output_subject_path):
    images = []
    try:
        for root, _, files in os.walk(input_subject_path):
            for file in files:
                if file.lower().endswith(('.nii', '.nii.gz')):
                    images.append(os.path.join(root, file))
    except FileNotFoundError as e:
        return

    keys = ["img"]
    data = [{"img" : file} for file in images]
    
    os.makedirs(output_subject_path, exist_ok=True)

    transform_pipeline = Compose([
        LoadImaged(keys, image_only=False),
        EnsureChannelFirstd(keys),
        ScaleIntensityd(keys),
        CropForegroundd(keys=keys, source_key=keys[0], select_fn=lambda x: x > 0.0, margin=4),
        CenterSpatialCropd(keys=keys, roi_size=(-1, -1, 155)),
        SpatialPadd(keys=keys, spatial_size=(240, 240, 155)),
        SaveImaged(keys=keys, output_dir=output_subject_path, output_postfix="final",
                            output_ext=".nii.gz", resample=False, print_log=False, separate_folder=False)
    ])

    for image in data:
        transform_pipeline(image)

def main():
    miriad_dataset_path = sys.argv[1]
    out_path = sys.argv[2]

    os.makedirs(out_path, exist_ok=True)

    subject_ids = []
    for subject_id in os.listdir(miriad_dataset_path):
        if os.path.isdir(os.path.join(miriad_dataset_path, subject_id)):
            subject_ids.append(subject_id)

    subject_ids = sorted(subject_ids)

    for subject_id in tqdm(subject_ids):
        in_subject_path = os.path.join(miriad_dataset_path, subject_id)
        out_subject_path = os.path.join(out_path, subject_id)
        process_subject(in_subject_path, out_subject_path)

if __name__ == "__main__":
    # get the start time
    st = time.time()
    # run main
    main()
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print()
    print('Execution time:', elapsed_time, 'seconds')
