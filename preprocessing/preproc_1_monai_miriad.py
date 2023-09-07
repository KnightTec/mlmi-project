import os
import sys
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from monai.transforms import ( 
    Compose, 
    LoadImaged, 
    Orientationd, 
    SaveImaged, 
    EnsureChannelFirstd,
    ResampleToMatchd,
    Spacingd,
    CropForegroundd,
    ScaleIntensityd,
    Lambdad,
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

    class InvalidScanException(Exception):
        pass

    def check_affine(data):    
        if data["qform_code"] == 0 and data["sform_code"] == 0:
            raise InvalidScanException
        return data

    def check_shape(data):    
        if len(data.shape) != 3:
            raise InvalidScanException
        return data
    
    os.makedirs(output_subject_path, exist_ok=True)

    transform_pipeline = Compose([
        LoadImaged(keys, image_only=False),
        #Lambdad(keys=[f"{key}_meta_dict" for key in all_keys], func=check_affine),
        #Lambdad(keys=all_keys, func=check_shape),
        EnsureChannelFirstd(keys),
        Orientationd(keys, axcodes="LPS"),
        ScaleIntensityd(keys),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0)), # isotropic resampling
        CropForegroundd(keys=keys, source_key=keys[0], select_fn=lambda x: x > 0.1, margin=10),
        SaveImaged(keys=keys, output_dir=output_subject_path, output_postfix="monai",
                            output_ext=".nii.gz", resample=False, print_log=False, separate_folder=False)
    ])

    for image in data:
        try:
            transform_pipeline(image)
        except InvalidScanException as e:
            print(f"Invalid scan detected - {image['img']}")
            return

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
