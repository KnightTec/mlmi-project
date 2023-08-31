import os
import sys
from tqdm import tqdm
import csv

from monai.transforms import Compose, LoadImaged, Orientationd, SaveImage, EnsureChannelFirstd

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from tqdm.asyncio import tqdm_asyncio

thread_executor = ThreadPoolExecutor(max_workers=8)
process_executor = ProcessPoolExecutor(max_workers=4)

keys = ["image"]
# Create a MONAI transform pipeline
transform_pipeline = Compose([
    EnsureChannelFirstd(keys),
    Orientationd(keys, axcodes="LPS"),  # Reorient to RAS orientation
])

async def async_load_image(keys, item):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_executor, LoadImaged(keys), item)

async def async_save_image(saver, image, meta_data):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(thread_executor, saver, image, meta_data)

async def process_mr_session(input_session_path, output_session_path):
    # Define keys for dictionary
    images = []
    for root, _, files in os.walk(input_session_path):
        for file in files:
            if file.lower().endswith(('.nii', '.nii.gz')):
                file_path = os.path.join(root, file)
                images.append(file_path)

    # Create data list as a list of dictionaries
    data = [{"image": img,} for img in images]

    image_saver = SaveImage(output_dir=output_session_path, output_postfix="LPS",
                             output_ext=".nii.gz", resample=False, print_log=False, separate_folder=False)

    # Apply transformations
    for item in data:
        transformed = await async_load_image(keys, item)
        transformed = transform_pipeline(transformed)

        if len(transformed["image"].shape) != 4:
            print(f'Skipping scan {transformed["image_meta_dict"]["filename_or_obj"]} because it does not have 3 dimensions')
            continue

        await async_save_image(image_saver, transformed["image"][0, :, :, :], transformed["image_meta_dict"])

    # TODO: write session ID to completed list

async def main():
    mr_sessions_path = sys.argv[1]
    mr_session_ids = sys.argv[2]
    out_path = sys.argv[3]

    session_ids = []
    with open(mr_session_ids, newline='') as csvfile:
        csv_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csv_content:
            session_ids.append(row[0])

    tasks = []

    for item in session_ids:
        session_path = os.path.join(mr_sessions_path, item)
        out_session_path = os.path.join(out_path, item)
        os.makedirs(out_session_path, exist_ok=True)
        tasks.append(process_mr_session(input_session_path=session_path, output_session_path=out_session_path))
    
    await tqdm_asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())