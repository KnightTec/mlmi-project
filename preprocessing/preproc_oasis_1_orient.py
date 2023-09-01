import os
import sys
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from monai.transforms import Compose, LoadImaged, Orientationd, SaveImage, EnsureChannelFirstd
import asyncio
import time

# Global ThreadPoolExecutor
thread_executor = ThreadPoolExecutor(max_workers=4) 

keys = ["image"]

async def async_load_image(keys, item):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_executor, LoadImaged(keys, image_only=False), item)

async def async_save_image(saver, image, meta_data):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(thread_executor, saver, image, meta_data)

async def process_mr_session(input_session_path, output_session_path):
    images = []
    try:
        for root, _, files in os.walk(input_session_path):
            for file in files:
                if file.lower().endswith(('.nii', '.nii.gz')):
                    images.append(os.path.join(root, file))
    except FileNotFoundError as e:
        return

    data = [{"image": img,} for img in images]

    image_saver = SaveImage(output_dir=output_session_path, output_postfix="LPS",
                            output_ext=".nii.gz", resample=False, print_log=False, separate_folder=False)

    transform_pipeline = Compose([
        EnsureChannelFirstd(keys),
        Orientationd(keys, axcodes="LPS"),
    ])

    for item in data:
        loaded_data = await async_load_image(keys, item)
        transformed = transform_pipeline(loaded_data)
        if len(transformed["image"].shape) != 4:
            print(f'Skipping scan {transformed["image_meta_dict"]["filename_or_obj"]} because it does not have 3 dimensions')
            continue
        await async_save_image(image_saver, transformed["image"][0, :, :, :], transformed["image_meta_dict"])

def run_mr_sessions_batch(batch_sessions, mr_sessions_path, out_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def batch_main():
        tasks = []
        for item in batch_sessions:
            session_path = os.path.join(mr_sessions_path, item)
            out_session_path = os.path.join(out_path, item)
            os.makedirs(out_session_path, exist_ok=True)
            tasks.append(process_mr_session(session_path, out_session_path))
        await asyncio.gather(*tasks)

    loop.run_until_complete(batch_main())

def chunk_sessions(session_ids, chunk_size):
    for i in range(0, len(session_ids), chunk_size):
        yield session_ids[i:i + chunk_size]

def main():
    mr_sessions_path = sys.argv[1]
    mr_session_ids = sys.argv[2]
    out_path = sys.argv[3]

    session_ids = []
    with open(mr_session_ids, newline='') as csvfile:
        csv_content = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csv_content:
            session_ids.append(row[0])

    chunk_size = 8
    print("Starting preprocessing...")
    with ProcessPoolExecutor(max_workers=os.cpu_count(), max_tasks_per_child=8) as executor, tqdm(total=len(session_ids)) as pbar:
        futures = {executor.submit(run_mr_sessions_batch, batch, mr_sessions_path, out_path): batch
                   for batch in chunk_sessions(session_ids, chunk_size=chunk_size)}
        print("Submitted tasks to executor.")
        for future in as_completed(futures):
            batch = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"A batch generated an exception: {exc}")
            pbar.update(len(batch))

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
