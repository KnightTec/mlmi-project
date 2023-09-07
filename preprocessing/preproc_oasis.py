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
import asyncio
import time

# Global ThreadPoolExecutor
thread_executor = ThreadPoolExecutor(max_workers=4)     

async def process_mr_session(input_session_path, output_session_path):
    images = []
    try:
        for root, _, files in os.walk(input_session_path):
            for file in files:
                if file.lower().endswith(('.nii', '.nii.gz')):
                    images.append(os.path.join(root, file))
    except FileNotFoundError as e:
        return

    found_t1w = False
    index = 0
    data_dict = {}
    all_keys = []
    non_ref_keys = []
    for img in images:
        if not found_t1w and "T1w" in img:
            data_dict["T1w"] = img
            found_t1w = True
            all_keys.append("T1w")
        else:
            data_dict[str(index)] = img
            all_keys.append(str(index))
            non_ref_keys.append(str(index))
            index += 1

    if not found_t1w:
        return
    if len(all_keys) <= 1:
        return
    
    loop = asyncio.get_running_loop()

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

    loaded_images = await loop.run_in_executor(thread_executor, LoadImaged(all_keys, image_only=False), data_dict)
    transform_pipeline = Compose([
        #Lambdad(keys=[f"{key}_meta_dict" for key in all_keys], func=check_affine),
        #Lambdad(keys=all_keys, func=check_shape),
        EnsureChannelFirstd(all_keys),
        Orientationd(all_keys, axcodes="LPS"),
        ScaleIntensityd(all_keys),
        ResampleToMatchd(keys=non_ref_keys, key_dst="T1w", padding_mode="zeros"), # resample to T1-weighted
        Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0)), # isotropic resampling
        CropForegroundd(keys=all_keys, source_key="T1w", select_fn=lambda x: x > 0.1, margin=10)
    ])
    try:
        transformed_images = transform_pipeline(loaded_images)
    except InvalidScanException as e:
        print("Invalid scan!")
        return

    os.makedirs(output_session_path, exist_ok=True)

    save_transform = SaveImaged(keys=all_keys, output_dir=output_session_path, output_postfix="proc",
                            output_ext=".nii.gz", resample=False, print_log=True, separate_folder=False)
    await loop.run_in_executor(thread_executor, save_transform, transformed_images)

def run_mr_sessions_batch(batch_sessions, mr_sessions_path, out_path):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def batch_main():
        tasks = []
        for item in batch_sessions:
            session_path = os.path.join(mr_sessions_path, item)
            out_session_path = os.path.join(out_path, item)
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

    session_ids = sorted(session_ids)
    chunk_size = 4
    print("Starting preprocessing...")
    total_batch_count = len(session_ids) // chunk_size
    batch_idx = 0
    for batch in tqdm(chunk_sessions(session_ids, chunk_size=chunk_size)):
        print(f"Processing {batch}...")
        run_mr_sessions_batch(batch, mr_sessions_path, out_path)
        print(f"Processed batch {batch_idx} / {total_batch_count}")
        batch_idx += 1

    # with ProcessPoolExecutor(max_workers=2) as executor, tqdm(total=len(session_ids)) as pbar:
    #     futures = {executor.submit(run_mr_sessions_batch, batch, mr_sessions_path, out_path): batch
    #                for batch in chunk_sessions(session_ids, chunk_size=chunk_size)}
    #     print("Submitted tasks to executor.")
    #     for future in as_completed(futures):
    #         batch = futures[future]
    #         try:
    #             future.result()
    #         except Exception as exc:
    #             print(f"A batch generated an exception: {exc}")
    #         pbar.update(len(batch))

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
