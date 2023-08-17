'''
This script construct a multimodal dataset using the following 3 OASIS-3 spreadsheets:
    - MR Sessions
    - Subjects
    - ADRC Clinical Data
The paths to the spreadsheets are given to the script via command line arguments.
The 4th command line argument should contain the path to the mr session image data

The resulting spreadsheet contains the following columns:
    - MR ID
    - Gender
    - Age
    - MMSE (Mini–mental state examination) result
    - CDR (Clinical Dementia Rating); 0 = Cognitively Normal, 0.5 = Mild Symptomatic AD, >1 = Alzheimer's Disease
'''

import sys
import pandas as pd
import os

def main():
    mr_session_csv = sys.argv[1]
    subjects_csv = sys.argv[2]
    adrc_csv = sys.argv[3]
    mr_sessions_path = sys.argv[4]

    mr_session_df = pd.read_csv(mr_session_csv)
    mr_session_df = mr_session_df.iloc[:-2]
    mr_session_df = mr_session_df[["MR ID", "Subject"]]
    mr_session_df["day"] = mr_session_df["MR ID"].apply(lambda x: int(x.split("_")[2][1:]))
    print(mr_session_df)

    subjects_df = pd.read_csv(subjects_csv)
    subjects_df = subjects_df.iloc[1:]
    subjects_df = subjects_df[["Subject", "M/F"]]
    subjects_df["Subject ID"] = subjects_df["Subject"].apply(lambda x: int(x[3:]))
    print(subjects_df)

    adrc_df = pd.read_csv(adrc_csv)
    adrc_df = adrc_df[["ADRC_ADRCCLINICALDATA ID", "Subject", "ageAtEntry", "mmse", "cdr"]]
    adrc_df["day"] = adrc_df["ADRC_ADRCCLINICALDATA ID"].apply(lambda x: int(x.split("_")[2][1:]))
    print(adrc_df)

    merged_df = pd.merge(left=mr_session_df, right=subjects_df, how="left", on="Subject")

    merged_df = merged_df.sort_values('day')
    adrc_df = adrc_df.sort_values('day')
    merged_df = pd.merge_asof(merged_df, adrc_df, on="day", by='Subject', direction='nearest')
    merged_df = merged_df.sort_values(["Subject ID", "day"])

    merged_df["year"] = merged_df["day"].apply(lambda x: x / 365.0)
    merged_df["Age"] = merged_df["ageAtEntry"] + merged_df["year"]

    merged_df = merged_df[["MR ID", "M/F", "Age", "mmse", "cdr"]]
    merged_df.rename(columns={'MR ID': 'MR Session ID', "M/F": "Gender"}, inplace=True)

    merged_df.reset_index()

    # get image file paths
    # if one modality contains
    mr_file_paths = {} 
    i = 0
    for index, row in merged_df.iterrows():
        mr_id = row["MR Session ID"]
        mr_file_paths[mr_id] = {}
        mr_session_path = os.path.join(mr_sessions_path, mr_id)
        session_file_paths = {
                "T1w": [],
                "T2w": [],
                "T2star": [],
                "angio": [],
                "FLAIR": [],
            }
        for current_root, dirs, files in os.walk(mr_session_path):
            for file in files:
                if file.endswith(".nii.gz"):
                    # if "sub-OAS30033_ses-d0133_run-02_T2w.nii.gz" in file:
                    #     continue
                    # elif "sub-OAS31065_ses-d0044_echo-1_run-02_FLAIR.nii.gz" in file:
                    #     continue
                    i += 1
                    print(i)
                    for key in session_file_paths.keys():
                        if key in file:
                            session_file_paths[key].append(os.path.join(current_root, file))
        
        for key in session_file_paths.keys():
            file_list = session_file_paths[key]
            if len(file_list) == 0:
                session_file_paths[key] = ""
                continue
            selected_file = sorted(session_file_paths[key])[-1]
            session_file_paths[key] = selected_file.replace(mr_sessions_path, "")[1:]
        
        mr_file_paths[mr_id] = session_file_paths

    t1w_column = []
    t2w_column = []
    t2_star_column = []
    flair_column = []
    tof_mra_column = []
    for mr_session_id in merged_df["MR Session ID"]:
        paths = mr_file_paths[mr_session_id]
        t1w_column.append(paths["T1w"])
        t2w_column.append(paths["T2w"])
        t2_star_column.append(paths["T2star"])
        flair_column.append(paths["FLAIR"])
        tof_mra_column.append(paths["angio"])

    merged_df["MR T1w"] = t1w_column
    merged_df["MR T2w"] = t2w_column
    merged_df["MR T2*"] = t2_star_column
    merged_df["MR FLAIR"] = flair_column
    merged_df["MR TOF-MRA"] = tof_mra_column

    merged_df["cdr"] = merged_df["cdr"].fillna(value=0)
    merged_df["label"] = merged_df["cdr"].apply(lambda x: 1 if x >= 0.5 else 0)

    print(merged_df)
    merged_df.to_csv("oasis_3_multimodal_full.csv", decimal=".", sep=";", index=False, float_format='%.1f')

    merged_df = merged_df.sample(frac=1)

    train_frac = 0.9
    train_set = merged_df.head(int(len(merged_df) * train_frac))
    print(train_set)
    val_set = merged_df.tail(int(len(merged_df) * (1 - train_frac)))
    print(val_set)
    
    train_set.to_csv("oasis_3_multimodal_train.csv", decimal=".", sep=";", index=False, float_format='%.1f')
    val_set.to_csv("oasis_3_multimodal_val.csv", decimal=".", sep=";", index=False, float_format='%.1f')

if __name__ == "__main__":
    main()