'''
This script construct a multimodal dataset using the following 3 OASIS-3 spreadsheets:
    - MR Sessions
    - Subjects
    - ADRC Clinical Data
The paths to the spreadsheets are given to the script via command line arguments.

The resulting spreadsheet contains the following columns:
    - MR ID
    - Gender
    - Age
    - MMSE (Mini–mental state examination) result
    - CDR (Clinical Dementia Rating); 0 = Cognitively Normal, 0.5 = Mild Symptomatic AD, >1 = Alzheimer's Disease
'''
# TODO: add more tabular data from ADRC Clinical Data table

import sys
import pandas as pd


def main():
    mr_session_csv = sys.argv[1]
    subjects_csv = sys.argv[2]
    adrc_csv = sys.argv[3]

    mr_session_df = pd.read_csv(mr_session_csv)
    mr_session_df = mr_session_df.iloc[:-2]
    mr_session_df = mr_session_df[["MR ID", "Subject", "Scans"]]
    mr_session_df["Scans"] = mr_session_df["Scans"].apply(lambda x: str(x))
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

    merged_df = merged_df[["MR ID", "Scans", "M/F", "Age", "mmse", "cdr"]]
    merged_df.rename(columns={'MR ID': 'MR Session ID', "M/F": "Gender"}, inplace=True)

    print(merged_df)

    merged_df.to_csv("oasis_3_mulitmodal_full.csv", decimal=".", sep=";", index=False, float_format='%.1f')


if __name__ == "__main__":
    main()