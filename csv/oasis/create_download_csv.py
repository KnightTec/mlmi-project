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

def split_dataframe(df, n):
    chunks = []
    chunk_size = len(df) // n
    for i in range(n):
        if i == n - 1:  # If it's the last chunk, take the remaining rows
            chunks.append(df[i*chunk_size:])
        else:
            chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def main():
    mr_session_csv = sys.argv[1]

    mr_session_df = pd.read_csv(mr_session_csv)

    mr_session_df = mr_session_df["MR ID"]
    mr_session_df.drop(mr_session_df.tail(2).index, inplace = True)    

    dfs = split_dataframe(mr_session_df, 10)

    for i, small_df in enumerate(dfs):
        small_df.to_csv(f"download/oasis_3_mr_session_download_{i}.csv", decimal=".", sep=";", index=False, float_format='%.1f', header=False)    

if __name__ == "__main__":
    main()