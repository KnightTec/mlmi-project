import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# 1. Read both CSV files into pandas DataFrames.
df1 = pd.read_csv('./oasis_3_multimodal_full.csv', sep=";")  # Assuming your first file is named file1.csv and contains a column 'MR_session_ID'
print(df1.columns)
df2 = pd.read_csv('./final/session_ids.csv', sep=";")  # Assuming your second file is named file2.csv and contains columns 'MR_session_ID' and 'label'
print(df2.columns)

# 2. Perform a left join on the MR session IDs.
result = pd.merge(df1, df2, on='MR Session ID', how='inner')
result = result.dropna(subset=['label'])

result["SubjectID"] = result["MR Session ID"].str[0:8]

subject_labels = result.groupby('SubjectID')['label'].mean().reset_index()
subject_labels['label'] = np.round(subject_labels['label']).astype(int)

train_subjects, val_subjects = train_test_split(subject_labels, test_size=0.1, stratify=subject_labels['label'], random_state=42)

train_subjects = train_subjects.drop("label", axis=1)
val_subjects = val_subjects.drop("label", axis=1)

# (Optional) Save the train and validation sets into separate CSV files.

result = result[["SubjectID", "MR Session ID", "MR T1w", "MR T2w", "MR T2*", "MR FLAIR", "MR TOF-MRA", "label"]]

result = result.rename(columns={'MR T1w': 'T1', 'MR T2w': 'T2', "MR T2*": "T2*", "MR FLAIR": "FLAIR", "MR TOF-MRA": "MRA"})

training_set = pd.merge(train_subjects, result, on='SubjectID', how='inner')
validation_set = pd.merge(val_subjects, result, on='SubjectID', how='inner')

training_set.to_csv('./final/unproc/oasis_3_multimodal_train.csv', index=False, sep=";")
validation_set.to_csv('./final/unproc/oasis_3_multimodal_val.csv', index=False, sep=";")

result.fillna("", inplace=True)
def concat_first_last(s):
    if not s:
        return s
    parts = s.split('/')
    filename = parts[-1]
    file_parts = filename.split(".")
    file_parts[0] = f"{file_parts[0]}_proc_final"
    filename = ".".join(file_parts)
    return parts[0] + '/' + filename

result["T1"] = result["T1"].apply(concat_first_last)
result["T2"] = result["T2"].apply(concat_first_last)
result["T2*"] = result["T2*"].apply(concat_first_last)
result["FLAIR"] = result["FLAIR"].apply(concat_first_last)
result["MRA"] = result["MRA"].apply(concat_first_last)

training_set = pd.merge(train_subjects, result, on='SubjectID', how='inner')
validation_set = pd.merge(val_subjects, result, on='SubjectID', how='inner')

os.makedirs("./final/proc", exist_ok=True)
training_set.to_csv('./final/proc/oasis_3_multimodal_train.csv', index=False, sep=";")
validation_set.to_csv('./final/proc/oasis_3_multimodal_val.csv', index=False, sep=";")


# # 4. Save the result to a new CSV file.
# subject_labels.to_csv('subject_labels.csv', index=False)

# # 3. Write the resultant DataFrame to a new CSV file.
# result.to_csv('./final/session_ids_labelled.csv', index=False, sep=";")
