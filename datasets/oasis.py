from torch.utils.data import Dataset
import os
import nibabel as nib
import pandas as pd


class Oasis3CTDataset(Dataset):

    def __init__(self, ct_sessions_csv_path: str, ct_sessions_data_path: str) -> None:
        super().__init__()

        df = pd.read_csv(ct_sessions_csv_path)
        # self._scan_count = table size
        self._file_list = []
        file_format = ".nii.gz"
        for current_root, dirs, files in os.walk(ct_sessions_data_path):
            for file in files:
                if file.endswith(file_format):
                    self._file_list.append(os.path.join(current_root, file))

    def __len__(self):
        return len(self._file_list)
    
    def __getitem__(self, index):
        file_name = self._file_list[index]

        mri = nib.load(file_name)
        mri_data = mri.get_fdata()

        mri_data = mri_data.reshape((1, *mri_data.shape))
        return mri_data
    

class Oasis3PETDataset(Dataset):

    def __init__(self, pet_sessions_csv_path: str, pet_sessions_data_path: str) -> None:
        super().__init__()

        df = pd.read_csv(pet_sessions_csv_path)
        # self._scan_count = table size
        self._file_list = []
        file_format = ".nii.gz"
        for current_root, dirs, files in os.walk(pet_sessions_data_path):
            for file in files:
                if file.endswith(file_format):
                    self._file_list.append(os.path.join(current_root, file))

    def __len__(self):
        return len(self._file_list)
    
    def __getitem__(self, index):
        file_name = self._file_list[index]

        mri = nib.load(file_name)
        mri_data = mri.get_fdata()

        mri_data = mri_data.reshape((1, *mri_data.shape))
        return mri_data
    

class Oasis3MRDataset(Dataset):

    def __init__(self, mr_sessions_csv_path: str, mr_sessions_data_path: str) -> None:
        super().__init__()

        df = pd.read_csv(mr_sessions_csv_path)
        # self._scan_count = table size
        self._file_list = []
        file_format = ".nii.gz"
        for current_root, dirs, files in os.walk(mr_sessions_data_path):
            for file in files:
                if file.endswith(file_format):
                    self._file_list.append(os.path.join(current_root, file))

    def __len__(self):
        return len(self._file_list)
    
    def __getitem__(self, index):
        file_name = self._file_list[index]

        mri = nib.load(file_name)
        mri_data = mri.get_fdata()

        mri_data = mri_data.reshape((1, *mri_data.shape))

        modality = file_name.split("_")[-1].split(".")[0]

        return mri_data, modality
    