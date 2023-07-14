from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import random


class MiriadDataset(Dataset):

    def __init__(self, root : str, train : bool, transform=None, crop=0.0) -> None:
        super().__init__()

        if not 0 <= crop < 1:
            raise ValueError("crop parameter must be between 0 and 1")
        self._crop = crop * 0.5

        self._transform = transform

        self._file_list = []
        file_format = ".nii"
        for current_root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(file_format):
                    self._file_list.append(os.path.join(current_root, file))

        # TODO: fix leakage from training to validation set caused by multiple scans of the same patient

        random.shuffle(self._file_list)

        reduced_file_list = []
        ad_count = 0
        hc_count = 0
        for file_name in self._file_list:
            if "AD" in file_name and ad_count < 200:
                reduced_file_list.append(file_name)
                ad_count += 1
            elif "HC" in file_name and hc_count < 200:
                reduced_file_list.append(file_name)
                hc_count += 1
        self._file_list = reduced_file_list
        random.shuffle(self._file_list)
        if train:
            self._file_list = self._file_list[: int(0.8 * len(self._file_list))]
        else:
            self._file_list = self._file_list[-int(0.2 * len(self._file_list)) :]


    def __len__(self):
        return len(self._file_list)
    
    def __getitem__(self, index):
        file_name = self._file_list[index]

        if "AD" in file_name:
            label = 1
        elif "HC" in file_name:
            label = 0
        else:
            assert False

        mri = nib.load(file_name)
        mri_data = mri.get_fdata()

        if self._crop > 0:
            start_idx_0 = int(mri_data.shape[0] * self._crop) 
            end_idx_0 = int(mri_data.shape[0] * (1.0 - self._crop))
            start_idx_1 = int(mri_data.shape[1] * self._crop) 
            end_idx_1 = int(mri_data.shape[1] * (1.0 - self._crop))
            start_idx_2 = int(mri_data.shape[2] * self._crop) 
            end_idx_2 = int(mri_data.shape[2] * (1.0 - self._crop))
            mri_data = mri_data[start_idx_0:end_idx_0, start_idx_1:end_idx_1, start_idx_2:end_idx_2]

        if self._transform:
            mri_data = self._transform(mri_data)

        mri_data = mri_data.reshape((1, *mri_data.shape))
        return mri_data, label
    
