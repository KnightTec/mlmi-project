from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import random


class MiriadDataset(Dataset):

    def __init__(self, root : str, train : bool) -> None:
        super().__init__()

        self._file_list = []
        file_format = ".nii"
        for current_root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(file_format):
                    self._file_list.append(os.path.join(current_root, file))

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

        return mri_data, label
    
