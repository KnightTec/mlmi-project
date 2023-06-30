from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib


class MiriadDataset(Dataset):

    def __init__(self, root : str) -> None:
        super().__init__()

        self._file_list = []
        file_format = ".nii"
        for current_root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(file_format):
                    self._file_list.append(os.path.join(current_root, file))

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