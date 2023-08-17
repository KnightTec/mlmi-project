"""
Script that takes convert the MR session 3D data of the OASIS-3 data into a 2D dataset by taking the central axial slice. 
"""

import sys
import os
from pathlib import Path
import nibabel as nib

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    file_format = ".nii.gz"
    for current_root, dirs, files in os.walk(input_path):
        new_root = current_root.replace(input_path, output_path)
        path = Path(new_root)
        path.mkdir(parents=True, exist_ok=True)
        for file in files:
            if file.endswith(file_format):                
                file_path = os.path.join(current_root, file)

                img1 = nib.load(file_path)
                data = img1.get_fdata()
                affine = img1.affine

                print(data.shape)

                data = data[:,:, data.shape[2] // 2]

                new_image = nib.Nifti1Image(data, affine)
                new_file_path = os.path.join(new_root, file)
                nib.save(new_image, new_file_path)

                print(f"Created 2D slice file: {new_file_path}")


if __name__ == "__main__":
    main()