import nibabel as nib
from deepbrain import Extractor


def main():
    # Load a nifti as 3d numpy image [H, W, D]
    img = nib.load(filename="./strip_test/sub-OAS30001_ses-d0129_run-01_T1w_LPS.nii.gz").get_fdata()

    ext = Extractor()

    # `prob` will be a 3d numpy image containing probability 
    # of being brain tissue for each of the voxels in `img`
    prob = ext.run(img) 

    # mask can be obtained as:
    mask = prob > 0.5

    print(mask.shape)

if __name__ == "__main__":
    main()