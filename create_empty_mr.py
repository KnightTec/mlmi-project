import numpy as np
import nibabel as nib

# Create an empty 8x8x8 numpy array filled with zeros
data = np.zeros((8, 8))

# Create an identity affine matrix. This is a 4x4 matrix that doesn't change
# the spatial coordinates when applied to the data.
affine = np.eye(4)

# Convert the numpy array into a Nifti1Image
img = nib.Nifti1Image(data, affine)

# Optionally, if you want to save this image to a file:
img.to_filename('empty_volume_2d.nii.gz')
