import numpy as np
import nibabel as nib
import sys


resolution = int(sys.argv[1])

# Generate Gaussian noise
data = np.random.randn(resolution, resolution, resolution)

# Normalize the data between 0 and 1
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Create an identity affine matrix. This is a 4x4 matrix that doesn't change
# the spatial coordinates when applied to the data.
affine = np.eye(4)

# Convert the numpy array into a Nifti1Image
img = nib.Nifti1Image(data, affine)

# Optionally, if you want to save this image to a file:
img.to_filename(f'dummy_{resolution}.nii.gz')
