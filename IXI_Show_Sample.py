import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the NIfTI file
nii_file = './IXI-T2/IXI002-Guys-0828-T2.nii.gz'
nii_data = nib.load(nii_file)

# Get the actual image data
image_data = nii_data.get_fdata()

# Get the shape of the data
data_shape = image_data.shape
print("Data shape:", image_data.shape)

num_slices = data_shape[2]
# Create a figure and axis for the animation
fig, ax = plt.subplots()
ax.axis('off')
# Define the update function for each frame of the animation
def update(frame):
    # Clear the axis
    ax.cla()
    # Extract the slice data for the current frame
    slice_data = image_data[:, :, frame]
    # Plot the image
    ax.imshow(slice_data, cmap='gray')
    # Set the title
    ax.set_title(f'MRI Slice {frame+1}/{num_slices}')
# Create the animation
animation = FuncAnimation(fig, update, frames=num_slices, interval=200)
# Display the animation
plt.show()
