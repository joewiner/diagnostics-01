#: import common modules
import numpy as np  # the Python array package
import matplotlib.pyplot as plt  # the Python plotting package
import numpy.linalg as npl
# Display array values to 6 digits of precision
np.set_printoptions(precision=4, suppress=True)
#%matplotlib

#- Change this section to load any image
import nibabel
img = nibabel.load('data/group01_sub01_run1.nii')
#- Get the data array from the image
data = img.get_data()

#- 'vol_shape' is the shape of volumes
vol_shape = data.shape[:-1]
#- 'n_vols' is the number of volumes
n_vols = data.shape[-1]
#- N is the number of voxels in a volume
N = np.prod(vol_shape)

"""
The first part of the code will use PCA to get component matrix U
and scalar projections matrix C
"""

#- Reshape to 2D array that is voxels by volumes (N x n_vols)
X = data.reshape(N, n_vols).T

#- Calculate unscaled covariance matrix for X
unscaled_covariance = X.dot(X.T)

#- Use SVD to return U, S, VT matrices from unscaled covariance
#U, S, VT = npl.svd(unscaled_covariance)
U, S, VT = npl.svd(unscaled_covariance)

#- Calculate the scalar projections for projecting X onto the vectors in U.
#- Put the result into a new array C.
C = U.T.dot(X)

#- Transpose C
#- Reshape the first dimension of C to have the 3D shape of the original data volumes.
C_vols = C.T.reshape(vol_shape + (n_vols,))

"""
The second part of the code determines which voxels are inside the brain
and which are outside the brain and creates a mask (boolean matrix)
"""

#get the mean voxel intensity of entire 4D object
mean_voxel = np.mean(data)
#get the mean volume (3D) across time series (axis 3)
mean_volume = np.mean(data, axis=3)
#boolean mask set to all voxels above .5 in the first volume
#(.125 is the SPM criterion but .5 seems like a better threshold)
mask = mean_volume > (.5 * mean_voxel) #threshold can be adjusted!
out_mask = ~mask

#plot a slice from the data next to slice of map to ensure it is the
#correct shape!
fig, axes = plt.subplots(2, 1)
axes[0].imshow(data[:, :, 14, 0], cmap='gray')
axes[1].imshow(mask[:, :, 14], cmap='gray')

"""
The third part of code finds the root mean square of U from step 1, then uses the
mask from step 2 to determine which components explain data outside the brain
Selects these "bad components" with high "outsideness"
"""

#Apply mask to C matrix to get all voxels outside of brain
outside = C_vols[out_mask]
#Get RMS of the voxels outside, reflecting "outsideness" of this scan
RMS_out = np.sqrt(np.mean((outside ** 2), axis=0))

#Apply mask to C matrix to get all voxels inside brain
inside = C_vols[mask]
#Get RMS of the voxels inside, reflecting "insideness" of this scan
RMS_in = np.sqrt(np.mean((inside ** 2), axis=0))

#The closer this ratio is to 1, the worse the volume
RMS_ratio = RMS_out / RMS_in

#plot the 3 RMS values to see visualize the "outsideness"
fig, axes = plt.subplots(3,1)
axes[0].plot(RMS_out)
axes[1].plot(RMS_in)
axes[2].plot(RMS_ratio)

"""
The fourth part of the code uses the "bad components" to generate a new
"bad data set" and then puts this dataset through the outlier detector
"""

#Create a boolean mask for the 10% worst PCs (meaning highest RMS ratio)
PC_bad = np.percentile(RMS_ratio,90)
PC_bad_mask = RMS_ratio > PC_bad

U_bad = U[:, PC_bad_mask] #selects columns
C_bad = C[PC_bad_mask]

#generates data set based on the bad PCs and (U and C matrices)
X_bad = U_bad.dot(C_bad).T.reshape(vol_shape + (n_vols,))

#Plot a slice from 'real' data compared to 'bad' data
fig, axes = plt.subplots(2, 1)
axes[0].imshow(data[:, :, 16, 0], cmap='gray')
axes[1].imshow(X_bad[:, :, 16, 0], cmap='gray')

#run through outlier detector script
#run scripts/find_outliers(X_bad)
