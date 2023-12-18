import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


import os
import glob
import nibabel as nib

class Noise:
    def __init__(self, patient_id, mask_name):
        self.patient_id = patient_id

        bvals_txt_fname = f'data/sorted/{patient_id}/bvals.txt'
        self.bvals = np.loadtxt(bvals_txt_fname)

        dwi_4d_fname = glob.glob(f'data/sorted/{patient_id}/multibval_4d/*.nii')[0]
        self.dwi_4d_data = nib.load(dwi_4d_fname).get_fdata()
        self.width, self.height, self.num_slices, self.num_bvals = self.dwi_4d_data.shape

        self.mask_name = mask_name
        mask_fname = f'data/sorted/{patient_id}/resampled_masks/{mask_name}/by_bval/resampled_mask_B_0.nii'
        self.mask_data = nib.load(mask_fname).get_fdata()

        self.dst_dir = f'data/analysis/noise/{patient_id}'
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)
    


        #mask_data_slice = self.mask_data[:, :, 15]
        #signal_data_slice = self.dwi_4d_data[:, :, 15, 0]

        #fig, ax = plt.subplots()
        #im = ax.imshow(signal_data_slice)
        #levels = np.array([0.5])
        #cs = ax.contour(mask_data_slice, levels)

        #plt.show() 



    def calculate_noise(self):
        z=15
        y_min=70
        y_max=90
        x_min=170
        x_max=190

        width = x_max - x_min
        height = y_max - y_min

        S0_3d = self.dwi_4d_data[:, :, :, 0]

        for i, bval in enumerate(self.bvals):
            Sb_3d = self.dwi_4d_data[:, :, :, i]

            Sb_homogenous_region = Sb_3d[x_min:x_max, y_min:y_max, z]
            std_homogenous_region = np.std(Sb_homogenous_region)
            
            snr = Sb_3d/std_homogenous_region
            
            Sb_homogenous_region_mask = np.ones(Sb_3d.shape)*0.75
            Sb_homogenous_region_mask[x_min:x_max, y_min:y_max, z] = 1
            rect = Rectangle((y_min, x_min), width=width, height=height, facecolor='none', edgecolor='b')


            fig, ax = plt.subplots()  

            im = ax.imshow(Sb_3d[:, :, z], alpha=Sb_homogenous_region_mask[:, :, z])
            cbar = fig.colorbar(im, ax = ax)

            ax.add_patch(rect)

            plt.show()     

            mean_snr = np.mean(snr[self.mask_data[:, :, z] == 1])





            


        

        

        