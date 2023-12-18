import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

import os
import glob
import nibabel as nib

class Noise:
    def __init__(self, patient_id, mask_name, ymin_homo, ymax_homo, xmin_homo, xmax_homo):
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
        
        self.z = np.argmax(np.sum(self.mask_data, axis=(0,1)))
        print(f'Slice num: {self.z}')

        self.ymin_homo = ymin_homo
        self.ymax_homo = ymax_homo
        self.xmin_homo = xmin_homo
        self.xmax_homo = xmax_homo

        self.width_homo = xmax_homo-xmin_homo
        self.height_homo = ymax_homo-ymin_homo

        self.levels = np.array([0.5])




    def calculate_noise(self):
        snrs_mean_GTV = np.zeros(len(self.bvals))
        for i in range(len(self.bvals)):
            Sb_3d = self.dwi_4d_data[:, :, :, i]
            Sb_homo_roi = Sb_3d[self.xmin_homo:self.xmax_homo, self.ymin_homo:self.ymax_homo, self.z]
            std_homo_roi = np.std(Sb_homo_roi)
            
            snr_3d = Sb_3d/std_homo_roi
            snrs_mean_GTV[i] = np.mean(snr_3d[self.mask_data[:, :, self.z] == 1])

        return snrs_mean_GTV
    
    def plot_homo_roi_and_GTV(self):
        sns.set()
        params = {'figure.figsize': (10, 5),
                  'lines.linewidth' : 1.5,
                  'axes.grid': False,
                  'legend.fontsize': 10,
                  }
        plt.rcParams.update(params)

        for i, bval in enumerate(self.bvals):
            mask_data_slice = self.mask_data[:, :, self.z]
            dwi_data_slice = self.dwi_4d_data[:, :, self.z, i]
        
            fig, ax = plt.subplots()
            im = ax.imshow(dwi_data_slice.T, cmap='gray')
            cbar = fig.colorbar(im, ax = ax)
            cs = ax.contour(mask_data_slice.T, self.levels, colors='C1')
            rect_dummy = Rectangle((self.xmin_homo, self.ymin_homo), width=self.width_homo, height=self.height_homo, linewidth = 1.5, facecolor='none', edgecolor='C1', label='GTV')
            rect = Rectangle((self.xmin_homo, self.ymin_homo), width=self.width_homo, height=self.height_homo, linewidth = 1.5, facecolor='none', edgecolor='C0', label='Homogeneous region')
            ax.add_patch(rect_dummy)
            ax.add_patch(rect)
            ax.legend()
            ax.axis("off")
            fig.tight_layout()

            plt.savefig(os.path.join(self.dst_dir, f'b{bval}.png'))    


    





            


        

        

        