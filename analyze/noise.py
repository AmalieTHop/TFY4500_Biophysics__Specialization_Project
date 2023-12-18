import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

import os
import glob
import nibabel as nib


class Noise:
    def __init__(self, patient_id, mask_name, ymin_homo, xmin_homo, width_homo, height_homo):
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

        self.ymin_homo = ymin_homo
        self.xmin_homo = xmin_homo

        self.width_homo = width_homo
        self.height_homo = height_homo

        self.levels = np.array([0.5])



    def calculate_noise(self):
        snrs_mean_GTV = np.zeros(len(self.bvals))
        for i in range(len(self.bvals)):
            Sb_3d = self.dwi_4d_data[:, :, :, i]
            Sb_homo_roi = Sb_3d[self.xmin_homo:self.xmin_homo+self.width_homo, self.ymin_homo:self.ymin_homo+self.height_homo, self.z]
            std_homo_roi = np.std(Sb_homo_roi)
            
            snr_3d = Sb_3d/std_homo_roi
            snrs_mean_GTV[i] = np.median(snr_3d[self.mask_data == 1]) #np.median(snr_3d[self.mask_data[:, :, self.z] == 1]) #np.mean(snr_3d[self.mask_data[:, :, self.z] == 1])

        return snrs_mean_GTV
    

    
    
    def plot_homo_roi_and_GTV(self):
        sns.set()
        params = {'figure.figsize': (10, 7.5),
                  'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'axes.grid': False,
                  'image.cmap': 'gray',#'turbo'
                  'lines.linewidth' : 1.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'upper right', ###best
                  'legend.framealpha': 0.75,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(params)

        for i, bval in enumerate(self.bvals):
            mask_data_slice = self.mask_data[:, :, self.z]
            dwi_data_slice = self.dwi_4d_data[:, :, self.z, i]
        
            fig, ax = plt.subplots()
            im = ax.imshow(dwi_data_slice.T)
            #cbar = fig.colorbar(im, ax = ax)
            cs = ax.contour(mask_data_slice.T, self.levels, colors='C8')
            rect_dummy = Rectangle((self.xmin_homo-0.5, self.ymin_homo-0.5), width=self.width_homo+1, height=self.height_homo+1, linewidth = 1.5, facecolor='none', edgecolor='C8', label='GTV')
            rect = Rectangle((self.xmin_homo-0.5, self.ymin_homo-0.5), width=self.width_homo+1, height=self.height_homo+1, linewidth = 1.5, facecolor='none', edgecolor='C2', label='HR: Side')
            rect_fig_n = Rectangle((185-0.5, 185-0.5), width=14+1, height=14+1, linewidth = 1.5, facecolor='none', edgecolor='C0', label='HR: Neck')
            rect_fig_m = Rectangle((170-0.5, 80-0.5), width=20+1, height=20+1, linewidth = 1.5, facecolor='none', edgecolor='C1', label='HR: Mouth')
            ax.add_patch(rect_dummy)
            ax.add_patch(rect_fig_n)
            ax.add_patch(rect_fig_m)
            ax.add_patch(rect)


            ax.legend()
            ax.axis("off")
            fig.tight_layout()

            plt.savefig(os.path.join(self.dst_dir, f'b{bval}.pdf'))    



class Noiseplt:
    def __init__(self, src_dir):
        self.dst_dir = src_dir

        self.snrs_median_GTV_patients_s = np.load(os.path.join(src_dir, 'snrs_median_GTV_patients_s.npy')) 
        self.snrs_median_GTV_patients_m = np.load(os.path.join(src_dir, 'snrs_median_GTV_patients_m.npy'))
        self.snrs_median_GTV_patients_n = np.load(os.path.join(src_dir, 'snrs_median_GTV_patients_n.npy'))

        self.bvals = [0, 50, 100, 200, 800]

        self.df_noise = pd.DataFrame(columns=[r'$b$-value [s/mm$^2$]', 'site', 'SNR'])

        for j, bval in enumerate(self.bvals):
            for i in range(len(self.snrs_median_GTV_patients_n)):
                row_n = pd.DataFrame({r'$b$-value [s/mm$^2$]': [bval], 'site': 'Neck', 'SNR': [self.snrs_median_GTV_patients_n[i][j]]})
                self.df_noise = pd.concat([self.df_noise, row_n])
            
            for i in range(len(self.snrs_median_GTV_patients_m)):
                row_m = pd.DataFrame({r'$b$-value [s/mm$^2$]': [bval], 'site': 'Mouth', 'SNR': [self.snrs_median_GTV_patients_m[i][j]]})
                self.df_noise = pd.concat([self.df_noise, row_m])
            
            for i in range(len(self.snrs_median_GTV_patients_s)):
                row_s = pd.DataFrame({r'$b$-value [s/mm$^2$]': [bval], 'site': 'Side', 'SNR': [self.snrs_median_GTV_patients_s[i][j]]})
                self.df_noise = pd.concat([self.df_noise, row_n, row_s])



    def plot_noise(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 20,
                  'legend.loc':'best', ###best
                  'legend.framealpha': 0.75,
                  'savefig.format': 'pdf'
                  }
        boxprops={'alpha': 0.5}
        showfliers=False
        jitter=0.1
        alpha=0.9

        plt.rcParams.update(params)

        fig, ax = plt.subplots(figsize=(10, 7.5))

        ax = sns.boxplot(ax=ax, data=self.df_noise, x=r'$b$-value [s/mm$^2$]', y='SNR', hue='site', showfliers=showfliers, boxprops=boxprops)
        sns.stripplot(data=self.df_noise, x=r'$b$-value [s/mm$^2$]', y='SNR', hue='site', dodge=True, alpha=alpha, jitter=jitter, legend=None, ax=ax)
        ax.legend(title=None)
        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'summary_of_noise_analysis'))    



    
class Noiseplt_old:
    def __init__(self, src_dir):
        self.dst_dir = src_dir

        self.mean_snrs_median_GTV_patients_s = np.load(os.path.join(src_dir, 'mean_snrs_median_GTV_patients_s.npy')) 
        self.mean_snrs_median_GTV_patients_m = np.load(os.path.join(src_dir, 'mean_snrs_median_GTV_patients_m.npy'))
        self.mean_snrs_median_GTV_patients_n = np.load(os.path.join(src_dir, 'mean_snrs_median_GTV_patients_n.npy'))

        self.bvals = [0, 50, 100, 200, 800]

        self.df_noise = pd.DataFrame(columns=[r'$b$-value [s/mm$^2$]', 'site', 'SNR'])

        for i, bval in enumerate(self.bvals):
            row_n = pd.DataFrame({r'$b$-value [s/mm$^2$]': [bval], 'site': 'Neck', 'SNR': [self.mean_snrs_median_GTV_patients_n[i]]})
            row_m = pd.DataFrame({r'$b$-value [s/mm$^2$]': [bval], 'site': 'Mouth', 'SNR': [self.mean_snrs_median_GTV_patients_m[i]]})
            row_s = pd.DataFrame({r'$b$-value [s/mm$^2$]': [bval], 'site': 'Side', 'SNR': [self.mean_snrs_median_GTV_patients_s[i]]})

            self.df_noise = pd.concat([self.df_noise, row_n, row_m, row_s])



    def plot_noise(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 20,
                  'legend.loc':'best', ###best
                  'legend.framealpha': 0.75,
                  'savefig.format': 'pdf'
                  }
        
        plt.rcParams.update(params)

        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax = sns.pointplot(data=self.df_noise, x=r'$b$-value [s/mm$^2$]', y='SNR', hue='site', markers=['o', 'v', 'D'], ax=ax)
        ax.legend(title=None)
        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'summary_of_noise_analysis'))    




        

        

        