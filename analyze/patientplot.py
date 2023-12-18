
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle
from scipy import stats

import seaborn as sns
sns.set_theme()

import os
import copy
import glob

import nibabel as nib


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat='%1.1f', offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        #if self._useMathText:
        #    self.format = r'$\mathdefault{%s}$' % self.format



class PatientpltADC:
    def __init__(self, patient_id, GTV_name):
        self.patient_id = patient_id
        self.GTV_name = GTV_name

        self.src_dir = f'data/results/{patient_id}/{GTV_name}/ADC'
        self.dst_dir = f'data/plots/{patient_id}/{GTV_name}/ADC'
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)


        bvals_txt_fname = f'data/sorted/{self.patient_id}/bvals.txt'
        dwi_4d_fname = glob.glob(f'data/sorted/{self.patient_id}/multibval_4d/*.nii')[0]
        fitted_adc_map_fname = f'{self.src_dir}' + '/ADC.nii'
        fitted_S0_map_fname = f'{self.src_dir}' + '/S0.nii'
        machine_adc_map_fname = glob.glob(f'data/sorted/{patient_id}/machine_adc_map/*.nii')[0]
        mask_fname = f'data/sorted/{patient_id}/resampled_masks/{GTV_name}/by_bval/resampled_mask_B_0.nii'

        self.bvals = np.loadtxt(bvals_txt_fname)
        self.dwi_4d_data = nib.load(dwi_4d_fname).get_fdata()
        self.fitted_adc_map_data = nib.load(fitted_adc_map_fname).get_fdata()
        self.fitted_S0_map_data = nib.load(fitted_S0_map_fname).get_fdata()
        self.machine_adc_map_data = nib.load(machine_adc_map_fname).get_fdata()*10**(-6)
        self.mask_data = nib.load(mask_fname).get_fdata()
        self.mask_bool = self.mask_data.astype(bool)

        self.width, self.height, self.num_slices, self.num_bvals = self.dwi_4d_data.shape

        S0 = np.nanmean(self.dwi_4d_data[:, :, :, self.bvals == 0], axis=3).astype('<f')        
        self.dwi_norm_4d_data = self.dwi_4d_data / S0[:, :, :, None]

        self.dwi_norm_GTV_vxls = self.dwi_norm_4d_data[self.mask_bool]
        self.fitted_adc_GTV_vxls = self.fitted_adc_map_data[self.mask_bool]
        self.fitted_S0_GTV_vxls = self.fitted_S0_map_data[self.mask_bool] #/ S0[self.mask_bool]
        self.machine_adc_GTV_vxls = self.machine_adc_map_data[self.mask_bool]
        self.relative_error_GTV_vxls = self.compute_relative_error_between_fitted_and_machine()
        self.rmse_fitted_adc_GTV_vxls = self.compute_rmse_of_fit(self.bvals, self.fitted_S0_GTV_vxls, self.fitted_adc_GTV_vxls)
        self.rmse_machine_adc_GTV_vxls = self.compute_rmse_of_fit(self.bvals, np.ones(self.fitted_S0_GTV_vxls.shape), self.machine_adc_GTV_vxls)



    def compute_relative_error_between_fitted_and_machine(self):

        absolute_error_GTV_vxls = abs(self.fitted_adc_GTV_vxls - self.machine_adc_GTV_vxls)
        relative_error_GTV_vxls = absolute_error_GTV_vxls / self.fitted_adc_GTV_vxls * 100
        return relative_error_GTV_vxls


    def adc_normed(self, bvals, S0, adc):
        return S0 * np.exp(-bvals*adc)


    def adcs_normed(self, bvals, S0, adc):
        num_adcs = adc.size
        S0_1d = S0.reshape(num_adcs)
        adc_1d = adc.reshape(num_adcs)

        adcs_1d = self.adc_normed(np.tile(np.expand_dims(bvals, axis=0), (num_adcs, 1)),
                                  np.tile(np.expand_dims(S0_1d, axis=1), (1, len(bvals))),
                                  np.tile(np.expand_dims(adc_1d, axis=1), (1, len(bvals)))).astype('f')
        return adcs_1d


    def rmse(self, dwi, Sb):
        rmse = np.sqrt(np.mean(np.square(dwi - Sb), axis=1)) #np.mean(abs(dwi - Sb), axis=1)
        return rmse


    def compute_rmse_of_fit(self, bvals, S0_GTV_vxls, adc_GTV_vxls):
        Sb_GTV_vxls = self.adcs_normed(self.bvals, S0_GTV_vxls, adc_GTV_vxls)
        rmse_GTV_vxls = self.rmse(self.dwi_norm_GTV_vxls, Sb_GTV_vxls)
        return rmse_GTV_vxls

    
    def plt_everything(self):
        # preparation for plot
        z = np.argmax(np.sum(self.mask_data, axis=(0,1)))
        mask_slice = self.mask_data[:, :, z]

        x_indecies, y_indecies = np.where(mask_slice == 1)
        x_min, x_max = np.min(x_indecies), np.max(x_indecies)
        y_min, y_max = np.min(y_indecies), np.max(y_indecies)

        
        #dwi_norm_GTV = np.zeros((self.width, self.height, self.num_slices)) #not really necessary
        fitted_adc = np.zeros((self.width, self.height, self.num_slices))
        fitted_S0 = np.zeros((self.width, self.height, self.num_slices))
        machine_adc = np.zeros((self.width, self.height, self.num_slices))
        relative_error = np.zeros((self.width, self.height, self.num_slices))
        rmse_fitted_adc = np.zeros((self.width, self.height, self.num_slices))
        rmse_machine_adc = np.zeros((self.width, self.height, self.num_slices))

        #dwi_norm_GTV[self.mask_bool] = self.dwi_norm_GTV_vxls #not really necessary
        fitted_adc[self.mask_bool] = self.fitted_adc_GTV_vxls
        fitted_S0[self.mask_bool] = self.fitted_S0_GTV_vxls
        machine_adc[self.mask_bool] = self.machine_adc_GTV_vxls
        relative_error[self.mask_bool] = self.relative_error_GTV_vxls
        rmse_fitted_adc[self.mask_bool] = self.rmse_fitted_adc_GTV_vxls
        rmse_machine_adc[self.mask_bool] = self.rmse_machine_adc_GTV_vxls 

        dwi_norm_GTV_slice = self.dwi_norm_4d_data[x_min:x_max, y_min:y_max, z]
        fitted_adc_slice = fitted_adc[x_min:x_max, y_min:y_max, z]
        fitted_S0_slice = fitted_S0[x_min:x_max, y_min:y_max, z]
        machine_adc_slice = machine_adc[x_min:x_max, y_min:y_max, z]
        relative_error_slice = relative_error[x_min:x_max, y_min:y_max, z]
        rmse_fitted_adc_slice = rmse_fitted_adc[x_min:x_max, y_min:y_max, z]
        rmse_machine_adc_slice = rmse_machine_adc[x_min:x_max, y_min:y_max, z]

        plt_b_vals = np.linspace(0, 800, 801)
        idx = np.unravel_index(relative_error_slice.argmax(), relative_error_slice.shape)
        plt_fitted_Sb = self.adc_normed(plt_b_vals, fitted_S0_slice[idx], fitted_adc_slice[idx])
        plt_machine_Sb = self.adc_normed(plt_b_vals, np.ones(fitted_S0_slice[idx].shape), fitted_adc_slice[idx])


        # plot parameters
        sns.set()
        sns.set_style("darkgrid")
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'axes.grid': False,
                  'lines.markersize': 7.5,
                  'lines.linewidth': 3,
                  'image.cmap': 'turbo',
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'upper right', ###best
                  'legend.framealpha': 0.75,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(params)
        cbformat = OOMFormatter(-3, "%1.1f")
        alpha = self.mask_data[x_min:x_max, y_min:y_max, z]

        vmin_adc = np.min([np.min(fitted_adc_slice[fitted_adc_slice > 0]), np.min(machine_adc_slice[machine_adc_slice > 0])])
        vmax_adc = np.max([np.max(fitted_adc_slice), np.max(machine_adc_slice)])
        vmin_error = 0
        vmax_error = np.max(relative_error_slice)
        vmin_rmse = 0
        vmax_rmse = np.max([np.max(rmse_fitted_adc_slice), np.max(rmse_machine_adc_slice)])

        lw = 5
        size_increase = 0.1 + 2*lw/100
        trans = 0.5 + size_increase


        # plot
        fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize=(15, 15))
        [(ax00, ax01), (ax10, ax11), (ax20, ax21)] = axes

        """
        extend=None
        im00 = ax00.imshow(fitted_adc_slice, vmin=vmin_adc, vmax=vmax_adc, alpha=alpha)
        im10 = ax10.imshow(machine_adc_slice, vmin=vmin_adc, vmax=vmax_adc, alpha=alpha)
        im20 = ax20.imshow(relative_error_slice)
        im01 = ax01.imshow(rmse_fitted_adc_slice, vmin=vmin_rmse, vmax=vmax_rmse, alpha=alpha)
        im11 = ax11.imshow(rmse_machine_adc_slice, vmin=vmin_rmse, vmax=vmax_rmse, alpha=alpha)
        """
        """
        #EMIN_1064
        extend='max'
        im00 = ax00.imshow(fitted_adc_slice, alpha=alpha, vmin=vmin_adc, vmax=0.0016)
        im10 = ax10.imshow(machine_adc_slice, alpha=alpha, vmin=vmin_adc, vmax=0.0016)
        im20 = ax20.imshow(relative_error_slice, alpha=alpha)
        im01 = ax01.imshow(rmse_fitted_adc_slice, alpha=alpha, vmax=0.1)
        im11 = ax11.imshow(rmse_machine_adc_slice, alpha=alpha, vmax=0.1)
        """
    
        #EMIN_1011/EMIN_1092
        extend='max'
        im00 = ax00.imshow(fitted_adc_slice, alpha=alpha, vmin=vmin_adc)
        im10 = ax10.imshow(machine_adc_slice, alpha=alpha, vmin=vmin_adc)
        im20 = ax20.imshow(relative_error_slice, alpha=alpha)
        im01 = ax01.imshow(rmse_fitted_adc_slice, alpha=alpha)
        im11 = ax11.imshow(rmse_machine_adc_slice, alpha=alpha)
        
        
        ax00.add_patch(Rectangle((idx[1]-trans, idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        ax10.add_patch(Rectangle((idx[1]-trans, idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        ax20.add_patch(Rectangle((idx[1]-trans, idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        ax01.add_patch(Rectangle((idx[1]-trans, idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        ax11.add_patch(Rectangle((idx[1]-trans, idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        
        ax21.plot(plt_b_vals, plt_fitted_Sb, label = "Fit")
        ax21.plot(plt_b_vals, plt_machine_Sb, label = "Scanner")
        ax21.scatter(self.bvals, dwi_norm_GTV_slice[idx], label='Measured data', color='C2', marker='o')
        ax21.set_ylabel(r'$S_{norm}(b)$')
        ax21.set_xlabel(r'$b$-value [s/mm$^2$]')
        ax21.grid(True)
        ax21.legend()

        for ax in axes.reshape(-1)[:-1]:
            ax.axis("off")

        cbar00 = fig.colorbar(im00, ax = ax00, format=cbformat) #extend=extend
        cbar00.set_label(label=r'$ADC_{fit}$ [mm$^2$/s]')

        cbar10 = fig.colorbar(im10, ax = ax10, format=cbformat) #extend=extend
        cbar10.set_label(label=r'$ADC_{scanner}$ [mm$^2$/s]')

        cbar20 = fig.colorbar(im20, ax = ax20)
        cbar20.set_label(label='Relative error [%]')

        cbar01 = fig.colorbar(im01, ax = ax01)
        cbar01.set_label(label=r'$RMSE_{fit}$')

        cbar11 = fig.colorbar(im11, ax = ax11)
        cbar11.set_label(label=r'$RMSE_{scanner}$')

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, 'ADC'))
        

    
    def plt_medfys2024(self):
        # preparation for plot
        z = np.argmax(np.sum(self.mask_data, axis=(0,1)))
        mask_slice = self.mask_data[:, :, z]

        x_indecies, y_indecies = np.where(mask_slice == 1)
        x_min, x_max = np.min(x_indecies), np.max(x_indecies)
        y_min, y_max = np.min(y_indecies), np.max(y_indecies)

        
        fitted_adc = np.zeros((self.width, self.height, self.num_slices))
        fitted_S0 = np.zeros((self.width, self.height, self.num_slices))
        rmse_fitted_adc = np.zeros((self.width, self.height, self.num_slices))

        fitted_adc[self.mask_bool] = self.fitted_adc_GTV_vxls
        fitted_S0[self.mask_bool] = self.fitted_S0_GTV_vxls
        rmse_fitted_adc[self.mask_bool] = self.rmse_fitted_adc_GTV_vxls


        dwi_norm_GTV_slice = self.dwi_norm_4d_data[x_min:x_max, y_min:y_max, z]
        fitted_adc_slice = fitted_adc[x_min:x_max, y_min:y_max, z]
        fitted_S0_slice = fitted_S0[x_min:x_max, y_min:y_max, z]
        rmse_fitted_adc_slice = rmse_fitted_adc[x_min:x_max, y_min:y_max, z]


        # plot parameters
        sns.set()
        sns.set_style("darkgrid")
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'axes.grid': False,
                  'lines.markersize': 7.5,
                  'lines.linewidth': 3,
                  'image.cmap': 'turbo',
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'upper right', ###best
                  'legend.framealpha': 0.75,
                  'savefig.format': 'svg'
                  }
        plt.rcParams.update(params)
        cbformat = OOMFormatter(-3, "%1.1f")
        alpha = self.mask_data[x_min:x_max, y_min:y_max, z]

        vmin_adc = np.min([np.min(fitted_adc_slice[fitted_adc_slice > 0])])

        # plot
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 5))
        [(ax00), (ax01)] = axes

        #EMIN_1064
        extend='max'
        im00 = ax00.imshow(fitted_adc_slice, alpha=alpha, vmin=vmin_adc, vmax=0.0016)
        im01 = ax01.imshow(rmse_fitted_adc_slice, alpha=alpha, vmax=0.08)

        for ax in axes.reshape(-1):
            ax.axis("off")

        cbar00 = fig.colorbar(im00, ax = ax00, format=cbformat, extend=extend)
        cbar00.set_label(label=r'$ADC$ [mm$^2$/s]')

        cbar01 = fig.colorbar(im01, ax = ax01, extend=extend)
        cbar01.set_label(label=r'$RMSE$')

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, 'ADC_MedFys2024.svg'))
        





class PatientpltIVIM:
    def __init__(self, patient_id, GTV_name):
        self.patient_id = patient_id
        self.GTV_name = GTV_name
        self.src_dir = f'data/results/{patient_id}/{GTV_name}/IVIM'
        self.src_dir_net = f'data/results/{patient_id}/{GTV_name}/IVIM/net'#f'data/results/{patient_id}'
        self.dst_dir = f'data/plots/{patient_id}/{GTV_name}/IVIM'
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)
        
        self.bvals = None

        self.z = None
        self.mask_slice = None
        self.mask_slice_dim_mask = None

        self.Dt_conv_mask_vxls, self.Fp_conv_mask_vxls, self.Dp_conv_mask_vxls = None, None, None
        self.Dt_seg_mask_vxls, self.Fp_seg_mask_vxls, self.Dp_seg_mask_vxls = None, None, None
        self.Dt_net_mask_vxls, self.Fp_net_mask_vxls, self.Dp_net_mask_vxls = None, None, None

        self.Dt_conv_dim_mask, self.Fp_conv_dim_mask, self.Dp_conv_dim_mask = None, None, None
        self.Dt_seg_dim_mask, self.Fp_seg_dim_mask, self.Dp_seg_dim_mask = None, None, None
        self.Dt_net_dim_mask, self.Fp_net_dim_mask, self.Dp_net_dim_mask = None, None, None

        self.shape_to_mask()

        self.idx = np.unravel_index(self.Dp_conv_dim_mask.argmax(), self.Dp_conv_dim_mask.shape)
        

    def ivim(self, bvals, Dt, Fp, Dp, S0):
        return (S0 * (Fp * np.exp(-bvals * Dp) + (1 - Fp) * np.exp(-bvals * Dt)))


    def ivims(self, bvals, Dt, Fp, Dp, S0):
        num_ivims = Dt.size
        Dt_1d, Fp_1d, Dp_1d, S0_1d = Dt.reshape(num_ivims), Fp.reshape(num_ivims), Dp.reshape(num_ivims), S0.reshape(num_ivims)

        ivims_1d = self.ivim(np.tile(np.expand_dims(bvals, axis=0), (num_ivims, 1)),
                          np.tile(np.expand_dims(Dt_1d, axis=1), (1, len(bvals))),
                          np.tile(np.expand_dims(Fp_1d, axis=1), (1, len(bvals))),
                          np.tile(np.expand_dims(Dp_1d, axis=1), (1, len(bvals))),
                          np.tile(np.expand_dims(S0_1d, axis=1), (1, len(bvals)))).astype('f')

        ivims = ivims_1d.reshape((Dt.shape[0], Dt.shape[1], Dt.shape[2], len(bvals)))
        return ivims


    def rmse(self, dwi, Sb):
        rmse = np.sqrt(np.mean(np.square(dwi - Sb), axis=3))
        return rmse

        
    def shape_to_mask(self):
        dwi_4d_fname = glob.glob(f'data/sorted/{self.patient_id}/multibval_4d/*.nii')[0]
        bvals_txt_fname = f'data/sorted/{self.patient_id}/bvals.txt'
        mask_fname = f'data/sorted/{self.patient_id}/resampled_masks/{self.GTV_name}/by_bval/resampled_mask_B_0.nii'

        dwi_4d_data = nib.load(dwi_4d_fname).get_fdata()
        self.bvals = np.loadtxt(bvals_txt_fname)
        mask_data = nib.load(mask_fname).get_fdata()

        S0 = np.nanmean(dwi_4d_data[:, :, :, self.bvals == 0], axis=3).astype('<f')        
        dwi_norm_4d_data = dwi_4d_data / S0[:, :, :, None]

    
        self.z = np.argmax(np.sum(mask_data, axis=(0,1)))
        mask_slice = mask_data[:, :, self.z]

        x_indecies, y_indecies = np.where(mask_slice == 1)
        x_min, x_max = np.min(x_indecies), np.max(x_indecies)
        y_min, y_max = np.min(y_indecies), np.max(y_indecies)


        Dt_conv = nib.load(os.path.join(self.src_dir, 'conv/Dt.nii')).get_fdata()
        Fp_conv = nib.load(os.path.join(self.src_dir, 'conv/Fp.nii')).get_fdata()
        Dp_conv = nib.load(os.path.join(self.src_dir, 'conv/Dp.nii')).get_fdata()
        S0_conv = nib.load(os.path.join(self.src_dir, 'conv/S0.nii')).get_fdata()
        Dt_seg = nib.load(os.path.join(self.src_dir, 'seg/Dt.nii')).get_fdata()
        Fp_seg = nib.load(os.path.join(self.src_dir, 'seg/Fp.nii')).get_fdata() 
        Dp_seg = nib.load(os.path.join(self.src_dir, 'seg/Dp.nii')).get_fdata()
        S0_seg = np.ones(np.shape(Dt_conv)) #change
        Dt_net = nib.load(os.path.join(self.src_dir_net, f'Dt_net.nii.gz')).get_fdata() #should change filename
        Fp_net = nib.load(os.path.join(self.src_dir_net, f'Fp_net.nii.gz')).get_fdata()
        Dp_net = nib.load(os.path.join(self.src_dir_net, f'Dp_net.nii.gz')).get_fdata()
        S0_net = nib.load(os.path.join(self.src_dir_net, f'S0_net.nii.gz')).get_fdata()

        Sb_conv = self.ivims(self.bvals, Dt_conv, Fp_conv, Dp_conv, S0_conv)
        Sb_seg = self.ivims(self.bvals, Dt_seg, Fp_seg, Dp_seg, S0_seg)
        Sb_net = self.ivims(self.bvals, Dt_net, Fp_net, Dp_net, S0_net)

        rmse_conv = self.rmse(dwi_norm_4d_data, Sb_conv)
        rmse_seg = self.rmse(dwi_norm_4d_data, Sb_seg)
        rmse_net = self.rmse(dwi_norm_4d_data, Sb_net)


        dwi_norm_4d_data[mask_data == 0] = 0
        Dt_conv[mask_data == 0] = 0
        Fp_conv[mask_data == 0] = 0
        Dp_conv[mask_data == 0] = 0
        Dt_seg[mask_data == 0] = 0
        Fp_seg[mask_data == 0] = 0
        Dp_seg[mask_data == 0] = 0
        Dt_net[mask_data == 0] = 0
        Fp_net[mask_data == 0] = 0
        Dp_net[mask_data == 0] = 0
        S0_conv[mask_data == 0] = 0
        S0_seg[mask_data == 0] = 0
        S0_net[mask_data == 0] = 0
        Sb_net[mask_data == 0] = 0

        rmse_conv[mask_data == 0] = 0
        rmse_seg[mask_data == 0] = 0
        rmse_net[mask_data == 0] = 0


        self.dwi_mask_vxls = dwi_norm_4d_data[mask_data != 0]
        self.Dt_conv_mask_vxls = Dt_conv[mask_data != 0]
        self.Fp_conv_mask_vxls = Fp_conv[mask_data != 0]
        self.Dp_conv_mask_vxls = Dp_conv[mask_data != 0]
        self.Dt_seg_mask_vxls = Dt_seg[mask_data != 0]
        self.Fp_seg_mask_vxls = Fp_seg[mask_data != 0]
        self.Dp_seg_mask_vxls = Dp_seg[mask_data != 0]
        self.Dt_net_mask_vxls = Dt_net[mask_data != 0]
        self.Fp_net_mask_vxls = Fp_net[mask_data != 0]
        self.Dp_net_mask_vxls = Dp_net[mask_data != 0]
        self.S0_conv_mask_vxls = S0_conv[mask_data != 0]
        self.S0_seg_mask_vxls = S0_seg[mask_data != 0]
        self.S0_net_mask_vxls = S0_net[mask_data != 0]
        self.Sb_net_mask_vxls = Sb_net[mask_data != 0]
        self.rmse_conv_mask_vxls = rmse_conv[mask_data != 0]
        self.rmse_seg_mask_vxls = rmse_seg[mask_data != 0]
        self.rmse_net_mask_vxls = rmse_net[mask_data != 0]

        self.dwi_dim_mask = dwi_norm_4d_data[x_min:x_max, y_min:y_max, self.z]
        self.Dt_conv_dim_mask = Dt_conv[x_min:x_max, y_min:y_max, self.z]
        self.Fp_conv_dim_mask = Fp_conv[x_min:x_max, y_min:y_max, self.z]
        self.Dp_conv_dim_mask = Dp_conv[x_min:x_max, y_min:y_max, self.z]
        self.Dt_seg_dim_mask = Dt_seg[x_min:x_max, y_min:y_max, self.z]
        self.Fp_seg_dim_mask = Fp_seg[x_min:x_max, y_min:y_max, self.z]
        self.Dp_seg_dim_mask = Dp_seg[x_min:x_max, y_min:y_max, self.z]
        self.Dt_net_dim_mask = Dt_net[x_min:x_max, y_min:y_max, self.z]
        self.Fp_net_dim_mask = Fp_net[x_min:x_max, y_min:y_max, self.z]
        self.Dp_net_dim_mask = Dp_net[x_min:x_max, y_min:y_max, self.z]
        self.S0_conv_dim_mask = S0_conv[x_min:x_max, y_min:y_max, self.z]
        self.S0_seg_dim_mask = S0_seg[x_min:x_max, y_min:y_max, self.z]
        self.S0_net_dim_mask = S0_net[x_min:x_max, y_min:y_max, self.z]
        self.Sb_net_dim_mask = Sb_net[x_min:x_max, y_min:y_max, self.z]
        self.rmse_cov_dim_mask = rmse_conv[x_min:x_max, y_min:y_max, self.z]
        self.rmse_seg_dim_mask = rmse_seg[x_min:x_max, y_min:y_max, self.z]
        self.rmse_net_dim_mask = rmse_net[x_min:x_max, y_min:y_max, self.z]

        self.mask_slice_dim_mask = mask_data[x_min:x_max, y_min:y_max, self.z]
    

    def plt_masked_param_map(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'axes.grid': False,
                  'image.cmap': 'turbo',
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'savefig.format': 'svg'
                  }
        plt.rcParams.update(params)

        alpha = self.mask_slice_dim_mask
        cbformat = OOMFormatter(-3, "%1.1f")
        
        lw = 3
        size_increase = 0.1 + 2*lw/100
        trans = 0.5 + size_increase
        
        """
        Dt_min = np.min([self.Dt_conv_dim_mask[self.mask_slice_dim_mask != 0],
                        self.Dt_seg_dim_mask[self.mask_slice_dim_mask != 0], 
                        self.Dt_net_dim_mask[self.mask_slice_dim_mask != 0]])
        Dt_max = np.max([self.Dt_conv_dim_mask[self.mask_slice_dim_mask != 0],
                        self.Dt_seg_dim_mask[self.mask_slice_dim_mask != 0], 
                        self.Dt_net_dim_mask[self.mask_slice_dim_mask != 0]])
        Fp_min = np.min([self.Fp_conv_dim_mask[self.mask_slice_dim_mask != 0],
                        self.Fp_seg_dim_mask[self.mask_slice_dim_mask != 0], 
                        self.Fp_net_dim_mask[self.mask_slice_dim_mask != 0]])*100
        Fp_max = np.max([self.Fp_conv_dim_mask[self.mask_slice_dim_mask != 0],
                        self.Fp_seg_dim_mask[self.mask_slice_dim_mask != 0], 
                        self.Fp_net_dim_mask[self.mask_slice_dim_mask != 0]])*100
        Dp_min = np.min([self.Dp_conv_dim_mask[self.mask_slice_dim_mask != 0],
                        self.Dp_seg_dim_mask[self.mask_slice_dim_mask != 0], 
                        self.Dp_net_dim_mask[self.mask_slice_dim_mask != 0]])
        Dp_max = np.max([self.Dp_conv_dim_mask[self.mask_slice_dim_mask != 0],
                        self.Dp_seg_dim_mask[self.mask_slice_dim_mask != 0], 
                        self.Dp_net_dim_mask[self.mask_slice_dim_mask != 0]])
        """

        fig, axes = plt.subplots(nrows = 4, ncols = 3, figsize=(15, 15))
        [(ax00, ax01, ax02), (ax10, ax11, ax12), (ax20, ax21, ax22), (ax30, ax31, ax32)] = axes
        #fig.suptitle(f'Parameter maps for {self.patient_id}', fontsize=20)

        """
        # DEFAULT, UNSCALED
        extend_max = None
        extend_min = None
        extend_both = None
        im00 = ax00.imshow(self.Dt_conv_dim_mask, alpha=alpha, vmin=np.min(self.Dt_conv_dim_mask[self.mask_slice_dim_mask != 0]), vmax=np.max(self.Dt_conv_dim_mask[self.mask_slice_dim_mask != 0]))
        im01 = ax01.imshow(self.Dt_seg_dim_mask, alpha=alpha, vmin=np.min(self.Dt_seg_dim_mask[self.mask_slice_dim_mask != 0]), vmax=np.max(self.Dt_seg_dim_mask[self.mask_slice_dim_mask != 0]))
        im02 = ax02.imshow(self.Dt_net_dim_mask, alpha=alpha, vmin=np.min(self.Dt_net_dim_mask[self.mask_slice_dim_mask != 0]), vmax=np.max(self.Dt_net_dim_mask[self.mask_slice_dim_mask != 0]))
        im10 = ax10.imshow(self.Fp_conv_dim_mask*100, alpha=alpha)
        im11 = ax11.imshow(self.Fp_seg_dim_mask*100, alpha=alpha)
        im12 = ax12.imshow(self.Fp_net_dim_mask*100, alpha=alpha)
        im20 = ax20.imshow(self.Dp_conv_dim_mask, alpha=alpha, vmax=np.max(self.Dp_conv_dim_mask[self.mask_slice_dim_mask != 0]))
        im21 = ax21.imshow(self.Dp_seg_dim_mask, alpha=alpha, vmax=np.max(self.Dp_seg_dim_mask[self.mask_slice_dim_mask != 0]))
        im22 = ax22.imshow(self.Dp_net_dim_mask, alpha=alpha, vmin=np.min(self.Dp_net_dim_mask[self.mask_slice_dim_mask != 0]), vmax=np.max(self.Dp_net_dim_mask[self.mask_slice_dim_mask != 0]))
        im30 = ax30.imshow(self.rmse_cov_dim_mask, alpha=alpha)
        im31 = ax31.imshow(self.rmse_seg_dim_mask, alpha=alpha)
        im32 = ax32.imshow(self.rmse_net_dim_mask, alpha=alpha)
        """

        """
        # EMIN_1064 SCALED
        extend_max = 'max'
        extend_min = 'min'
        im00 = ax00.imshow(self.Dt_conv_dim_mask, alpha=alpha, vmin=0.0005, vmax=np.max(self.Dt_conv_dim_mask[self.mask_slice_dim_mask != 0]))
        im01 = ax01.imshow(self.Dt_seg_dim_mask, alpha=alpha, vmin=0.0005, vmax=np.max(self.Dt_seg_dim_mask[self.mask_slice_dim_mask != 0]))
        im02 = ax02.imshow(self.Dt_net_dim_mask, alpha=alpha, vmin=0.0005, vmax=np.max(self.Dt_net_dim_mask[self.mask_slice_dim_mask != 0]))
        im10 = ax10.imshow(self.Fp_conv_dim_mask*100, alpha=alpha, vmax = 25)
        im11 = ax11.imshow(self.Fp_seg_dim_mask*100, alpha=alpha, vmax = 25)
        im12 = ax12.imshow(self.Fp_net_dim_mask*100, alpha=alpha, vmax = 25)
        im20 = ax20.imshow(self.Dp_conv_dim_mask, alpha=alpha, vmax=0.07)
        im21 = ax21.imshow(self.Dp_seg_dim_mask, alpha=alpha, vmax=0.07)
        im22 = ax22.imshow(self.Dp_net_dim_mask, alpha=alpha, vmin=0.01, vmax=np.max(self.Dp_net_dim_mask[self.mask_slice_dim_mask != 0]))
        im30 = ax30.imshow(self.rmse_cov_dim_mask, alpha=alpha, vmax=0.05)
        im31 = ax31.imshow(self.rmse_seg_dim_mask, alpha=alpha, vmax=0.05)
        im32 = ax32.imshow(self.rmse_net_dim_mask, alpha=alpha, vmax=0.05)
        """

        
        # EMIN_1011 SCALED
        extend_max = 'max'
        extend_min = 'min'
        im00 = ax00.imshow(self.Dt_conv_dim_mask, alpha=alpha, vmin=np.min(self.Dt_conv_dim_mask[self.mask_slice_dim_mask != 0]))
        im01 = ax01.imshow(self.Dt_seg_dim_mask, alpha=alpha, vmin=np.min(self.Dt_seg_dim_mask[self.mask_slice_dim_mask != 0]))
        im02 = ax02.imshow(self.Dt_net_dim_mask, alpha=alpha, vmin=np.min(self.Dt_net_dim_mask[self.mask_slice_dim_mask != 0]))
        im10 = ax10.imshow(self.Fp_conv_dim_mask*100, alpha=alpha, vmax = 25)
        im11 = ax11.imshow(self.Fp_seg_dim_mask*100, alpha=alpha, vmax = 25)
        im12 = ax12.imshow(self.Fp_net_dim_mask*100, alpha=alpha, vmax = 25)
        im20 = ax20.imshow(self.Dp_conv_dim_mask, alpha=alpha, vmax=0.125)
        im21 = ax21.imshow(self.Dp_seg_dim_mask, alpha=alpha, vmax=0.125)
        im22 = ax22.imshow(self.Dp_net_dim_mask, alpha=alpha, vmax=0.025)
        im30 = ax30.imshow(self.rmse_cov_dim_mask, alpha=alpha, vmax=0.15)
        im31 = ax31.imshow(self.rmse_seg_dim_mask, alpha=alpha, vmax=0.15)
        im32 = ax32.imshow(self.rmse_net_dim_mask, alpha=alpha, vmax=0.15)
        

        """
        # EMIN_1092 SCALED
        extend_max = 'max'
        extend_min = 'min'
        im00 = ax00.imshow(self.Dt_conv_dim_mask, alpha=alpha, vmin=np.min(self.Dt_conv_dim_mask[self.mask_slice_dim_mask != 0]))
        im01 = ax01.imshow(self.Dt_seg_dim_mask, alpha=alpha, vmin=np.min(self.Dt_seg_dim_mask[self.mask_slice_dim_mask != 0]))
        im02 = ax02.imshow(self.Dt_net_dim_mask, alpha=alpha, vmin=np.min(self.Dt_net_dim_mask[self.mask_slice_dim_mask != 0]))
        im10 = ax10.imshow(self.Fp_conv_dim_mask*100, alpha=alpha, vmax = 40)
        im11 = ax11.imshow(self.Fp_seg_dim_mask*100, alpha=alpha, vmax = 40)
        im12 = ax12.imshow(self.Fp_net_dim_mask*100, alpha=alpha, vmax = 40)
        im20 = ax20.imshow(self.Dp_conv_dim_mask, alpha=alpha, vmax=0.1)
        im21 = ax21.imshow(self.Dp_seg_dim_mask, alpha=alpha, vmax=0.1)
        im22 = ax22.imshow(self.Dp_net_dim_mask, alpha=alpha, vmax=0.1)
        im30 = ax30.imshow(self.rmse_cov_dim_mask, alpha=alpha, vmax=0.06)
        im31 = ax31.imshow(self.rmse_seg_dim_mask, alpha=alpha, vmax=0.06)
        im32 = ax32.imshow(self.rmse_net_dim_mask, alpha=alpha, vmax=0.06)
        """


        #ax00.add_patch(Rectangle((self.idx[1]-trans, self.idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        #ax01.add_patch(Rectangle((self.idx[1]-trans, self.idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        #ax02.add_patch(Rectangle((self.idx[1]-trans, self.idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        #ax10.add_patch(Rectangle((self.idx[1]-trans, self.idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        #ax11.add_patch(Rectangle((self.idx[1]-trans, self.idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        #ax12.add_patch(Rectangle((self.idx[1]-trans, self.idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        #ax20.add_patch(Rectangle((self.idx[1]-trans, self.idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        #ax21.add_patch(Rectangle((self.idx[1]-trans, self.idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        #ax22.add_patch(Rectangle((self.idx[1]-trans, self.idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        #ax30.add_patch(Rectangle((self.idx[1]-trans, self.idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        #ax31.add_patch(Rectangle((self.idx[1]-trans, self.idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))
        #ax32.add_patch(Rectangle((self.idx[1]-trans, self.idx[0]-trans), width=1+size_increase, height=1+size_increase, linewidth = lw, facecolor='none', edgecolor='white'))

        for ax in axes:
            ax[0].set(title='LSQ')
            ax[1].set(title='SEG')
            ax[2].set(title='DNN')

        cbar00 = fig.colorbar(im00, ax = ax00, format=cbformat, extend=extend_min)
        cbar00.set_label(label=r'$D_t$ [mm$^2$/s]')
        cbar01 = fig.colorbar(im01, ax = ax01, format=cbformat, extend=extend_min)
        cbar01.set_label(label=r'$D_t$ [mm$^2$/s]')
        cbar02 = fig.colorbar(im02, ax = ax02, format=cbformat)
        cbar02.set_label(label=r'$D_t$ [mm$^2$/s]')
        cbar10 = fig.colorbar(im10, ax = ax10, extend=extend_max)
        cbar10.set_label(label=r'$f_p$ [%]')
        cbar11 = fig.colorbar(im11, ax = ax11, extend=extend_max)
        cbar11.set_label(label=r'$f_p$ [%]')
        cbar12 = fig.colorbar(im12, ax = ax12, extend=extend_max)
        cbar12.set_label(label=r'$f_p$ [%]')
        cbar20 = fig.colorbar(im20, ax = ax20, format=cbformat, extend=extend_max)
        cbar20.set_label(label=r'$D_p$ [mm$^2$/s]')
        cbar21 = fig.colorbar(im21, ax = ax21, format=cbformat, extend=extend_max)
        cbar21.set_label(label=r'$D_p$ [mm$^2$/s]')
        cbar22 = fig.colorbar(im22, ax = ax22, format=cbformat, extend=extend_max)
        cbar22.set_label(label=r'$D_p$ [mm$^2$/s]')
        cbar30 = fig.colorbar(im30, ax = ax30, extend=extend_max)
        cbar30.set_label(label=r'RMSE')
        cbar31 = fig.colorbar(im31, ax = ax31, extend=extend_max)
        cbar31.set_label(label=r'RMSE')
        cbar32 = fig.colorbar(im32, ax = ax32, extend=extend_max)
        cbar32.set_label(label=r'RMSE')

        for ax in axes.reshape(-1):
            ax.axis("off")

        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'param_map_{self.patient_id}'))        




    def plt_signal_curves(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'upper right',
                  'legend.framealpha': 0.75,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(params)
        save_name = f'IVIM_signal_curves_{self.patient_id}'
        plot_b_vals = np.linspace(0, 800, 801)

        Sb_conv = self.ivim(plot_b_vals, self.Dt_conv_dim_mask[self.idx], self.Fp_conv_dim_mask[self.idx], self.Dp_conv_dim_mask[self.idx], self.S0_conv_dim_mask[self.idx])
        Sb_seg = self.ivim(plot_b_vals, self.Dt_seg_dim_mask[self.idx], self.Fp_seg_dim_mask[self.idx], self.Dp_seg_dim_mask[self.idx], self.S0_seg_dim_mask[self.idx])
        Sb_net = self.ivim(plot_b_vals, self.Dt_net_dim_mask[self.idx], self.Fp_net_dim_mask[self.idx], self.Dp_net_dim_mask[self.idx], self.S0_net_dim_mask[self.idx])
                
        plt.figure(figsize = [10,5])
        plt.plot(plot_b_vals, Sb_conv, label = "LSQ", color='C0')
        plt.plot(plot_b_vals, Sb_seg, label = "SEG", color='C1')
        plt.plot(plot_b_vals, Sb_net, label = "DNN", color='C2')
        plt.scatter(self.bvals, self.dwi_dim_mask[self.idx], label = "Measured data", color='C8')
        plt.xlabel(r'$b$-values')
        plt.ylabel(r'$S_{norm}(b)$')
        plt.yticks()
        plt.xticks()
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, save_name))



    def plt_param_space(self, color_dim = None, size_dim_indecies = False):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'image.cmap': 'turbo',
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(params)
        save_name = f'parameter_space_{self.patient_id}'

        alpha = 0.8

        """
        # DEFAULT, UNSCALED
        vmax = None #np.max([self.rmse_conv_mask_vxls, self.rmse_seg_mask_vxls, self.rmse_net_mask_vxls])
        vmin = None
        extend = 'neither'
        """
        """
        # EMIN_1064      
        vmax = 0.07 #np.max([self.rmse_conv_mask_vxls, self.rmse_seg_mask_vxls, self.rmse_net_mask_vxls])
        vmin = 0
        extend = 'max'
        """
        """
        # EMIN_1011      
        vmax = 0.125 #np.max([self.rmse_conv_mask_vxls, self.rmse_seg_mask_vxls, self.rmse_net_mask_vxls])
        vmin = 0
        extend = 'max'
        """
        
        # EMIN_1092     
        vmax = 0.1 #np.max([self.rmse_conv_mask_vxls, self.rmse_seg_mask_vxls, self.rmse_net_mask_vxls])
        vmin = 0
        extend = 'max'
        
        

        fig, [(ax00, ax01, ax02), (ax10, ax11, ax12), (ax20, ax21, ax22)] = plt.subplots(nrows = 3, ncols = 3, figsize=(20, 15))
        #fig.suptitle(suptitle, fontsize=20)

        scatter00 = ax00.scatter(x=self.Dt_conv_mask_vxls, y=self.Fp_conv_mask_vxls*100, c=self.rmse_conv_mask_vxls, alpha=alpha, vmin=vmin, vmax=vmax)
        scatter01 = ax01.scatter(x=self.Dt_seg_mask_vxls, y=self.Fp_seg_mask_vxls*100, c=self.rmse_seg_mask_vxls, alpha=alpha, vmin=vmin, vmax=vmax)
        scatter02 = ax02.scatter(x=self.Dt_net_mask_vxls, y=self.Fp_net_mask_vxls*100, c=self.rmse_net_mask_vxls, alpha=alpha, vmin=vmin, vmax=vmax)

        scatter10 = ax10.scatter(x=self.Fp_conv_mask_vxls*100, y=self.Dp_conv_mask_vxls, c=self.rmse_conv_mask_vxls, alpha=alpha, vmin=vmin, vmax=vmax)
        scatter11 = ax11.scatter(x=self.Fp_seg_mask_vxls*100, y=self.Dp_seg_mask_vxls, c=self.rmse_seg_mask_vxls, alpha=alpha, vmin=vmin, vmax=vmax)
        scatter12 = ax12.scatter(x=self.Fp_net_mask_vxls*100, y=self.Dp_net_mask_vxls, c=self.rmse_net_mask_vxls, alpha=alpha, vmin=vmin, vmax=vmax)

        scatter20 = ax20.scatter(x=self.Dp_conv_mask_vxls, y=self.Dt_conv_mask_vxls, c=self.rmse_conv_mask_vxls, alpha=alpha, vmin=vmin, vmax=vmax)
        scatter21 = ax21.scatter(x=self.Dp_seg_mask_vxls, y=self.Dt_seg_mask_vxls, c=self.rmse_seg_mask_vxls, alpha=alpha, vmin=vmin, vmax=vmax)
        scatter22 = ax22.scatter(x=self.Dp_net_mask_vxls, y=self.Dt_net_mask_vxls, c=self.rmse_net_mask_vxls, alpha=alpha, vmin=vmin, vmax=vmax)

    
        cbar00 = fig.colorbar(scatter00, ax = ax00, extend=extend)
        cbar01 = fig.colorbar(scatter01, ax = ax01, extend=extend)
        cbar02 = fig.colorbar(scatter02, ax = ax02, extend=extend)
        cbar10 = fig.colorbar(scatter10, ax = ax10, extend=extend)
        cbar11 = fig.colorbar(scatter11, ax = ax11, extend=extend)
        cbar12 = fig.colorbar(scatter12, ax = ax12, extend=extend)
        cbar20 = fig.colorbar(scatter20, ax = ax20, extend=extend)
        cbar21 = fig.colorbar(scatter21, ax = ax21, extend=extend)
        cbar22 = fig.colorbar(scatter22, ax = ax22, extend=extend)

        cbar_description = 'RMSE'
        cbar00.set_label(label=fr'{cbar_description}')
        cbar01.set_label(label=fr'{cbar_description}')
        cbar02.set_label(label=fr'{cbar_description}')
        cbar10.set_label(label=fr'{cbar_description}')
        cbar11.set_label(label=fr'{cbar_description}')
        cbar12.set_label(label=fr'{cbar_description}')
        cbar20.set_label(label=fr'{cbar_description}')
        cbar21.set_label(label=fr'{cbar_description}')
        cbar22.set_label(label=fr'{cbar_description}')

        ax00.set(xlim=(0, 0.005), ylim=(0, 70), xlabel=r'$D_t$ [mm$^2$/s]', ylabel=r'$f_p$ [%]', title='LSQ')
        ax01.set(xlim=(0, 0.005), ylim=(0, 70), xlabel=r'$D_t$ [mm$^2$/s]', ylabel=r'$f_p$ [%]', title='SEG')
        ax02.set(xlim=(0, 0.005), ylim=(0, 70), xlabel=r'$D_t$ [mm$^2$/s]', ylabel=r'$f_p$ [%]', title='DNN')
                
        ax10.set(xlim=(0, 70), ylim=(0, 0.3), xlabel=r'$f_p$ [%]', ylabel=r'$D_p$ [mm$^2$/s]', title='LSQ')
        ax11.set(xlim=(0, 70), ylim=(0, 0.3), xlabel=r'$f_p$ [%]', ylabel=r'$D_p$ [mm$^2$/s]', title='SEG')
        ax12.set(xlim=(0, 70), ylim=(0, 0.3), xlabel=r'$f_p$ [%]', ylabel=r'$D_p$ [mm$^2$/s]', title='DNN')

        ax20.set(xlim=(0, 0.3), ylim=(0, 0.005), xlabel=r'$D_p$ [mm$^2$/s]', ylabel=r'$D_t$ [mm$^2$/s]', title='LSQ')
        ax21.set(xlim=(0, 0.3), ylim=(0, 0.005), xlabel=r'$D_p$ [mm$^2$/s]', ylabel=r'$D_t$ [mm$^2$/s]', title='SEG')
        ax22.set(xlim=(0, 0.3), ylim=(0, 0.005), xlabel=r'$D_p$ [mm$^2$/s]', ylabel=r'$D_t$ [mm$^2$/s]', title='DNN')
                
        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, save_name))


    def plt_param_distr(self):
        sns.set()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'upper right',
                  'legend.framealpha': 0.75,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(params)
        save_name = f'param_dists_{self.patient_id}'

        alpha = 0.4
        num_bins = 50
        Dt_bins = np.linspace(0, 0.005, num_bins)
        Fp_bins = np.linspace(0, 70, num_bins)
        Dp_bins = np.linspace(0, 0.3, num_bins)
        rmse_bins = np.linspace(0, 0.08, num_bins)

        kde_Dt_range = np.linspace(0, 0.005, 500)
        kde_Fp_range = np.linspace(0, 70, 500)
        kde_Dp_range = np.linspace(0, 0.3, 500)
        kde_rmse_range = np.linspace(0, 0.15, 500)

        kde_Dt_conv = stats.gaussian_kde(self.Dt_conv_mask_vxls)
        kde_Dt_seg = stats.gaussian_kde(self.Dt_seg_mask_vxls)
        kde_Dt_net = stats.gaussian_kde(self.Dt_net_mask_vxls)
        kde_Fp_conv = stats.gaussian_kde(self.Fp_conv_mask_vxls*100)
        kde_Fp_seg = stats.gaussian_kde(self.Fp_seg_mask_vxls*100)
        kde_Fp_net = stats.gaussian_kde(self.Fp_net_mask_vxls*100)
        kde_Dp_conv = stats.gaussian_kde(self.Dp_conv_mask_vxls)
        kde_Dp_seg = stats.gaussian_kde(self.Dp_seg_mask_vxls)
        kde_Dp_net = stats.gaussian_kde(self.Dp_net_mask_vxls)
        kde_rmse_conv = stats.gaussian_kde(self.rmse_conv_mask_vxls)
        kde_rmse_seg = stats.gaussian_kde(self.rmse_seg_mask_vxls)
        kde_rmse_net = stats.gaussian_kde(self.rmse_net_mask_vxls)

        suptitle = 'test'
        fig, [(ax00), (ax10), (ax20), (ax30)] = plt.subplots(nrows = 4, ncols = 1, figsize=(15, 20))
        #fig.suptitle(suptitle, fontsize=20)

        ax00.hist(self.Dt_conv_mask_vxls, density=True, bins=Dt_bins, color='C0', alpha=alpha, label='LSQ')
        ax00.hist(self.Dt_seg_mask_vxls, density=True, bins=Dt_bins, color='C1', alpha=alpha, label='SEG')
        ax00.hist(self.Dt_net_mask_vxls, density=True, bins=Dt_bins, color='C2', alpha=alpha, label='DNN')
        ax00.plot(kde_Dt_range, kde_Dt_conv.pdf(kde_Dt_range), color='C0', label=r'$PDF_{LSQ}$')
        ax00.plot(kde_Dt_range, kde_Dt_seg.pdf(kde_Dt_range), color='C1', label=r'$PDF_{SEG}$')
        ax00.plot(kde_Dt_range, kde_Dt_net.pdf(kde_Dt_range), color='C2', label=r'$PDF_{DNN}$')
        ax00.set(xlim=(0, 0.005), xlabel=r'$D_t$ [mm$^2$/s]', ylabel='Probability density')
        ax00.legend()

        ax10.hist(self.Fp_conv_mask_vxls*100, density=True, bins=Fp_bins, color='C0', alpha=alpha, label='LSQ')
        ax10.hist(self.Fp_seg_mask_vxls*100, density=True, bins=Fp_bins, color='C1', alpha=alpha, label='SEG')
        ax10.hist(self.Fp_net_mask_vxls*100, density=True, bins=Fp_bins, color='C2', alpha=alpha, label='DNN')
        ax10.plot(kde_Fp_range, kde_Fp_conv.pdf(kde_Fp_range), color='C0', label=r'$PDF_{LSQ}$')
        ax10.plot(kde_Fp_range, kde_Fp_seg.pdf(kde_Fp_range),color='C1',  label=r'$PDF_{SEG}$')
        ax10.plot(kde_Fp_range, kde_Fp_net.pdf(kde_Fp_range), color='C2', label=r'$PDF_{DNN}$')
        ax10.set(xlim=(0, 70), xlabel=r'$f_p$ [%]', ylabel='Probability density')
        ax10.legend()

        ax20.hist(self.Dp_conv_mask_vxls, density=True, bins=Dp_bins, color='C0', alpha=alpha, label='LSQ')
        ax20.hist(self.Dp_seg_mask_vxls, density=True, bins=Dp_bins, color='C1', alpha=alpha, label='SEG')
        ax20.hist(self.Dp_net_mask_vxls, density=True, bins=Dp_bins, color='C2', alpha=alpha, label='DNN')
        ax20.plot(kde_Dp_range, kde_Dp_conv.pdf(kde_Dp_range), color='C0', label=r'$PDF_{LSQ}$')
        ax20.plot(kde_Dp_range, kde_Dp_seg.pdf(kde_Dp_range), color='C1', label=r'$PDF_{SEG}$')
        ax20.plot(kde_Dp_range, kde_Dp_net.pdf(kde_Dp_range), color='C2', label=r'$PDF_{DNN}$')
        ax20.set(xlim=(0, 0.3), xlabel=r'$D_p$ [mm$^2$/s]', ylabel='Probability density')
        ax20.legend()

        ax30.hist(self.rmse_conv_mask_vxls, density=True, bins=rmse_bins, color='C0', alpha=alpha, label='LSQ')
        ax30.hist(self.rmse_seg_mask_vxls, density=True, bins=rmse_bins, color='C1', alpha=alpha, label='SEG')
        ax30.hist(self.rmse_net_mask_vxls, density=True, bins=rmse_bins, color='C2', alpha=alpha, label='DNN')
        ax30.plot(kde_rmse_range, kde_rmse_conv.pdf(kde_rmse_range), color='C0', label=r'$PDF_{LSQ}$')
        ax30.plot(kde_rmse_range, kde_rmse_seg.pdf(kde_rmse_range), color='C1', label=r'$PDF_{SEG}$')
        ax30.plot(kde_rmse_range, kde_rmse_net.pdf(kde_rmse_range), color='C2', label=r'$PDF_{DNN}$')
        ax30.set(xlim=(0, 0.08), xlabel=r'RMSE', ylabel='Probability density')
        ax30.legend()


        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, save_name))




