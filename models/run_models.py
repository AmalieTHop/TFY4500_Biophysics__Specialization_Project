
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pydicom
import seaborn as sns

import glob
import os
import time
from tqdm import tqdm

import nibabel as nib
import pydicom

import models.MRItools.tools.getADCDKI as from_getADCDKI
import models.IVIM_OSIPI.src.original.OGC_AmsterdamUMC.LSQ_fitting as from_OGC_AmsterdamUMC
import models.IVIMNET.IVIMNET.deep as deep
from models.IVIMNET.hyperparams import hyperparams as hp




class ADC:
    def __init__(self, 
                 patient_id, 
                 mask_name,
                 fit_dki = 0, 
                 algo = "linear", 
                 ncpu = 1):
        
        self.patient_id = patient_id
        self.mask_name = mask_name

        self.dwi_4d_fname = glob.glob(f'data/sorted/{patient_id}/multibval_4d/*.nii')[0]
        self.bvals_txt_fname = f'data/sorted/{patient_id}/bvals.txt'
        self.mask_fname = f'data/sorted/{patient_id}/resampled_masks/{mask_name}/by_bval/resampled_mask_B_0.nii'


        self.fit_dki = fit_dki
        self.dst_dir = f'data/results/{patient_id}/{mask_name}/ADC'
        self.algo = algo
        self.ncpu = ncpu

        self.adc_map_fname = f'{self.dst_dir}' + '/ADC.nii'
        self.machine_adc_map_fname = glob.glob(f'data/sorted/{patient_id}/machine_adc_map/*.nii')[0]


    def fit_adc(self):
        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)

        from_getADCDKI.FitModel(self.dwi_4d_fname, self.bvals_txt_fname, self.fit_dki, self.dst_dir, self.algo, self.ncpu, self.mask_fname)


    def calculate_descriptive_statistics(self):

        adc_obj = nib.load(self.adc_map_fname)
        adc_data = adc_obj.get_fdata()
        adc_mean = np.mean(adc_data[adc_data != 0])
        adc_median = np.median(adc_data[adc_data != 0])
        adc_min = np.min(adc_data[adc_data != 0])
        adc_max = np.max(adc_data[adc_data != 0])
        adc_range = adc_max - adc_min
        descriptive_statistics = [[adc_mean, adc_median, adc_min, adc_max, adc_range]]
        
        np.savetxt(os.path.join(self.dst_dir, 'descriptive_statistics.txt'), descriptive_statistics, delimiter = '\t', header = 'mean, median, min, max, range')
        print('Done calculating descriptive statistics.')


    def plot_ADC_map_for_slice_with_mask(self, slice):
        
        adc_obj = nib.load(self.adc_map_fname)
        adc_data = adc_obj.get_fdata()

        adc_data_slice = adc_data[:, :, 16]
    
        plt.figure(figsize=(5,5))
        plt.imshow(adc_data_slice)
        plt.show()

    def fit_adc_full_slice(self, slice):
        dwi_4d_obj = nib.load(self.dwi_4d_fname)
        dwi_4d_data = dwi_4d_obj.get_fdata()
        dwi_3d_data = dwi_4d_data[:, :, slice, :]
        
        bvals = np.loadtxt(self.bvals_txt_fname)

        mask_ones = np.ones((dwi_3d_data.shape[0], dwi_3d_data.shape[1]))

        data_input = [dwi_3d_data, bvals, self.algo, self.fit_dki, mask_ones, slice] 

        data_out = from_getADCDKI.FitSlice(data_input)
        s0_slice, adc_slice, k_slice, exit_slice, mse_slice, idx_slice = data_out

        np.save(self.dst_dir + f'/SL_{idx_slice}_S0.npy', s0_slice)
        np.save(self.dst_dir + f'/SL_{idx_slice}_ADC.npy', adc_slice)

        adc_slice_nifti = nib.Nifti1Image(adc_slice, dwi_4d_obj.affine)
        nib.save(adc_slice_nifti, os.path.join(self.dst_dir, f'SL_{idx_slice}_ADC.nii'))

    def plot_ADC_map_for_slice_with_mask(self, slice):
        
        adc_fname = f'{self.dst_dir}' + f'/SL_{slice}_ADC.npy'
        adc_data_slice = np.load(adc_fname)
    
        plt.figure(figsize=(5,5))
        plt.title(f'SL_{slice}_ADC')
        plt.imshow(adc_data_slice.T)
        plt.colorbar()
        plt.show()


    def calculate_deviation_from_machine_adc(self):
        
        machine_adc_map_obj = nib.load(self.machine_adc_map_fname)
        modeled_adc_map_obj = nib.load(self.adc_map_fname)
        #mask_obj = nib.load(self.mask_fname)

        machine_adc_map_data = machine_adc_map_obj.get_fdata()
        modeled_adc_map_data = modeled_adc_map_obj.get_fdata()
        #mask_data = mask_obj.get_fdata()

        

        deviation = abs(modeled_adc_map_data[(modeled_adc_map_data != 0) & (machine_adc_map_data != 0)]*10**6 - machine_adc_map_data[(modeled_adc_map_data != 0) & (machine_adc_map_data != 0)])/machine_adc_map_data[(modeled_adc_map_data != 0) & (machine_adc_map_data != 0)]
        mean_deviation = np.mean(deviation)
        #print(modeled_adc_map_data[modeled_adc_map_data != 0]*10**6)
        #print(machine_adc_map_data[modeled_adc_map_data != 0])
        #print(deviation)
        #np.savetxt('test.txt', machine_adc_map_data[modeled_adc_map_data != 0])
        print(f'Mean deviation: {mean_deviation}')




"""
class IVIM_SL:
    def __init__(self, 
                 patient_id, 
                 mask_name, 
                 model = 'one-step', 
                 multithreading = True):
        
        self.patient_id = patient_id
        self.mask_name = mask_name

        self.dwi_4d_fname = glob.glob(f'data/sorted/{patient_id}/multibval_4d/*.nii')[0]
        self.bvals_txt_fname = f'data/sorted/{patient_id}/bvals.txt'
        self.mask_fname = f'data/sorted/{patient_id}/resampled_masks/{mask_name}/by_bval/resampled_mask_B_0.nii'
        
        self.model = model
        self.dst_dir = f'data/results/{patient_id}/{mask_name}/IVIM_SL/{model}'
        self.multithreading = multithreading

        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)


    def fit_ivim_sl(self):
        fit_ivim_sl(self.dwi_4d_fname, self.bvals_txt_fname, self.mask_fname, self.model, self.dst_dir, self.multithreading)


    def calculate_descriptive_statistics(self):
        params = ['D_map', 'Fivim_map', 'Dstar_map']
        results = np.zeros((len(params), 5))

        for i, param in enumerate(params):
            param_fname = f'data/results/{self.patient_id}/IVIM_SL/{self.mask_name}/{self.model}/{param}.nii'
            param_obj = nib.load(param_fname)
            param_data = param_obj.get_fdata()
            param_mean = np.mean(param_data[param_data != 0])
            param_median = np.median(param_data[param_data != 0])
            param_min = np.min(param_data[param_data != 0])
            param_max = np.max(param_data[param_data != 0])
            param_range = param_max - param_min
            results[i] = [param_mean, param_median, param_min, param_max, param_range]

        np.savetxt(f'{self.dst_dir}' + '/results.txt', results, delimiter = '\t', header = 'mean, median, min, max, range')
        print('Done calculating descriptive statistics.')
"""



class IVIM_OSIPI():
    def __init__(self, patient_id, mask_name, method):
        self.patient_id = patient_id
        self.mask_name = mask_name
        self.method = method

        self.dwi_4d_fname = glob.glob(f'data/sorted/{patient_id}/multibval_4d/*.nii')[0]
        self.bvals_txt_fname = f'data/sorted/{patient_id}/bvals.txt'
        self.mask_fname = f'data/sorted/{patient_id}/resampled_masks/{mask_name}/by_bval/resampled_mask_B_0.nii'

        self.dst_dir = f'data/results/{patient_id}/{mask_name}/IVIM/{method}'

        self.bvals = None

        self.dwi_4d_data = None
        self.dwi_4d_affine = None
        self.num_slices = None
        self.num_bvals = None
        self.width = None
        self.height = None

        self.mask_data = None

        self.load_data()

        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)




    def load_data(self):
        self.bvals = np.loadtxt(self.bvals_txt_fname)

        dwi_4d_file = nib.load(self.dwi_4d_fname)
        self.dwi_4d_data = dwi_4d_file.get_fdata()
        self.dwi_4d_affine = dwi_4d_file.affine
        self.width = self.dwi_4d_data.shape[0]
        self.height = self.dwi_4d_data.shape[1]
        self.num_slices = self.dwi_4d_data.shape[2]
        self.num_bvals = self.dwi_4d_data.shape[3]

        mask_file = nib.load(self.mask_fname)
        self.mask_data = mask_file.get_fdata()



    def fit_ivim_osipi(self, S0_output = True, fitS0 = True, njobs = 1, bounds = None):
        dwi_2d_data = np.reshape(self.dwi_4d_data, (self.width*self.height*self.num_slices, self.num_bvals))
        mask_1d_data = np.reshape(self.mask_data, (self.width*self.height*self.num_slices))


        params = ['Dt', 'Fp', 'Dp', 'S0']
        if self.method == 'conv':
            Dt_1d, Fp_1d, Dp_1d, S0_1d = from_OGC_AmsterdamUMC.fit_least_squares_array(self.bvals, dwi_2d_data, mask_1d_data, S0_output=True, fitS0=True, njobs=1)
        if self.method == 'seg':
            Dt_1d, Fp_1d, Dp_1d, S0_1d = from_OGC_AmsterdamUMC.fit_segmented_array(self.bvals, dwi_2d_data, mask_1d_data, njobs=4, cutoff=200) 
        params_data = [Dt_1d, Fp_1d, Dp_1d, S0_1d]


        for i, param in enumerate(params):
            param_3d = np.reshape(params_data[i], (self.width, self.height, self.num_slices))
            #np.save(self.dst_dir + f'/{param}/{param}_{self.method}.npy', param_3d)
            param_3d_lsq_nifti = nib.Nifti1Image(param_3d, self.dwi_4d_affine)
            nib.save(param_3d_lsq_nifti, os.path.join(self.dst_dir, f'{param}.nii'))


    """
    def run_ivim_osipi_full_slice(self, S0_output = True, fitS0 = True, njobs = 1, bounds = None):
        if not os.path.exists(self.dst_dir):
            os.makedirs(os.path.join(self.dst_dir, f'Dt'))
            os.makedirs(os.path.join(self.dst_dir, f'Fp'))
            os.makedirs(os.path.join(self.dst_dir, f'Dp'))
            os.makedirs(os.path.join(self.dst_dir, f'S0'))

        for i in tqdm(range(self.num_slices)):
            idx_slice = i
            dwi_slice_3d_data = self.dwi_4d_data[:, :, idx_slice, :]
            mask_slice_2d_data = self.mask_data[:, :, idx_slice]

            dwi_slice_2d_data = np.reshape(dwi_slice_3d_data, (self.width*self.height, self.num_bvals))
            mask_slice_1d_data = np.reshape(mask_slice_2d_data, (self.width*self.height))

            Dt_1d_lsq, Fp_1d_lsq, Dp_1d_lsq, S0_1d_lsq = from_LSQ_fitting.fit_least_squares_array(self.bvals, dwi_slice_2d_data, mask_slice_1d_data, S0_output=True, fitS0=True, njobs=1)

            Dt_2d = np.reshape(Dt_1d_lsq, (self.width, self.height))
            Fp_2d = np.reshape(Fp_1d_lsq, (self.width, self.height))
            Dp_2d = np.reshape(Dp_1d_lsq, (self.width, self.height))
            S0_2d = np.reshape(S0_1d_lsq, (self.width, self.height))

            np.save(self.dst_dir + f'/SL_{idx_slice}__Dt_lsq.npy', Dt_2d)
            np.save(self.dst_dir + f'/SL_{idx_slice}__Fp_lsq.npy', Fp_2d)
            np.save(self.dst_dir + f'/SL_{idx_slice}__Dp_lsq.npy', Dp_2d)
            np.save(self.dst_dir + f'/SL_{idx_slice}__S0_lsq.npy', S0_2d)

            Dt_2d_lsq_nifti = nib.Nifti1Image(Dt_2d, None)
            Fp_2d_lsq_nifti = nib.Nifti1Image(Fp_2d, None)
            Dp_2d_lsq_nifti = nib.Nifti1Image(Dp_2d, None)
            S0_2d_lsq_nifti =nib.Nifti1Image(S0_2d, None)
            nib.save(Dt_2d_lsq_nifti, os.path.join(self.dst_dir, f'Dt/SL_{idx_slice}__Dt_lsq.nii'))
            nib.save(Fp_2d_lsq_nifti, os.path.join(self.dst_dir, f'Fp/SL_{idx_slice}__Fp_lsq.nii'))
            nib.save(Dp_2d_lsq_nifti, os.path.join(self.dst_dir, f'Dp/SL_{idx_slice}__Dp_lsq.nii'))
            nib.save(S0_2d_lsq_nifti, os.path.join(self.dst_dir, f'S0/SL_{idx_slice}__S0_lsq.nii'))
    """

    def calculate_descriptive_statistics(self):
        params = ['Dt', 'Fp', 'Dp']
        descriptive_statistics = np.zeros((3, 5))

        for i, param in enumerate(params):
            param_fname = os.path.join(self.dst_dir, f'{param}.nii')
            param_obj = nib.load(param_fname)
            param_data = param_obj.get_fdata()

            param_mean = np.mean(param_data[self.mask_data != 0])
            param_median = np.median(param_data[self.mask_data != 0])
            param_min = np.min(param_data[self.mask_data != 0])
            param_max = np.max(param_data[self.mask_data != 0])
            param_range = param_max - param_min

            descriptive_statistics[i] = [param_mean, param_median, param_min, param_max, param_range]

        np.savetxt(os.path.join(self.dst_dir, 'descriptive_statistics.txt'), descriptive_statistics, delimiter = '\t', header = 'mean, median, min, max, range')
        print(f'Done calculating descriptive statistics for {self.method}.')



class IVIM_NN():
    def __init__(self, patient_id, mask_name):
        self.patient_id = patient_id
        self.mask_name = mask_name
        self.method = 'net'

        self.dwi_4d_fname = glob.glob(f'data/sorted/{patient_id}/multibval_4d/*.nii')[0]
        self.bvals_txt_fname = f'data/sorted/{patient_id}/bvals.txt'
        self.mask_fname = f'data/sorted/{patient_id}/resampled_masks/{mask_name}/by_bval/resampled_mask_B_0.nii'

        self.dst_dir = f'data/results/{patient_id}/{mask_name}/IVIM/{self.method}'

        # member variables below are given value though the functions 
        # load_data and preprocess_data, such that by the end of the construnctor
        # all member varaibles has a defined value
        self.bvals = None

        self.dwi_4d_file = None
        self.dwi_4d_data = None
        self.dwi_4d_affine = None
        self.num_slices = None
        self.num_bvals = None
        self.width = None
        self.height = None

        self.mask_data = None

        arg = hp()
        self.arg = deep.checkarg(arg)
        self.net_data = None
        self.valid_id = None

        self.load_data()
        self.preprocess_data()

        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)


    def load_data(self):
        self.bvals = np.loadtxt(self.bvals_txt_fname)

        self.dwi_4d_file = nib.load(self.dwi_4d_fname)
        self.dwi_4d_data = self.dwi_4d_file.get_fdata()
        self.dwi_4d_affine = self.dwi_4d_file.affine
        self.width, self.height, self.num_slices, self.num_bvals = self.dwi_4d_data.shape

        mask_file = nib.load(self.mask_fname)
        self.mask_data = mask_file.get_fdata()



    def preprocess_data(self):
        # reshaope mask
        mask_1d_data = np.reshape(self.mask_data, self.width*self.height*self.num_slices).astype(bool)

        # reshape data to be trained
        dwi_2d_data = np.reshape(self.dwi_4d_data, (self.width*self.height*self.num_slices, self.num_bvals))

        # select only relevant values
        S0 = np.nanmean(dwi_2d_data[:, self.bvals == 0], axis=1)
        S0[S0 != S0] = 0
        S0 = np.squeeze(S0)

        # delete background, but keep mask
        self.valid_id =  np.logical_or(S0 > (np.mean(S0)), mask_1d_data) #mask_1d_data #(S0 > (0.5 * np.median(S0[S0 > 0])))
        datatot = dwi_2d_data[self.valid_id, :]

        # normalise data
        S0 = np.nanmean(datatot[:, self.bvals == 0], axis=1).astype('<f')
        self.net_data = datatot / S0[:, None]

        # plot traning and validation data
        img = np.zeros([self.width * self.height * self.num_slices])
        img[self.valid_id] = self.net_data[:, 0]
        img = np.reshape(img, [self.width, self.height, self.num_slices])
        img_2d = img[:, :, 16]

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

        fig, ax = plt.subplots()
        im = ax.imshow(img_2d.T)
        ax.axis("off")
        fig.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'filtered_for_bg_noise.pdf')) 

        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(self.dwi_4d_data[:, :, 16, 0].T)
        ax2.axis("off")
        fig2.tight_layout()
        plt.savefig(os.path.join(self.dst_dir, f'not_filtered_for_bg_noise.pdf')) 


    def train_ivim(self):
        # training data
        res = [i for i, val in enumerate(self.net_data != self.net_data) if not val.any()] # Remove NaN data

        # train model
        start_time = time.time()
        model_net = deep.learn_IVIM(self.net_data[res], self.bvals, self.arg)
        elapsed_time1net = time.time() - start_time
        print('\ntime elapsed for Net: {}\n'.format(elapsed_time1net))

        # save model
        torch.save(model_net, os.path.join(self.dst_dir, f'trained_model_net'))
        print("Trained model is saved")

        return model_net


    def predict_ivim(self, net):
        # predict
        start_time = time.time()
        paramsNN = deep.predict_IVIM(self.net_data, self.bvals, net, self.arg)
        elapsed_time1netinf = time.time() - start_time
        print('\ntime elapsed for Net inf: {}\n'.format(elapsed_time1netinf))
        print('\ndata length: {}\n'.format(len(self.net_data)))

        if self.arg.train_pars.use_cuda:
            torch.cuda.empty_cache()

        # define names IVIM params
        names = ['Dt_net', 'Fp_net', 'Dp_net', 'S0_net']
        
        tot = 0
        # fill image array and make nifti
        for k in range(len(names)):
            img = np.zeros([self.width * self.height * self.num_slices])
            img[self.valid_id] = paramsNN[k][tot:(tot + sum(self.valid_id))]
            img = np.reshape(img, [self.width, self.height, self.num_slices])
            nib.save(nib.Nifti1Image(img, self.dwi_4d_file.affine, self.dwi_4d_file.header),'{folder}/{name}.nii.gz'.format(folder = self.dst_dir, name=names[k])),
        print('NN prediction done\n')

    
    def calculate_descriptive_statistics(self):
        #param_names = [f'Dt_t{trained_on[0]}_{trained_on[1]}',f'Fp_t{trained_on[0]}_{trained_on[1]}',f'Dp_t{trained_on[0]}_{trained_on[1]}']
        param_names = ['Dt_net', 'Fp_net', 'Dp_net']
        descriptive_statistics = np.zeros((len(param_names), 5))

        for i, param_name in enumerate(param_names):
            param_fname = os.path.join(self.dst_dir, f'{param_name}.nii.gz')
            param_obj = nib.load(param_fname)
            param_data = param_obj.get_fdata()
            
            param_mean = np.mean(param_data[self.mask_data != 0])
            param_median = np.median(param_data[self.mask_data != 0])
            param_min = np.min(param_data[self.mask_data != 0])
            param_max = np.max(param_data[self.mask_data != 0])
            param_range = param_max - param_min

            descriptive_statistics[i] = [param_mean, param_median, param_min, param_max, param_range]

        np.savetxt(os.path.join(self.dst_dir, 'descriptive_statistics.txt'), descriptive_statistics, delimiter = '\t', header = 'mean, median, min, max, range')
        print('Done calculating descriptive statistics for net.')




class IVIM_NN_combined_training():
    def __init__(self, patient_ids):
        self.patient_ids = patient_ids
        self.mask_root_names = ['GTVp', 'GTVn']
        self.method = 'net'
        self.dst_dir = f'data/results'

        bvals_txt_fname = f'data/sorted/EMIN_1064/bvals.txt'
        self.bvals = np.loadtxt(bvals_txt_fname)# np.array([0, 50, 100, 200, 800])
        self.num_bvals = len(self.bvals)
        self.width = 364
        self.height = 256

        self.dwis_4d_obj = []
        self.nums_slices = []
        self.masks_data = []

        self.preprocessed_dwis_2d_data = []
        self.valid_idx_sets = []

        self.preprocessed_combined_dwis_2d_data = None
        self.prepare_combined_training_data()

        arg = hp()
        self.arg = deep.checkarg(arg)

        if not os.path.exists(self.dst_dir):
            os.makedirs(self.dst_dir)


    def prepare_combined_training_data(self):
        preprocessed_combined_dwis_2d_data = []

        for patient_id in self.patient_ids:
            ### loading data
            # load dwi data
            dwi_4d_fname = glob.glob(f'data/sorted/{patient_id}/multibval_4d/*.nii')[0]

            dwi_4d_obj = nib.load(dwi_4d_fname)
            dwi_4d_data = dwi_4d_obj.get_fdata()
            dwi_4d_affine = dwi_4d_obj.affine
            num_slices = dwi_4d_data.shape[2]

            self.dwis_4d_obj.append(dwi_4d_obj)
            self.nums_slices.append(num_slices)


            # load mask data
            masks_data_patient = []
            for mask_root_name in self.mask_root_names:
                src_fnames_raw_mask = glob.glob(f'data/raw/{patient_id}/Segmentation/mask_{mask_root_name}' + '*.nii.gz')
                for src_fname_raw_mask in src_fnames_raw_mask:
                    mask_name = src_fname_raw_mask.split(f'data/raw/{patient_id}/Segmentation/mask_')[1].split('.nii.gz')[0]
                    mask_fname = f'data/sorted/{patient_id}/resampled_masks/{mask_name}/by_bval/resampled_mask_B_0.nii'
                    mask_obj = nib.load(mask_fname)
                    mask_data_patient = mask_obj.get_fdata()
                    masks_data_patient.append(mask_data_patient)
            self.masks_data.append(masks_data_patient)



            ### preprocessing data
            # creating combined mask
            combined_mask_data = np.sum(masks_data_patient, axis = 0)

            # reshape mask
            combined_mask_1d_data = np.reshape(combined_mask_data, self.width*self.height*num_slices).astype(bool)

            # reshape dwi data
            dwi_2d_data = np.reshape(dwi_4d_data, (self.width*self.height*num_slices, self.num_bvals))

            # select only relevant values
            S0 = np.nanmean(dwi_2d_data[:, self.bvals == 0], axis=1)
            S0[S0 != S0] = 0
            S0 = np.squeeze(S0)

            # delete background, but keep mask
            valid_idx_set =  np.logical_or(S0 > 0.5*(np.mean(S0)), combined_mask_1d_data) #mask_1d_data #(S0 > (0.5 * np.median(S0[S0 > 0])))
            self.valid_idx_sets.append(valid_idx_set)
            valid_dwi_2d_data = dwi_2d_data[valid_idx_set, :]

            # normalise data
            S0 = np.nanmean(valid_dwi_2d_data[:, self.bvals == 0], axis=1).astype('<f')
            preprocessed_dwi_2d_data = valid_dwi_2d_data / S0[:, None]
            self.preprocessed_dwis_2d_data.append(preprocessed_dwi_2d_data)

            for voxl in preprocessed_dwi_2d_data:
                preprocessed_combined_dwis_2d_data.append(np.array(voxl))

        self.preprocessed_combined_dwis_2d_data = np.array(preprocessed_combined_dwis_2d_data)


    
    def train_ivim(self):
        # train model
        start_time = time.time()
        model_net = deep.learn_IVIM(self.preprocessed_combined_dwis_2d_data, self.bvals, self.arg)
        elapsed_time1net = time.time() - start_time
        print('\ntime elapsed for Net: {}\n'.format(elapsed_time1net))

        # save model
        torch.save(model_net, os.path.join(self.dst_dir, f'trained_model_net'))
        print("Trained model is saved")

        return model_net


    def predict_ivim(self, net):
        for i, patient_id in enumerate(self.patient_ids):
            dst_dir = os.path.join(self.dst_dir, f'{patient_id}')
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            preprocessed_dwi_2d_data = self.preprocessed_dwis_2d_data[i]
            num_slices = self.nums_slices[i]
            valid_idx_set = self.valid_idx_sets[i]
            dwi_4d_obj = self.dwis_4d_obj[i]
            
            start_time = time.time()
            paramsNN = deep.predict_IVIM(preprocessed_dwi_2d_data, self.bvals, net, self.arg)
            elapsed_time1netinf = time.time() - start_time
            print('\ntime elapsed for Net inf: {}\n'.format(elapsed_time1netinf))
            print('\ndata length: {}\n'.format(len(preprocessed_dwi_2d_data)))

            if self.arg.train_pars.use_cuda:
                torch.cuda.empty_cache()

            # define names IVIM params
            names = ['Dt_net', 'Fp_net', 'Dp_net', 'S0_net']
            
            tot = 0
            # fill image array and make nifti
            for k in range(len(names)):
                img = np.zeros([self.width * self.height * num_slices])
                img[valid_idx_set] = paramsNN[k][tot:(tot + sum(valid_idx_set))]
                img = np.reshape(img, [self.width, self.height, num_slices])
                nib.save(nib.Nifti1Image(img, dwi_4d_obj.affine, dwi_4d_obj.header),'{folder}/{name}.nii.gz'.format(folder=dst_dir, name=names[k])),
            print('NN prediction done\n')

    
    def calculate_descriptive_statistics(self):
        #param_names = [f'Dt_t{trained_on[0]}_{trained_on[1]}',f'Fp_t{trained_on[0]}_{trained_on[1]}',f'Dp_t{trained_on[0]}_{trained_on[1]}']
        param_names = ['Dt_net', 'Fp_net', 'Dp_net']
        descriptive_statistics = np.zeros((len(param_names), 5))

        for i, patient_id in enumerate(self.patient_ids):
            src_dir = os.path.join(self.dst_dir, f'{patient_id}')

            for mask_root_name in self.mask_root_names:
                src_fnames_raw_mask = glob.glob(f'data/raw/{patient_id}/Segmentation/mask_{mask_root_name}' + '*.nii.gz')
                for src_fname_raw_mask in src_fnames_raw_mask:
                    mask_name = src_fname_raw_mask.split(f'data/raw/{patient_id}/Segmentation/mask_')[1].split('.nii.gz')[0]
                    mask_fname = f'data/sorted/{patient_id}/resampled_masks/{mask_name}/by_bval/resampled_mask_B_0.nii'
                    mask_file = nib.load(mask_fname)
                    mask_data = mask_file.get_fdata()

                    dst_dir = os.path.join(self.dst_dir, f'{patient_id}/{mask_name}/IVIM/net')
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    
                    for j, param_name in enumerate(param_names):
                        param_fname = os.path.join(src_dir, f'{param_name}.nii.gz')
                        param_obj = nib.load(param_fname)
                        param_data = param_obj.get_fdata()
                        
                        param_mean = np.mean(param_data[mask_data != 0])
                        param_median = np.median(param_data[mask_data != 0])
                        param_min = np.min(param_data[mask_data != 0])
                        param_max = np.max(param_data[mask_data != 0])
                        param_range = param_max - param_min

                        descriptive_statistics[j] = [param_mean, param_median, param_min, param_max, param_range]

                    np.savetxt(os.path.join(dst_dir, 'descriptive_statistics.txt'), descriptive_statistics, delimiter = '\t', header = 'mean, median, min, max, range')
                    print('Done calculating descriptive statistics for net.')
        


