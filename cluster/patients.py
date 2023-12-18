import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch

import glob

# patients
import models.run_models as from_models
import processing.sort_files as from_sort_files


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--patient_id', dest='patient_id', type=str)
args = parser.parse_args()
patient_id = args.patient_id


def run_patient_data(patient_id, mask_root_names):
    for mask_root_name in mask_root_names:
        src_dri_raw_dwi = glob.glob(f'data/raw/{patient_id}/MRIdata/EP2D_STIR_DIFF_TRA_2_09I_5B-VERDIER_TRACEW_*')[0]
        src_fnames_raw_mask = glob.glob(f'data/raw/{patient_id}/Segmentation/mask_{mask_root_name}' + '*.nii.gz')
        src_dir_raw_machine_adc_output = glob.glob(f'data/raw/{patient_id}/MRIdata/EP2D_STIR_DIFF_TRA_2_09I_5B-VERDIER_ADC_*')[0]

        if src_fnames_raw_mask: 
            for src_fname_raw_mask in src_fnames_raw_mask:
                mask_name = src_fname_raw_mask.split(f'data/raw/{patient_id}/Segmentation/mask_')[1].split('.nii.gz')[0]
                    
                # Preprocess
                #preprocess = from_sort_files.Preprocess(patient_id, mask_name, src_dri_raw_dwi, src_fname_raw_mask, src_dir_raw_machine_adc_output)
                #preprocess.run_preprocessing()
                
                # ADC
                adc = from_models.ADC(patient_id, mask_name)
                adc.fit_adc()
                adc.calculate_descriptive_statistics()
                #adc.plot_ADC_map_for_slice(16)
                #adc.run_adc_full_slice(16)
                #adc.resampled_mask_slices()
                #adc.plot_ADC_map_for_slice_with_mask(16)
                #adc.calculate_deviation_from_machine_adc()

                method_osipi_conv = 'conv' #'PV_MUMC_two_step' # OGC_AmsterdamUMC
                ivim_osipi_conv = from_models.IVIM_OSIPI(patient_id, mask_name, method_osipi_conv)
                ivim_osipi_conv.fit_ivim_osipi()
                ivim_osipi_conv.calculate_descriptive_statistics()

                method_osipi_seg = 'seg' #'PV_MUMC_two_step' # OGC_AmsterdamUMC
                ivim_osipi_seg = from_models.IVIM_OSIPI(patient_id, mask_name, method_osipi_seg)
                ivim_osipi_seg.fit_ivim_osipi()
                ivim_osipi_seg.calculate_descriptive_statistics()

                ivim_net = from_models.IVIM_NN(patient_id, mask_name)
                trained_model_ivim_net = ivim_net.train_ivim()
                ### trained_model_ivim_net = torch.load(f'data/results/EMIN_1048/GTVp/IVIM/net/trained_model_net')
                trained_model_ivim_net.eval()
                ivim_net.predict_ivim(trained_model_ivim_net)
                ivim_net.calculate_descriptive_statistics()


run_patient_data(patient_id, mask_root_names = ['GTVp', 'GTVn'])