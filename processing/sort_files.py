
import numpy as np
import pydicom

import glob
import os
import SimpleITK as sitk
import dicom2nifti
import nibabel as nib



class Preprocess:
    def __init__(self, patient_id, mask_name, src_dir_raw_dwi, fname_raw_mask, src_dir_raw_machine_adc_output):
        self.patient_id = patient_id
        self.mask_name = mask_name
        self.src_dir_raw_dwi = src_dir_raw_dwi
        self.fnames_raw_dwi = src_dir_raw_dwi + '/*.IMA'
        self.fname_raw_mask = fname_raw_mask

        self.sorted_patient_dir = f'data/sorted/{patient_id}'
        self.dst_dir_sort_by_slice_num = os.path.join(self.sorted_patient_dir, 'by_slice_num')
        self.dst_dir_sort_by_bval_num = os.path.join(self.sorted_patient_dir, 'by_bval')

        self.slice_locactions = None    # not really necessarily
        self.num_slices = None
        self.bvals = None

        self.src_dir_raw_machine_adc_output = src_dir_raw_machine_adc_output
        self.fnames_raw_machine_adc_output = src_dir_raw_machine_adc_output + '/*IMA'

        if not os.path.exists(self.sorted_patient_dir):
            os.makedirs(self.sorted_patient_dir)

        


    def sort_files_by_slice_number(self, src, dst):
        """
        Makes the folder by_slice_num with subfolders corresponding to each slice. Each 
        subfolder contains the dicom files with the different b-values for that given slice. 
        """

        unsorted_files = []
    
        # open all files in source, extract Slice Location and B-value and store the data of 
        # the files as a subarray in the array unsorted_files
        paths = glob.glob(src)
        for path in paths:
            dataset = pydicom.dcmread(path)
            
            SliceLocation = round(dataset.get("SliceLocation","NA"), ndigits = 3)
            B_value = dataset[0x0019, 0x0100c].value
            
            unsorted_files.append([path, dataset, SliceLocation, B_value])
        
        # sorted lists of unique slice locations and b-values
        sorted_set_SL = sorted(set(el[2] for el in unsorted_files))
        sorted_set_B = sorted(set(el[3] for el in unsorted_files))

        # assgin value to member variables
        self.slice_locactions = sorted_set_SL
        self.num_slices = len(sorted_set_SL)
        self.bvals = sorted_set_B

        # sort the files on slice location
        sorted_files_SL = sorted(unsorted_files, key = lambda unsorted_file: unsorted_file[2])
        
        # save file in sorted file system based on slice number
        for i in range(len(sorted_files_SL)):
            path, dataset, SliceLocation, B_value = sorted_files_SL[i]
            slice_number = sorted_set_SL.index(SliceLocation) #+ 1
            fileName = f'SN_{slice_number}__B_{str(B_value)}.dcm'
        
            # save files to a nested folder structure
            if not os.path.exists(os.path.join(dst, f'SN_{slice_number}/dcm')):
                os.makedirs(os.path.join(dst, f'SN_{slice_number}/dcm'))
            dataset.save_as(os.path.join(dst, f'SN_{slice_number}/dcm', fileName))
        
        print("Done sorting files by slice number")



    def sort_files_by_bval(self, src, dst):
        """
        Makes by_bval with subfolders corresponding to each b-value. Each subfolder contains the
        dicom image series (e.g. all the slices) for that given b-value. 
        """

        unsorted_files = []
        
        # open all files in source, extract Slice Location and B-value and store the data of 
        # the files as a subarray in the array unsorted_files
        paths = glob.glob(src)
        for path in paths:
            dataset = pydicom.dcmread(path)
            
            SliceLocation = round(dataset.get("SliceLocation","NA"), ndigits = 3)
            B_value = dataset[0x0019, 0x0100c].value
            
            unsorted_files.append([path, dataset, SliceLocation, B_value])
        
        # sort the list unique of slice locations
        sorted_set_SL = sorted(set(el[2] for el in unsorted_files))

        # sort the files on slice location
        sorted_files = sorted(unsorted_files, key = lambda unsorted_file: (unsorted_file[3], unsorted_file[2]))

        # save file in sorted file system based on slice number
        for i in range(len(sorted_files)):
            path, dataset, SliceLocation, B_value = sorted_files[i]
            slice_number = sorted_set_SL.index(SliceLocation) #+ 1
            fileName = f'SN_{slice_number}__B_{str(B_value)}.dcm'
        
            # save files to a nested folder structure
            if not os.path.exists(os.path.join(dst, f'B_{B_value}/dcm')):
                os.makedirs(os.path.join(dst, f'B_{B_value}/dcm'))
            dataset.save_as(os.path.join(dst, f'B_{B_value}/dcm', fileName))
        
        print("Done sorting files by b-value")


    """ 
    # OLD CODE

    def bval_series_as_nifti(self, bvals):
        assert self.bvals != None, "Need to first run the function sort_files_by_slice_number"

        for bval in bvals:
            src_dir = f"../data/sorted/{self.patient_id}/by_bval/B_{bval}/dcm"
            dst_dir = f"../data/sorted/{self.patient_id}/by_bval/B_{bval}"

            convert_dicom_to_nifti(src_dir, dst_dir, f'B_{bval}')

        print("Done coverting series of dcm files into nifti files for each b-value.")
    """


    def bval_series_as_nifti(self, dir, bvals):
        """
        Each b-value the dicom image series (e.g. all the slices), made by the function
        sort_files_by_bval, is converted into one nifti file.
        """
        
        for bval in bvals:
            src_dir = os.path.join(dir, f'B_{bval}/dcm')
            dst_fname = os.path.join(dir, f'B_{bval}/B_{bval}')

            dicom2nifti.dicom_series_to_nifti(src_dir, dst_fname, reorient_nifti=False)
        
        print("Done coverting series of dcm files into nifti files for each b-value.")



    def resample_mask(self, sorted_patient_dir, bvals, fname_raw_mask, mask_name):

        dst_dir_to_resampled_mask = os.path.join(sorted_patient_dir, f'resampled_masks/{mask_name}/by_bval')
        if not os.path.exists(dst_dir_to_resampled_mask):
            os.makedirs(dst_dir_to_resampled_mask)


        raw_mask_img = sitk.ReadImage(fname_raw_mask)

        for bval in bvals:
            src_path_to_ref_img = os.path.join(sorted_patient_dir, f'by_bval/B_{bval}/B_{bval}.nii')
            dst_path_to_resampled_mask = os.path.join(dst_dir_to_resampled_mask, f'resampled_mask_B_{bval}.nii')

            ref_img= sitk.ReadImage(src_path_to_ref_img)

            resample = sitk.ResampleImageFilter()
            resample.SetReferenceImage(ref_img)
            resample.SetInterpolator(sitk.sitkNearestNeighbor) # other option sitk.sitkLinear, sitk.sitkNearestNeighbor
            resampled_image = resample.Execute(raw_mask_img) # Run the resampling

            sitk.WriteImage(resampled_image, dst_path_to_resampled_mask) # save the result
        
        print("Done resampling mask")


    def create_bvalstxt(self, dst_dir, bvals):
        np.savetxt(os.path.join(dst_dir + '/bvals.txt'), bvals) 
        print("Done writing b-values to file.")


    def create_multib_val_4d_as_nifti(self, sorted_patient_dir, src_dir_raw_dwi):
        dst_dir = os.path.join(sorted_patient_dir, 'multibval_4d')
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        dicom2nifti.convert_directory(src_dir_raw_dwi, dst_dir, compression=False, reorient=False)
        print("Done creating multi-b-value-4d-data as nifti file")




    def resampled_mask_slices(self, sorted_patient_dir, mask_name, num_slices):
        dst_dir_to_resampled_mask_slices = os.path.join(sorted_patient_dir, f'resampled_masks/{mask_name}/by_slice_num')
        if not os.path.exists(dst_dir_to_resampled_mask_slices):
            os.makedirs(dst_dir_to_resampled_mask_slices)

        resampled_mask_fname = sorted_patient_dir + f'/resampled_masks/{mask_name}/by_bval/resampled_mask_B_0.nii'
        resampled_mask_obj = nib.load(resampled_mask_fname)
        resampled_mask_data = resampled_mask_obj.get_fdata()

        dwi_4d_obj = nib.load(glob.glob(os.path.join(sorted_patient_dir, 'multibval_4d/*.nii'))[0])

        for slice_num in range(num_slices):
            resampled_mask_slice_data = resampled_mask_data[:, :, slice_num]
            resampled_mask_slice_nifti = nib.Nifti1Image(resampled_mask_slice_data, dwi_4d_obj.affine)
            nib.save(resampled_mask_slice_nifti, os.path.join(dst_dir_to_resampled_mask_slices, f'resampled_mask_B_0_SL_{slice_num}.nii'))

        print("Done creating nifti mask slices")
    

    def create_adc_machine_output_as_nifti(self, sorted_patient_dir, src_dir_raw_machine_adc_output):
        """
        Note to self: this can be merged with the function create_multib_val_4d_as_nifti
        """
        dst_dir = os.path.join(sorted_patient_dir, 'machine_adc_map')
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        dicom2nifti.convert_directory(src_dir_raw_machine_adc_output, dst_dir, compression=False, reorient=False)
        print("Done creating machine adc map as nifti file")

            

    def run_preprocessing(self):
        self.sort_files_by_slice_number(self.fnames_raw_dwi, self.dst_dir_sort_by_slice_num)
        self.sort_files_by_bval(self.fnames_raw_dwi, self.dst_dir_sort_by_bval_num)

        #self.sorted_patient_dir
        self.create_bvalstxt(self.sorted_patient_dir, self.bvals)
        self.create_multib_val_4d_as_nifti(self.sorted_patient_dir, self.src_dir_raw_dwi)

        self.bval_series_as_nifti(self.dst_dir_sort_by_bval_num, self.bvals)
        self.resample_mask(self.sorted_patient_dir, self.bvals, self.fname_raw_mask, self.mask_name)
        #self.resampled_mask_slices(self.sorted_patient_dir, self.mask_name, self.num_slices)

        self.create_adc_machine_output_as_nifti(self.sorted_patient_dir, self.src_dir_raw_machine_adc_output)

           
        




