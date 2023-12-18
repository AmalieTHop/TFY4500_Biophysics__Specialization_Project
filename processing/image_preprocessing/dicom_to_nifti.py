import argparse
import logging
import os
import sys
from typing import List

import SimpleITK as sitk
import numpy as np
from pandas import DataFrame

from processing.image_preprocessing.utils.dicom_header import DicomHeader, HeaderEntry
#from utils.dicom_header import DicomHeader, HeaderEntry


class DicomToNifti:
    def __init__(self):
        self._file_ending = '.dcm'
        self._dcm_header = DicomHeader()

        self._position_fields = ['image_position_patient',
                                 'image_orientation_patient']

        self.POSITION = 'position_in_stack'
        self.FILE_PATH = 'file'

    def run(self, path_to_src_dir, path_to_dst_dir, split_by: List[str] = None,
            save_prefix='img'):
        """

        :param path_to_src_dir: path to directory with dicom files (.dcm)
        :param path_to_dst_dir: path to directory where nifti files are saved.
                                It will be created, if it does not already exit.
        :param split_by: Optional name of dicom header fields to separate sub-images
        :param save_prefix: Optional, default 'img'
        """
        file_list = self._get_file_list(path_to_src_dir)
        meta_info_table = self._read_file_info(file_list, split_by)
        meta_info_table[self.POSITION] = meta_info_table.apply(
            self._calculate_position_in_stack, axis=1)

        os.makedirs(path_to_dst_dir, exist_ok=True)
        if split_by:
            img_set_list, meta_info_summary = self._split_images(meta_info_table, split_by)
            for img_id, meta_tab in img_set_list:
                self._load_and_save_single_image(meta_tab, path_to_dst_dir, save_prefix,
                                                 img_numbering=img_id)
                self._save_meta_info_summary(meta_info_summary, path_to_dst_dir, save_prefix)
        else:
            self._load_and_save_single_image(meta_info_table, path_to_dst_dir, save_prefix)


    def _load_and_save_single_image(self, info_table: DataFrame, dst: str, prefix: str,
                             img_numbering: str='') -> None:
        #load image
        info_table.sort_values(self.POSITION, inplace=True, ascending=True)
        image = sitk.ReadImage(list(info_table[self.FILE_PATH]))

        # save image
        file_name =  os.path.join(dst, f'{prefix}_{img_numbering}.nii')
        sitk.WriteImage(image, file_name)

    @staticmethod
    def _save_meta_info_summary(meta_info_tab: DataFrame, dst: str, prefix: str) -> None:
        file_name = os.path.join(dst, f'{prefix}_info.csv')
        meta_info_tab.to_csv(file_name)

    @staticmethod
    def _split_images(tab: DataFrame, group_by):
        img_sets = tab[group_by].drop_duplicates().reset_index(drop=True)
        subset_list = []
        for idx, values in img_sets.iterrows():
            index_col = [True] * tab.shape[0]
            for key, key_value in values.items():
                index_col &= tab[key] == key_value

            subset_list.append((idx, tab.loc[index_col]))

        return subset_list, img_sets

    def _get_file_list(self, path) -> List[str]:
        assert os.path.isdir(path)
        logging.info(f'Convert folder {path} ...')
        files = [f for f in os.listdir(path)
                     if f.endswith(self._file_ending)]
        assert len(files) > 0
        logging.info(f'\t {len(files)} files found')

        files_path = sorted_alphanumeric([os.path.join(path, file) for file in files]) ### added sorted_alphanumeric

        return files_path

    def _read_file_info(self, files: List[str], meta_fields: List[str]) -> DataFrame:

        fields_of_interest = self._position_fields + meta_fields if meta_fields \
            else self._position_fields

        meta_info = []
        reader = sitk.ImageFileReader()
        for file in files:
            reader.SetFileName(file)
            reader.ReadImageInformation()
            meta_info_file = {self.FILE_PATH: file}
            for field in fields_of_interest:
                dcm_field: HeaderEntry = getattr(self._dcm_header, field)
                meta_info_file[field] = dcm_field.conversion(reader.GetMetaData(dcm_field.tag))
            meta_info.append(meta_info_file)

        meta_info = DataFrame(meta_info)
        logging.info(meta_info.head())

        return meta_info

    @staticmethod
    def _calculate_position_in_stack(row):
        """
              Use ImagePositionPatient and ImageOrientationPatient to calculate the
              position of this slice in the image stack
              (can be used to sort the images)

              """
        return np.dot(np.cross(row['image_orientation_patient'][0:3],
                               row['image_orientation_patient'][3:6]),
                      row['image_position_patient'])


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_src_dir')
    parser.add_argument('path_to_dst_dir')
    parser.add_argument('--split_by', type=str, default=None)
    parser.add_argument('--save_prefix', type=str, default='img')
    args = parser.parse_args()

    split_by_arg = args.split_by.split(',') if args.split_by else None

    converter = DicomToNifti()
    converter.run(args.path_to_src_dir, args.path_to_dst_dir,
                  split_by=split_by_arg, save_prefix=args.save_prefix)



### Code from: 
### https://gist.github.com/SeanSyue/8c8ff717681e9ecffc8e43a686e68fd9
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(data, key=alphanum_key)