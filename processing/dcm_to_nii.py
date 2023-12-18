from processing.image_preprocessing.dicom_to_nifti import DicomToNifti


def convert_dicom_to_nifti(src_dir, dst_dir, save_prefix):
    converter = DicomToNifti()
    converter.run(src_dir, dst_dir, save_prefix = save_prefix)


