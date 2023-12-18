# image_preprocessing
Collection of scripts for image preprocessing

---
### Set up conda environment:
You can use the environment.yml file to set up a new conda environment and install the necessary packages
Just run the following command in a terminal
```
conda env create --file environment.yml
```
---

## Dicom to nifti

Use within a python file:

```python
from dicom_to_nifti import DicomToNifti

src_dir = 'path/to/dcm/files'
dst_dir = 'path/to/destination/folder'

converter = DicomToNifti()

# if the source folder contains a single 3d image:
converter.run(src_dir, dst_dir)

# if the source folder contains several 3d images:
# use the parameter 'split_by' to set the dicom tags that differentiate the different sub images
# Check utils/dicom_header for currently implemented options
converter.run(src_dir, dst_dir, split_by=['echo_time', 'acquisition_time'])

# you can use the optional parameter save_prefix to change the default prefix 'img'

```

Use from the command line:
```
python dicom_to_nifti.py   path/to/src path/to/dst 

python dicom_to_nifti.py --split_by=echo_time --save_prefix=img  path/to/src path/to/dst 
python dicom_to_nifti.py --splitby=echo_time,acquisition_time path/to/src path/to/dst
```
**Note**: If you want to split by several properties, these need to be listed separated by a comma (,) without spaces!


## Resample image

You can use the function: SimpleITK.Resample (function) or SimpleITK.ResampleImageFilter (class)

Here is an example with SimpleITK.ResampleImageFilter:
```python
import SimpleITK as sitk
img = sitk.ReadImage('path/to/image.nii')
reference_image = sitk.ReadImage('path/to/reference/image.nii')

resample = sitk.ResampleImageFilter()
resample.SetReferenceImage(reference_image)
resample.SetInterpolator(sitk.sitkBSpline) # other option sitk.sitkLinear, sitk.sitkNearestNeighbor
resampled_image = resample.Execute(img) # Run the resampling

sitk.WriteImage(resampled_image, 'path/to/dst/img.nii') # save the result

```

