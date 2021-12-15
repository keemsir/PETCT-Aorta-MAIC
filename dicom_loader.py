import nibabel as nib
import os
import numpy as np
import dicom2nifti
import pydicom

cur_dir = os.getcwd()

Patient_Number = '33451492'

# nii_dir = 'media/ncc/Tasks/Task01_BrainTumour/labelsTr/'
# nii_dir = 'media/ncc/Tasks/Task77_KidneyTumour/labelsTr/'
dicom_dir = 'sample/{}/'.format(Patient_Number)
# nii_name = 'BRATS_001.nii.gz'
# nii_name = 'training010.nii.gz'
# nii_path = os.path.join(cur_dir, nii_dir+nii_name)

# nib_proxy = nib.load(nii_path)
# print(nib_proxy.shape)
# print(nib_proxy.affine)

# nib_image = np.array(nib.load(nii_path).dataobj)

# affine = ([[-1., 0., 0., 255.5], [0., 1., 0., -255.5], [0., 0., 1., -31.5], [0., 0., 0., 1.]])

# np.set_printoptions(threshold=sys.maxsize) # sys.maxsize full size array
##
import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
import pydicom
import json
from collections import OrderedDict

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

base_dir = '/home/ncc/PycharmProjects/nnUNet/'
KiTS_dir = os.path.join(base_dir, 'tests/KiTS/')

dicom_dir = os.path.join(KiTS_dir, 'train/DICOM/training001/')
train_dir = os.path.join(KiTS_dir, 'train/DICOM/')
label_dir = os.path.join(KiTS_dir, 'train/Label/')
test_dir = os.path.join(KiTS_dir, 'test/DICOM/')
save_dir = os.path.join(KiTS_dir, 'nifti/')
dcm_file = os.path.join(KiTS_dir, 'train/DICOM/training002/00019.dcm')
png_dir = os.path.join(KiTS_dir, 'train/Label/training001/00039.png')

# def png2nifti
npy = sitk.GetArrayFromImage(sitk.ReadImage(png_dir))

imagesTr = os.path.join(save_dir, 'imagesTr/')
labelsTr = os.path.join(save_dir, 'labelsTr/')
imagesTs = os.path.join(save_dir, 'imagesTs/')



# Dicom to nifti

def dcm2nifti(dcm_folder: str, save_folder: str):
    # dcm_folder : [train_dir, test_dir] dicom file path
    # save_folder : [imagesTr, imagesTs] Save Folder path
    maybe_mkdir_p(save_folder)
    print('Creating "{}" Image..'.format(os.path.basename(os.path.normpath(save_folder))))
    DCM_files = os.listdir(dcm_folder)
    for DCM_file in DCM_files:
        images = np.empty([512, 512, 0], dtype=np.single)
        DCM_path = os.path.join(dcm_folder, DCM_file)
        DCM_path_sort = sorted(os.listdir(DCM_path))
        for f in DCM_path_sort:
            os.path.join(DCM_path, f)
            f_dcm = pydicom.read_file(os.path.join(DCM_path, f))
            slice = f_dcm.pixel_array
            f_slope = f_dcm.RescaleSlope
            f_intercept = f_dcm.RescaleIntercept
            image = (slice*f_slope + f_intercept)
            image = np.expand_dims(image, axis=2)
            images = np.append(images, image, axis=2)

        niim = nib.Nifti1Image(images, affine=np.eye(4))
        nib.save(niim, os.path.join(save_folder, '{}.nii.gz'.format(DCM_file)))
    print('"{}" Image Completed !!'.format(os.path.basename(os.path.normpath(save_folder))))

'''
# Image Test
DCM_files = os.listdir(test_dir)
for DCM_file in DCM_files:
    images = np.empty([512, 512, 0], dtype=np.single)
    DCM_path = os.path.join(test_dir, DCM_file)
    DCM_path_sort = sorted(os.listdir(DCM_path))
    for f in DCM_path_sort:
        os.path.join(DCM_path, f)
        f_dcm = pydicom.read_file(os.path.join(DCM_path, f))
        slice = f_dcm.pixel_array
        f_slope = f_dcm.RescaleSlope
        f_intercept = f_dcm.RescaleIntercept
        image = (slice*f_slope + f_intercept)
        image = np.expand_dims(image, axis=2)
        images = np.append(images, image, axis=2)

    niim = nib.Nifti1Image(images, affine=np.eye(4))
    nib.save(niim, os.path.join(imagesTs, '{}.nii.gz'.format(DCM_file)))
'''

# png2nifti
def png2nifti(png_folder: str, save_folder: str):
    # png_folder : png file folder
    # save_folder : Save folder path
    DCM_files = os.listdir(png_folder)
    print('Creating "{}" Label..'.format(os.path.basename(os.path.normpath(save_folder))))
    maybe_mkdir_p(save_folder)
    for DCM_file in DCM_files:
        labels = np.empty([512, 512, 0], dtype=np.uint8)
        DCM_path = os.path.join(png_folder, DCM_file)
        DCM_path_sort = sorted(os.listdir(DCM_path))
        for f in DCM_path_sort:
            os.path.join(DCM_path, f)
            f_png = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(DCM_path, f)))

            label = np.expand_dims(f_png, axis=2)
            labels = np.append(labels, label, axis=2)

        nila = nib.Nifti1Image(labels, affine=np.eye(4))
        nib.save(nila, os.path.join(save_folder, '{}.nii.gz'.format(DCM_file)))
    print('"{}" Label Completed !!'.format(os.path.basename(os.path.normpath(save_folder))))


dcm2nifti(train_dir, imagesTr)
dcm2nifti(test_dir, imagesTs)

png2nifti(label_dir, labelsTr)


# Json creating

overwrite_json_file = True
json_file_exist = False

if os.path.exists(os.path.join(save_dir, 'dataset.json')):
    print('dataset.json already exist!')
    json_file_exist = True

if json_file_exist == False or overwrite_json_file:

    json_dict = OrderedDict()
    json_dict['name'] = "Kidney"
    json_dict['description'] = "MOAI 2021 Body Morphometry AI Segmentation Online Challenge"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "https://www.kaggle.com/c/body-morphometry-kidney-and-tumor/data"
    json_dict['licence'] = "CC-BY-NC-SA"
    json_dict['release'] = "07/09/2021"

    json_dict['modality'] = {
        "0": "CT"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "kidney",
        "2": "tumor"
    }

    train_ids = sorted(os.listdir(imagesTr))
    test_ids = sorted(os.listdir(imagesTs))
    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = len(test_ids)

    json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in train_ids]

    json_dict['test'] = ["./imagesTs/%s" % i for i in test_ids] #(i[:i.find("_0000")])

    with open(os.path.join(save_dir, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=False)

    if os.path.exists(os.path.join(save_dir, 'dataset.json')):
        if json_file_exist == False:
            print('dataset.json created!')
        else:
            print('dataset.json overwritten!')


## Visualization
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

cur_dir = os.getcwd()
test_kind = ['imagesTr', 'imagesTs', 'labelsTr']
kind_NUM = 0
test1_dir = 'media/ncc/Tasks/Task98_test/{}/'.format(test_kind[kind_NUM])
test2_dir = 'media/ncc/Tasks/Task77_KidneyTumour/{}/'.format(test_kind[kind_NUM])

test_name = 'training060.nii.gz'
# test_name = 'test030.nii.gz'
RESULT_LEN_RAN = np.random.randint(0, 60)
# RESULT_LEN_RAN = 15

nii_path1 = os.path.join(cur_dir, test1_dir+test_name)
nii_path2 = os.path.join(cur_dir, test2_dir+test_name)

test1 = np.array(nib.load(nii_path1).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]
test2 = np.array(nib.load(nii_path2).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]


max_rows = 2
max_cols = 5

fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20, 8))
for idx in range(max_cols):
    axes[0, idx].axis("off")
    axes[0, idx].set_title('Test1_' + str(idx + 1) + '_{}'.format(RESULT_LEN_RAN + idx))
    axes[0, idx].imshow(test1[:, :, idx])
for idx in range(max_cols):
    axes[1, idx].axis("off")
    axes[1, idx].set_title('Test2_' + str(idx + 1) + '_{}'.format(RESULT_LEN_RAN + idx))
    axes[1, idx].imshow(test2[:, :, idx])

