
import os
import numpy as np
import nibabel as nib

# path
cur_dir = os.getcwd()

PETCT_dir = 'media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task066_petct/imagesTs'
AOR_dir = 'media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task066_petct/labelsTs'
PRED_dir = 'OUTPUT/test/'

# NAMES = os.listdir('sample/')

NAME = '23010018_20141226' # 23010017_20141226, 23010019_20141224, 23010018_20141226

CT_NAME = '{}_0000.nii.gz'.format(NAME)
PET_NAME = '{}_0001.nii.gz'.format(NAME)

AOR_NAME = '{}.nii.gz'.format(NAME)
PRED_NAME = '{}.nii.gz'.format(NAME)

CT_PATH = os.path.join(PETCT_dir, CT_NAME)
PET_PATH = os.path.join(PETCT_dir, PET_NAME)
AOR_PATH = os.path.join(AOR_dir, AOR_NAME)
PRED_PATH = os.path.join(PRED_dir, PRED_NAME)

ptarr = np.array(nib.load(PET_PATH).dataobj)
ctarr = np.array(nib.load(CT_PATH).dataobj)
gtarr = np.array(nib.load(AOR_PATH).dataobj)
predarr = np.array(nib.load(PRED_PATH).dataobj)

size = ([4.07283, 4.07283, 3.])

#Calculate Mean SUV and Max SUV
def get_suv_params(ptarr, roi):
    roi = np.asarray(roi>0, dtype=np.float)
    suvmax = np.max(ptarr*roi)
    suvmean = np.sum(ptarr*roi)/np.sum(roi)
    return suvmax, suvmean
suvmax, suvmean = get_suv_params(ptarr,gtarr)
print('suvmax :', suvmax, 'suvmean :', suvmean)

#Calculate Volume
def get_vol_params(ptzoom, roi):
    roi = np.asarray(roi>0, dtype=np.float)
    return np.prod(ptzoom) * np.sum(roi)
aorvol = get_vol_params(size, gtarr)
print('volume :', aorvol)

# check_petctroi(petarr, ctarr, aorarr)
##
# hdf5 to nifti
import os
import h5py
import nibabel as nib
import numpy as np

import json
from collections import OrderedDict


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

def hdf2nifti(hdf_folder: str, save_folder: str):
    # hdf_folder : [train_dir, test_dir] hdf5 file path
    # save_folder : [imagesTr, imagesTs] Save Folder path
    maybe_mkdir_p(os.path.join(save_folder, 'imagesTr'))
    maybe_mkdir_p(os.path.join(save_folder, 'labelsTr'))
    print('Creating "{}" Image & Label ..'.format(os.path.basename(os.path.normpath(save_folder))))
    hdf5_files = os.listdir(hdf_folder)


    for hdf5_file in hdf5_files:


        hdf5_path = os.path.join(hdf_folder, hdf5_file)

        # image
        f_i = h5py.File(hdf5_path, 'r')
        ctarr = np.asarray(f_i['CT'])
        petarr = np.asarray(f_i['PET'])
        labels = np.asarray(f_i['Aorta'])
        f_i.close()

        _, _, SLICE_COUNT = ctarr.shape
        images = np.empty([200, 200, SLICE_COUNT, 0], dtype=np.single)

        image_ct = np.expand_dims(ctarr, axis=3)
        images = np.append(images, image_ct, axis=3)
        image_pet = np.expand_dims(petarr, axis=3)
        images = np.append(images, image_pet, axis=3)


        hdf5_file_NAME = hdf5_file

        niim = nib.Nifti1Image(images, affine=np.eye(4))
        nib.save(niim, os.path.join(save_folder, 'imagesTr/{}.nii.gz'.format(hdf5_file[:-8])))

        nila = nib.Nifti1Image(labels, affine=np.eye(4))
        nib.save(nila, os.path.join(save_folder, 'labelsTr/{}.nii.gz'.format(hdf5_file[:-8])))


    print('"{}" Image & Label Completed !!'.format(os.path.basename(os.path.normpath(save_folder))))

hdf2nifti('sample/files', 'sample/saves')

## Json creating

def json_mk(save_dir: str):
    # Path
    imagesTr = os.path.join(save_dir, 'imagesTr')
    imagesTs = os.path.join(save_dir, 'imagesTs')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(imagesTs)

    overwrite_json_file = True
    json_file_exist = False

    if os.path.exists(os.path.join(save_dir, 'dataset.json')):
        print('dataset.json already exist!')
        json_file_exist = True

    if json_file_exist == False or overwrite_json_file:

        json_dict = OrderedDict()
        json_dict['name'] = "PETCT"
        json_dict['description'] = "Medical Image AI Challenge 2021"
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = "https://maic.or.kr/competitions/"
        json_dict['licence'] = "SNUH"
        json_dict['release'] = "18/10/2021"

        json_dict['modality'] = {
            "0": "CT",
            "1": "PET"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "Aorta"
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


##

def check_petctroi(ptarr_, ctarr_, roi):
    zind = int(np.median(np.where(roi)[2]))
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(ctarr_[:, :, zind], cmap='gray', vmin=-150, vmax=250)
    plt.imshow(ptarr_[:, :, zind], cmap='hot', alpha=0.3)
    plt.imshow(roi[:, :, zind], alpha=0.6, cmap='Greens')
    plt.axis('off')

    xind = int(np.median(np.where(roi)[0]))
    plt.subplot(1, 3, 2)
    plt.imshow(np.rot90(ctarr_[xind, :, :]), cmap='gray', vmin=-150, vmax=250)
    plt.imshow(np.rot90(ptarr_[xind, :, :]), alpha=0.3, cmap='hot')
    plt.imshow(np.rot90(roi[xind, :, :]), alpha=0.6, cmap='Greens')
    plt.axis('off')

    yind = int(np.median(np.where(roi)[1]))
    plt.subplot(1, 3, 3)
    plt.imshow(np.rot90(ctarr_[:, yind, :]), cmap='gray', vmin=-150, vmax=250)
    plt.imshow(np.rot90(ptarr_[:, yind, :]), alpha=0.3, cmap='hot')
    plt.imshow(np.rot90(roi[:, yind, :]), alpha=0.6, cmap='Greens')
    plt.axis('off')

    plt.show()


##
#Calculate Mean SUV and Max SUV
def get_suv_params(ptarr, roi):
    roi = np.asarray(roi>0, dtype=np.float)
    suvmax = np.max(ptarr*roi)
    suvmean = np.sum(ptarr*roi)/np.sum(roi)
    return suvmax, suvmean
suvmax, suvmean = get_suv_params(ptarr,roi)
print('suvmax :', suvmax, 'suvmean :', suvmean)

#Calculate Volume
def get_vol_params(ptzoom, roi):
    roi = np.asarray(roi>0, dtype=np.float)
    return np.prod(ptzoom) * np.sum(roi)
aorvol = get_vol_params(size, roi)
print('volume :', aorvol)