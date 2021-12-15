## Train-image Visualization (2d_predict, 3d_cascade_predict, 3d_fullres_predict, 509_ensemble)

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def view_nii(image_num: str, path: str):
    # Example image_num = 56, path = 'media/ncc/Tasks/Task77_KidneyTumour'
    cur_dir = os.getcwd()
    main_path = os.path.join(cur_dir, path)
    train_image_dir = os.path.join(main_path, 'imagesTr')
    train_label_dir = os.path.join(main_path, 'labelsTr')
    test_image_dir = os.path.join(main_path, 'imagesTs')

    TRAIN_NAME = 'training{}.nii.gz'.format(NUM)
    TEST_NAME = 'test{}.nii.gz'.format(NUM)

    pass


# path

cur_dir = os.getcwd()

main_path = os.path.join(cur_dir, 'media/ncc/Tasks/Task77_KidneyTumour')

train_image_dir = os.path.join(main_path, 'imagesTr')
train_label_dir = os.path.join(main_path, 'labelsTr')
test_image_dir = os.path.join(main_path, 'imagesTs')

NUM = '056'

TRAIN_NAME = 'training{}.nii.gz'.format(NUM)
TEST_NAME = 'test{}.nii.gz'.format(NUM)

pred1 = os.path.join(train_image_dir, TRAIN_NAME)
pred2 = os.path.join(train_label_dir, TRAIN_NAME)

RESULT_LEN_RAN = np.random.randint(0, 63)
# RESULT_LEN_RAN = 80 # custom number
RESULT_LEN_RAN = 15
# custom number


pr_label1 = np.array(nib.load(pred1).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]
pr_label2 = np.array(nib.load(pred2).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]

max_rows = 3
max_cols = 5

fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20, 8))
for idx in range(max_cols):
    axes[0, idx].axis("off")
    axes[0, idx].set_title('Image' + str(idx + 1) + '_{}'.format(RESULT_LEN_RAN + idx))
    axes[0, idx].imshow(pr_label1[:, :, idx])
for idx in range(max_cols):
    axes[1, idx].axis("off")
    axes[1, idx].set_title('Label' + str(idx + 1) + '_{}'.format(RESULT_LEN_RAN + idx))
    axes[1, idx].imshow(pr_label2[:, :, idx])
for idx in range(max_cols):
    axes[2, idx].axis("off")
    axes[2, idx].set_title('Label' + str(idx + 1) + '_{}'.format(RESULT_LEN_RAN + idx))
    axes[2, idx].imshow(pr_label2[:, :, idx])

plt.suptitle('Train image NUM : training{}.nii.gz'.format(NUM))
plt.subplots_adjust(wspace=.1, hspace=.2)
plt.show()

## Check petctroi (PET, CT, GroundTruth) -- GroundTruth Ver

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# path
cur_dir = os.getcwd()

PETCT_dir = 'media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task066_petct/imagesTr'
AOR_dir = 'media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task066_petct/labelsTr'
PRED_dir = 'OUTPUT/'

# NAMES = os.listdir('sample/')

NAME = '23010019_20141224' # 23010017_20141226, 23010019_20141224

CT_NAME = '{}_0000.nii.gz'.format(NAME)
PET_NAME = '{}_0001.nii.gz'.format(NAME)

AOR_NAME = '{}.nii.gz'.format(NAME)
PRED_NAME = '{}.nii.gz'.format(NAME)

CT_PATH = os.path.join(PETCT_dir, CT_NAME)
PET_PATH = os.path.join(PETCT_dir, PET_NAME)
AOR_PATH = os.path.join(AOR_dir, AOR_NAME)

ptarr = np.array(nib.load(PET_PATH).dataobj)
ctarr = np.array(nib.load(CT_PATH).dataobj)
gtarr = np.array(nib.load(AOR_PATH).dataobj)

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


check_petctroi(ptarr, ctarr, gtarr)


## Check petctroi (PET, CT, GroundTruth) -- PRED Ver

import os
import numpy as np
import nibabel as nib

# path
cur_dir = os.getcwd()

PETCT_dir = 'media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task066_petct/imagesTs'
AOR_dir = 'sample/Task66_petct/labelsTs'
PRED_dir = 'OUTPUT/test'

# NAMES = os.listdir('sample/')

NAME = '23010018_20141226' # 23010017_20141226, 23010019_20141224

CT_NAME = '{}_0000.nii.gz'.format(NAME)
PET_NAME = '{}_0001.nii.gz'.format(NAME)

AOR_NAME = '{}.nii.gz'.format(NAME)
PRED_NAME = '{}.nii.gz'.format(NAME)

CT_PATH = os.path.join(PETCT_dir, CT_NAME)
PET_PATH = os.path.join(PETCT_dir, PET_NAME)
AOR_PATH = os.path.join(PRED_dir, AOR_NAME)

ptarr = np.array(nib.load(PET_PATH).dataobj)
ctarr = np.array(nib.load(CT_PATH).dataobj)
aorarr = np.array(nib.load(AOR_PATH).dataobj)

def check_petctroi(ptarr_, ctarr_, roi):
    zind = int(np.median(np.where(roi)[2]))
    print(zind)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(ctarr_[:, :, zind], cmap='gray', vmin=-150, vmax=250)
    plt.imshow(ptarr_[:, :, zind], cmap='hot', alpha=0.3)
    plt.imshow(roi[:, :, zind], alpha=0.6, cmap='Greens')
    plt.axis('off')

    xind = int(np.median(np.where(roi)[0]))
    print(xind)
    plt.subplot(1, 3, 2)
    plt.imshow(np.rot90(ctarr_[xind, :, :]), cmap='gray', vmin=-150, vmax=250)
    plt.imshow(np.rot90(ptarr_[xind, :, :]), alpha=0.3, cmap='hot')
    plt.imshow(np.rot90(roi[xind, :, :]), alpha=0.6, cmap='Greens')
    plt.axis('off')

    yind = int(np.median(np.where(roi)[1]))
    print(yind)
    plt.subplot(1, 3, 3)
    plt.imshow(np.rot90(ctarr_[:, yind, :]), cmap='gray', vmin=-150, vmax=250)
    plt.imshow(np.rot90(ptarr_[:, yind, :]), alpha=0.3, cmap='hot')
    plt.imshow(np.rot90(roi[:, yind, :]), alpha=0.6, cmap='Greens')
    plt.axis('off')

    plt.show()


check_petctroi(ptarr, ctarr, aorarr)



## Cross Validation (pp or raw) Visualization (2d_predict, 3d_cascade_predict, 3d_fullres_predict, 509_ensemble)

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# path
IMG_NUM = 1

cur_dir = os.getcwd()
cv_path = os.path.join(cur_dir, 'media/input/77_KidneyTumour/labelsTs')

task_dir = os.path.join(cur_dir, 'media/input/77_KidneyTumour/')
output_dir = os.path.join(cur_dir, 'OUTPUT_DIRECTORY/577/')

RESULT_LEN_RAN = 40

gt_dir = os.path.join(cv_path, 'gt_niftis')
gt_list = os.listdir(gt_dir)
gt_list.sort()
gt_name = gt_list[IMG_NUM]
gt_img = np.array(nib.load(os.path.join(gt_dir, gt_name)).dataobj)
gt_img_range = np.array(nib.load(os.path.join(gt_dir, gt_name)).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]

cvraw_dir = os.path.join(cv_path, 'cv_niftis_raw')
cv_raw_list = os.listdir(cvraw_dir)
cv_raw_list.sort()
cv_raw_name = cv_raw_list[IMG_NUM+1]
cv_raw_img = np.array(nib.load(os.path.join(cvraw_dir, cv_raw_name)).dataobj)
cv_raw_img_range = np.array(nib.load(os.path.join(cvraw_dir, cv_raw_name)).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]

cvpp_dir = os.path.join(cv_path, 'cv_niftis_postprocessed')
cv_pp_list = os.listdir(cvpp_dir)
cv_pp_list.sort()
cv_pp_name = cv_pp_list[IMG_NUM+1]
cv_pp_img = np.array(nib.load(os.path.join(cvpp_dir, cv_pp_name)).dataobj)
cv_pp_img_range = np.array(nib.load(os.path.join(cvpp_dir, cv_pp_name)).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]

print('CV Row Image Shape: {}'.format(cv_raw_name), cv_raw_img.shape)
print('CV PP Image Shape: {}'.format(cv_pp_name), cv_pp_img.shape)

max_rows = 4
max_cols = test_img.shape[2]

fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20, 8))
for idx in range(max_cols):
    axes[0, idx].axis("off")
    axes[0, idx].set_title('Test Image' + str(idx + 1) + '_{}'.format(RESULT_LEN_RAN + idx))
    axes[0, idx].imshow(test_img[:, :, idx])
for idx in range(max_cols):
    axes[1, idx].axis("off")
    axes[1, idx].set_title('image1_{}_'.format('GroundTruth') + str(idx + 1))
    axes[1, idx].imshow(gt_img_range[:, :, idx])
for idx in range(max_cols):
    axes[2, idx].axis("off")
    axes[2, idx].set_title('image2_{}_'.format('CV Raw') + str(idx + 1))
    axes[2, idx].imshow(cv_raw_img_range[:, :, idx])
for idx in range(max_cols):
    axes[3, idx].axis("off")
    axes[3, idx].set_title('image3_{}_'.format('CV PP') + str(idx + 1))
    axes[3, idx].imshow(cv_pp_img_range[:, :, idx])

plt.suptitle('Path : {}, Image size : {}'.format(output_dir, image_size))
plt.subplots_adjust(wspace=.1, hspace=.2)
plt.show()


## 5 Case Test-image Visualization (2d_predict, 3d_cascade_predict, 3d_fullres_predict, 509_ensemble)

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# path
cur_dir = os.getcwd()

NUM = '001'

TASK_NAME = 'test{}.nii.gz'.format(NUM)
TASK_NAME_0 = 'test{}_0000.nii.gz'.format(NUM)


task_dir = os.path.join(cur_dir, 'media/input/77_KidneyTumour/')

PRED1_NAME = '3d_fullres_e150'
PRED2_NAME = '3d_CEGDL'
PRED3_NAME = '3d_CEGDL'
PRED4_NAME = '3d_CEGDL'
PRED5_NAME = '3d_CEGDL'

pred1 = os.path.join(os.path.join(cur_dir, 'OUTPUT_DIRECTORY/577/'), '{}/{}'.format(PRED1_NAME, TASK_NAME))
pred2 = os.path.join(os.path.join(cur_dir, 'OUTPUT_DIRECTORY/577/'), '{}/{}'.format(PRED2_NAME, TASK_NAME))
pred3 = os.path.join(os.path.join(cur_dir, 'OUTPUT_DIRECTORY/577/'), '{}/{}'.format(PRED3_NAME, TASK_NAME))
pred4 = os.path.join(os.path.join(cur_dir, 'OUTPUT_DIRECTORY/577/'), '{}/{}'.format(PRED4_NAME, TASK_NAME))
pred5 = os.path.join(os.path.join(cur_dir, 'OUTPUT_DIRECTORY/577/'), '{}/{}'.format(PRED5_NAME, TASK_NAME))


ts_image = os.path.join(task_dir, 'imagesTs/{}'.format(TASK_NAME_0))
pred_label = os.path.join(task_dir, TASK_NAME_0)

image_size = np.array(nib.load(ts_image).dataobj).shape

result_img = np.array(nib.load(ts_image).dataobj)
_, _, RESULT_LEN = result_img.shape
RESULT_LEN_RAN = np.random.randint(0, RESULT_LEN)
# RESULT_LEN_RAN = 80 # custom number
# RESULT_LEN_RAN = 37

test_img = np.array(nib.load(ts_image).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]

pr_label1 = np.array(nib.load(pred1).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]
pr_label2 = np.array(nib.load(pred2).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]



max_rows = 3
max_cols = test_img.shape[2]

fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20, 8))
for idx in range(max_cols):
    axes[0, idx].axis("off")
    axes[0, idx].set_title('Test Image' + str(idx + 1) + '_{}'.format(RESULT_LEN_RAN + idx))
    axes[0, idx].imshow(test_img[:, :, idx])
for idx in range(max_cols):
    axes[1, idx].axis("off")
    axes[1, idx].set_title('image1_{}_'.format(PRED1_NAME) + str(idx + 1))
    axes[1, idx].imshow(pr_label1[:, :, idx])
for idx in range(max_cols):
    axes[2, idx].axis("off")
    axes[2, idx].set_title('image2_{}_'.format(PRED2_NAME) + str(idx + 1))
    axes[2, idx].imshow(pr_label2[:, :, idx])


plt.suptitle('Train image NUM : training{}.nii.gz'.format(NUM))
plt.subplots_adjust(wspace=.1, hspace=.2)
plt.show()


## png reader
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


image = Image.open('/home/ncc/PycharmProjects/nnUNet/png/test/test001/00064.png')
np_array = np.array(image)

plt.imshow(np_array, cmap='gray')

# plt.savefig("rgb_image_rotation_scipy_matplotlib_02.png", bbox_inches='tight', dpi=100)

plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# path
cur_dir = os.getcwd()

NUM = '001'

TASK_NAME = 'test{}.nii.gz'.format(NUM)
TASK_NAME_0 = 'test{}_0000.nii.gz'.format(NUM)


task_dir = os.path.join(cur_dir, 'media/input/77_KidneyTumour/')

PRED1_NAME = '3d_fullres_e150'
PRED2_NAME = '3d_CEGDL'
PRED3_NAME = '3d_CEGDL'
PRED4_NAME = '3d_CEGDL'
PRED5_NAME = '3d_CEGDL'

pred1 = os.path.join(os.path.join(cur_dir, 'OUTPUT_DIRECTORY/577/'), '{}/{}'.format(PRED1_NAME, TASK_NAME))
pred2 = os.path.join(os.path.join(cur_dir, 'OUTPUT_DIRECTORY/577/'), '{}/{}'.format(PRED2_NAME, TASK_NAME))
pred3 = os.path.join(os.path.join(cur_dir, 'OUTPUT_DIRECTORY/577/'), '{}/{}'.format(PRED3_NAME, TASK_NAME))
pred4 = os.path.join(os.path.join(cur_dir, 'OUTPUT_DIRECTORY/577/'), '{}/{}'.format(PRED4_NAME, TASK_NAME))
pred5 = os.path.join(os.path.join(cur_dir, 'OUTPUT_DIRECTORY/577/'), '{}/{}'.format(PRED5_NAME, TASK_NAME))


ts_image = os.path.join(task_dir, 'imagesTs/{}'.format(TASK_NAME_0))
pred_label = os.path.join(task_dir, TASK_NAME_0)

image_size = np.array(nib.load(ts_image).dataobj).shape

result_img = np.array(nib.load(ts_image).dataobj)
_, _, RESULT_LEN = result_img.shape
RESULT_LEN_RAN = np.random.randint(0, RESULT_LEN)
# RESULT_LEN_RAN = 80 # custom number
# RESULT_LEN_RAN = 37

test_img = np.array(nib.load(ts_image).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]

pr_label1 = np.array(nib.load(pred1).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]
pr_label2 = np.array(nib.load(pred2).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]



max_rows = 3
max_cols = test_img.shape[2]

fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20, 8))
for idx in range(max_cols):
    axes[0, idx].axis("off")
    axes[0, idx].set_title('Test Image' + str(idx + 1) + '_{}'.format(RESULT_LEN_RAN + idx))
    axes[0, idx].imshow(test_img[:, :, idx])
for idx in range(max_cols):
    axes[1, idx].axis("off")
    axes[1, idx].set_title('image1_{}_'.format(PRED1_NAME) + str(idx + 1))
    axes[1, idx].imshow(pr_label1[:, :, idx])
for idx in range(max_cols):
    axes[2, idx].axis("off")
    axes[2, idx].set_title('image2_{}_'.format(PRED2_NAME) + str(idx + 1))
    axes[2, idx].imshow(pr_label2[:, :, idx])



plt.suptitle('Train image NUM : training{}.nii.gz'.format(NUM))
plt.subplots_adjust(wspace=.1, hspace=.2)
plt.show()

## Visualization (Split modality version)

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# path
IMG_NUM = 1

cur_dir = os.getcwd()
cv_path = os.path.join(cur_dir, 'media/input/77_KidneyTumour/labelsTs')

task_dir = os.path.join(cur_dir, 'media/input/77_KidneyTumour/')
output_dir = os.path.join(cur_dir, 'OUTPUT_DIRECTORY/577/')

RESULT_LEN_RAN = 40

gt_dir = os.path.join(cv_path, 'gt_niftis')
gt_list = os.listdir(gt_dir)
gt_list.sort()
gt_name = gt_list[IMG_NUM]
gt_img = np.array(nib.load(os.path.join(gt_dir, gt_name)).dataobj)
gt_img_range = np.array(nib.load(os.path.join(gt_dir, gt_name)).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]

cvraw_dir = os.path.join(cv_path, 'cv_niftis_raw')
cv_raw_list = os.listdir(cvraw_dir)
cv_raw_list.sort()
cv_raw_name = cv_raw_list[IMG_NUM+1]
cv_raw_img = np.array(nib.load(os.path.join(cvraw_dir, cv_raw_name)).dataobj)
cv_raw_img_range = np.array(nib.load(os.path.join(cvraw_dir, cv_raw_name)).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]

cvpp_dir = os.path.join(cv_path, 'cv_niftis_postprocessed')
cv_pp_list = os.listdir(cvpp_dir)
cv_pp_list.sort()
cv_pp_name = cv_pp_list[IMG_NUM+1]
cv_pp_img = np.array(nib.load(os.path.join(cvpp_dir, cv_pp_name)).dataobj)
cv_pp_img_range = np.array(nib.load(os.path.join(cvpp_dir, cv_pp_name)).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]

print('CV Row Image Shape: {}'.format(cv_raw_name), cv_raw_img.shape)
print('CV PP Image Shape: {}'.format(cv_pp_name), cv_pp_img.shape)

max_rows = 4
max_cols = test_img.shape[2]

fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20, 8))
for idx in range(max_cols):
    axes[0, idx].axis("off")
    axes[0, idx].set_title('Test Image' + str(idx + 1) + '_{}'.format(RESULT_LEN_RAN + idx))
    axes[0, idx].imshow(test_img[:, :, idx])
for idx in range(max_cols):
    axes[1, idx].axis("off")
    axes[1, idx].set_title('image1_{}_'.format('GroundTruth') + str(idx + 1))
    axes[1, idx].imshow(gt_img_range[:, :, idx])
for idx in range(max_cols):
    axes[2, idx].axis("off")
    axes[2, idx].set_title('image2_{}_'.format('CV Raw') + str(idx + 1))
    axes[2, idx].imshow(cv_raw_img_range[:, :, idx])
for idx in range(max_cols):
    axes[3, idx].axis("off")
    axes[3, idx].set_title('image3_{}_'.format('CV PP') + str(idx + 1))
    axes[3, idx].imshow(cv_pp_img_range[:, :, idx])

plt.suptitle('Path : {}, Image size : {}'.format(output_dir, image_size))
plt.subplots_adjust(wspace=.1, hspace=.2)
plt.show()


##
import os
import re

txt_path = './txt'

x_path = os.path.join(txt_path, '2d.txt')
y_path = os.path.join(txt_path, '3d.txt')

f = open(x_path)

data = f.readlines()

epoch_list = []
training_list = []
validation_list = []
dice_list = []


for i in range(len(data)):
    num_line1 = re.findall('TRAINING', data[i])


num_line1 = re.findall('TRAINING', data[1])


##
