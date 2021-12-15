
import h5py, glob
import matplotlib.pyplot as plt
import numpy as np

h5files = glob.glob('sample/files/*')
print(h5files[2])
f = h5py.File(h5files[2],'r') #Example File

ptarr = np.asarray(f['PET'])
ctarr = np.asarray(f['CT'])
roi = np.asarray(f['Aorta'])
size = np.asarray(f['Size'])

f.close()

def check_petctroi(ptarr_,ctarr_,roi):
    zind = int(np.median(np.where(roi)[2]))
    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(ctarr_[:,:,zind], cmap='gray', vmin=-150, vmax=250)
    plt.imshow(ptarr_[:,:,zind],cmap='hot', alpha=0.3)
    plt.imshow(roi[:,:,zind], alpha=0.6, cmap='Greens')
    plt.axis('off')
    
    xind = int(np.median(np.where(roi)[0]))
    plt.subplot(1,3,2)
    plt.imshow(np.rot90(ctarr_[xind,:,:]), cmap='gray', vmin=-150, vmax=250)
    plt.imshow(np.rot90(ptarr_[xind,:,:]),alpha=0.3, cmap='hot')
    plt.imshow(np.rot90(roi[xind,:,:]), alpha=0.6, cmap='Greens')
    plt.axis('off')
    
    yind = int(np.median(np.where(roi)[1]))
    plt.subplot(1,3,3)
    plt.imshow(np.rot90(ctarr_[:,yind,:]), cmap='gray', vmin=-150, vmax=250)
    plt.imshow(np.rot90(ptarr_[:,yind,:]),alpha=0.3, cmap='hot')
    plt.imshow(np.rot90(roi[:,yind,:]), alpha=0.6, cmap='Greens')
    plt.axis('off')
    
    plt.show()
    
check_petctroi(ptarr,ctarr,roi)


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
#####Final Outputs are SUVmean and aorvol
