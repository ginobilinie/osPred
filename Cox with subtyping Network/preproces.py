import os
import json
import pickle
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
from scipy.ndimage import binary_fill_holes
from utils.utils import get_bbox
from core.config import config


def preprocess_BraTS19(raw_dir, preprocessed_dir, normalization):
    names = []
    theshape = [0, 0, 0]
    modalities = ['flair', 't1', 't1ce', 't2', 'seg']

    with open(os.path.join(preprocessed_dir, 'dataset.json'), 'r') as f:
        dataset_json = json.load(f)
    modality = dataset_json['modality']
    training = dataset_json['training']
    for case in training:
        images = []
        age = case['age']
        ostime = case['ostime']
        if os.path.exists(os.path.join(preprocessed_dir, os.path.basename(case['image'][4:-7]) + '.npz')):
            name = os.path.basename(case['image'][4:-7])
            names.append(name)
            continue
        for i in range(len(modality)):
            image = sitk.ReadImage(os.path.join(raw_dir, case['image'][:-7], case['image'][4:-7] + '_' + modalities[i] + '.nii.gz'))
            image = sitk.GetArrayFromImage(image).astype('float')
            images.append(image)
        image = np.stack(images)
        name = os.path.basename(case['image'][4:-7])
        names.append(name)

        label = sitk.ReadImage(os.path.join(raw_dir, case['image'][:-7], case['image'][4:-7] + '_' + modalities[4] + '.nii.gz'))
        label = sitk.GetArrayFromImage(label)

        # crop to non_zero regions
        mask = np.zeros(image.shape[1:], dtype=bool)
        for i in range(len(modality)):
            mask = mask | (image[i] != 0)
        mask = binary_fill_holes(mask)
        l_box = get_bbox(label)
        theshape[0] = max(theshape[0], l_box[0].stop - l_box[0].start)
        theshape[1] = max(theshape[1], l_box[1].stop - l_box[1].start)
        theshape[2] = max(theshape[2], l_box[2].stop - l_box[2].start)
        image = image[:, l_box[0], l_box[1], l_box[2]]
        label = label[l_box[0], l_box[1], l_box[2]]
        mask = mask[l_box[0], l_box[1], l_box[2]]

        # intensity normalization within foreground
        for i in range(len(modality)):
            if normalization == 'min-max':
                image[i][mask] = (image[i][mask] - image[i][mask].min()) / (image[i][mask].max() - image[i][mask].min())
            elif normalization == 'z-score':
                image[i][mask] = (image[i][mask] - image[i][mask].mean()) / (image[i][mask].std() + 1e-8)
            else:
                raise NotImplementedError('Only min-max and z-score normalization are supported !!!')
            image[i][mask == 0] = 0  # all modalities share same background

        # label should not exist at background region
        label[mask == 0] = 0

        # save preprocessed data and selected locations
        data = np.concatenate([image, label[np.newaxis]])

        if not os.path.exists(os.path.join(preprocessed_dir, name + '.npz')):
            np.savez(os.path.join(preprocessed_dir, name+'.npz'), data=data.astype('float32'), data1=age, data2=ostime)  # 保存的是，image和label的cat、death、ostime

    print(len(names))
    names = np.sort(names)
    vals = names
    splits = OrderedDict()
    splits['val'] = vals

    with open(os.path.join(preprocessed_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)



if __name__ == '__main__':
    raw_dir = config.RAWDATA
    preprocessed_dir = 'DATA/BraTS19'
    if not os.path.exists(preprocessed_dir):
        os.mkdir(preprocessed_dir)
    preprocess_BraTS19(raw_dir, preprocessed_dir, 'z-score')
