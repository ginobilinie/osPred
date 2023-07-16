import os
import json
import pickle
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
from scipy.ndimage import binary_fill_holes
from utils.utils import get_bbox
from core.config import config


def preprocess_dead(raw_dir, preprocessed_dir, normalization):
    seed = 12345

    names = []
    modalities = ['FLAIR', 'T1', 'T1C', 'T2', 'MASK_FLAIR']

    with open(os.path.join(preprocessed_dir, 'dataset.json'), 'r') as f:
        dataset_json = json.load(f)
    modality = dataset_json['modality']
    training = dataset_json['training']
    for case in training:
        images = []
        age = case['age']
        who = case['who']
        death = case['death']
        ostime = case['ostime']
        if death == 2:
            continue
        if os.path.exists(os.path.join(preprocessed_dir, os.path.basename(case['image'][:-7]) + '.npz')):
            name = os.path.basename(case['image'][:-7])
            names.append(name)
            continue
        for i in range(len(modality)):
            image = sitk.ReadImage(os.path.join(raw_dir, case['image'][:-7], modalities[i] + '.nii.gz'))
            image = sitk.GetArrayFromImage(image).astype('float')  # （z, x, y)
            images.append(image)
        image = np.stack(images)
        name = os.path.basename(case['image'][:-7])
        names.append(name)
        label = sitk.ReadImage(os.path.join(raw_dir, case['image'][:-7], modalities[4] + '.nii.gz'))
        label = sitk.GetArrayFromImage(label)

        # crop to non_zero regions
        mask = np.zeros(image.shape[1:], dtype=bool)
        for i in range(len(modality)):
            mask = mask | (image[i] != 0)
        mask = binary_fill_holes(mask)
        l_box = get_bbox(label)
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
            np.savez(os.path.join(preprocessed_dir, name+'.npz'), data=data.astype('float32'), data1=age, data2=who, data3=death, data4=ostime)  # 保存的是，image和label的cat、death、ostime

    np.random.seed(seed)
    np.random.shuffle(names)
    trains = names

    splits = OrderedDict()
    splits['train'] = trains
    with open(os.path.join(preprocessed_dir, 'splits_all.pkl'), 'wb') as f:
        pickle.dump(splits, f)


def preprocess_all(raw_dir, preprocessed_dir, normalization):
    names = []
    modalities = ['FLAIR', 'T1', 'T1C', 'T2', 'MASK_FLAIR']

    with open(os.path.join(preprocessed_dir, 'dataset.json'), 'r') as f:
        dataset_json = json.load(f)
    modality = dataset_json['modality']
    training = dataset_json['training']
    for case in training:
        images = []
        age = case['age']
        who = case['who']
        death = case['death']
        ostime = case['ostime']
        if os.path.exists(os.path.join(preprocessed_dir, os.path.basename(case['image'][:-7]) + '.npz')):
            name = os.path.basename(case['image'][:-7])
            names.append(name)
            continue
        for i in range(len(modality)):
            image = sitk.ReadImage(os.path.join(raw_dir, case['image'][:-7], modalities[i] + '.nii.gz'))
            image = sitk.GetArrayFromImage(image).astype('float')  # （z, x, y)
            images.append(image)
        image = np.stack(images)
        name = os.path.basename(case['image'][:-7])
        names.append(name)

        label = sitk.ReadImage(os.path.join(raw_dir, case['image'][:-7], modalities[4] + '.nii.gz'))
        label = sitk.GetArrayFromImage(label)

        # crop to non_zero regions
        mask = np.zeros(image.shape[1:], dtype=bool)
        for i in range(len(modality)):
            mask = mask | (image[i] != 0)
        mask = binary_fill_holes(mask)
        l_box = get_bbox(label)
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
            np.savez(os.path.join(preprocessed_dir, name+'.npz'), data=data.astype('float32'), data1=age, data2=who, data3=death, data4=ostime)  # 保存的是，image和label的cat、death、ostime


def splits_all(preprocessed_dir):
    seed = 12345

    names1, names2, names3 = [], [], []

    with open(os.path.join(preprocessed_dir, 'dataset.json'), 'r') as f:
        dataset_json = json.load(f)
    training = dataset_json['training']
    for case in training:
        who = case['who']
        name = os.path.basename(case['image'][:-7])
        if who == 1:
            names1.append(name)
        else:
            if who == 2:
                names2.append(name)
            else:
                names3.append(name)

    names1 = np.sort(names1)
    names2 = np.sort(names2)
    names3 = np.sort(names3)

    np.random.seed(seed)
    np.random.shuffle(names1)
    np.random.shuffle(names2)
    np.random.shuffle(names3)

    val_size = 10
    trains, vals = [], []
    vals = np.concatenate((vals, names1[:val_size]))
    trains = np.concatenate((trains, names1[val_size:]))
    vals = np.concatenate((vals, names2[:val_size]))
    trains = np.concatenate((trains, names2[val_size:]))
    vals = np.concatenate((vals, names3[:val_size]))
    trains = np.concatenate((trains, names3[val_size:val_size + 285]))

    np.random.shuffle(trains)
    np.random.shuffle(vals)

    splits = OrderedDict()
    splits['train'] = trains
    splits['val'] = vals
    with open(os.path.join(preprocessed_dir, 'splits_equaldivision.pkl'), 'wb') as f:
        pickle.dump(splits, f)


if __name__ == '__main__':
    raw_dir = config.RAWDATA
    preprocessed_dir = 'DATA/zhengdayi_pre_dead'
    if not os.path.exists(preprocessed_dir):
        os.mkdir(preprocessed_dir)
    preprocess_dead(raw_dir, preprocessed_dir, 'z-score')

    raw_dir = config.RAWDATA
    preprocessed_dir = 'zhengdayi_pre_all'
    if not os.path.exists(preprocessed_dir):
        os.mkdir(preprocessed_dir)
    preprocess_all(raw_dir, preprocessed_dir, 'z-score')
    splits_all('DATA/zhengdayi_pre_all')
