import os
import pickle

import numpy as np
from core.config import config
from collections import OrderedDict
from torch.utils.data import Dataset


class Dataset3D_itemread(Dataset):
    def __init__(self, data, patch_size):
        super(Dataset3D_itemread, self).__init__()
        self._data = data
        self.patch_size = patch_size
        print(self.patch_size)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        sels = list(self._data.keys())
        name = sels[index]
        data = np.load(self._data[name]['path'])['data']
        age = np.load(self._data[name]['path'])['data1']
        who = np.load(self._data[name]['path'])['data2']
        who = np.array([1, 0, 0]) if who == 1 else (np.array([0, 1, 0]) if who == 2 else np.array([0, 0, 1]))
        who = who.astype(np.float)
        ostime = np.load(self._data[name]['path'])['data4']
        ostime = ostime.astype(np.float32)
        # change month to day
        ostime *= 30

        shape = np.array(data.shape[1:])  # slice的shape，(x, y, z)
        pad_length = self.patch_size - shape
        pad_left = pad_length // 2
        pad_right = pad_length - pad_length // 2
        data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]), (pad_left[2], pad_right[2])))  # 在slice上填充0，使其达到patch_size(192, 224, 192)

        images = data[:4]
        return {'data': images, 'age': age, 'who': who, 'ostime': ostime}


def get_dataset(mode, split):
    # list data path and properties
    with open(os.path.join(config.DATASET.ROOT, split), 'rb') as f:
        splits = pickle.load(f)
    datas = splits['train'] if mode == 'train' else splits['val']
    print(datas.shape)
    print(datas)
    dataset = OrderedDict()
    for name in datas:
        dataset[name] = OrderedDict()
        dataset[name]['path'] = os.path.join(config.DATASET.ROOT, name + '.npz')
    assert len(config.MODEL.INPUT_SIZE) == 3, 'must be 3 dimensional patch size'
    return Dataset3D_itemread(dataset, config.MODEL.INPUT_SIZE)
