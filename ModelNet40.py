import h5py
import glob
import os
from torch.utils.data import Dataset
from transforms import *


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class ModelNet40(Dataset):
    def __init__(self,
                 root,
                 num_points=1024,
                 partition='train',
                 unseen=False,
                 transforms=None
                 ):

        super(ModelNet40, self).__init__()

        self.root = root
        self.num_points = num_points
        self.partition = partition
        self.unseen = unseen
        self.transforms = transforms

        if self.unseen:
            self.data, self.label = self.load_data(partition, self.root)
            self.label = self.label.squeeze()
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]
            else:
                raise Exception('Invalid partition')
        else:
            self.data, self.label = self.load_data(partition, self.root)
            self.label = self.label.squeeze()

    def __getitem__(self, item):
        sample = {'points': self.data[item, :, :], 'idx': np.array(item, dtype=np.int32), 'label': self.label[item]}
        sample = self.transforms(sample)

        pointcloud1 = sample['points_src']
        pointcloud2 = sample['points_ref']  # n, 3

        euler_ab = sample['euler']
        T_gt = sample['T']

        pointcloud1 = pointcloud1.T  # 3, n
        pointcloud2 = pointcloud2.T

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'),\
                euler_ab.astype('float32'), T_gt.astype('float32')

    def __len__(self):
        return self.data.shape[0]

    def load_data(self, partition, root):
        DATA_DIR = root
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
            f = h5py.File(h5_name)
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label
