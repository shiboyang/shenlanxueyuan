import os
import cv2
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataset import T_co


class Rotate:
    def __call__(self, img):
        pass

    def __repr__(self):
        pass



class MNISTDataSet(data.Dataset):
    training_file = 'train.txt'
    test_file = 'test.txt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = self._load_data(os.path.join(self.root, data_file))

    @classmethod
    def _load_data(cls, filepath):
        img_path = []
        labels = []
        with open(filepath, 'r') as f:
            text = f.readline()
            while text:
                _img_path, _label = text.strip().split(' ')
                img_path.append(_img_path.strip())
                labels.append(_label.strip())
                text = f.readline()

        return img_path, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        img_path, target = self.data[index], int(self.targets[index])
        img = cv2.imread(img_path)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def download(self):
        raise NotImplemented

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.test_file))
