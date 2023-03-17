import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
from imblearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode



# Image statistics
RGB_statistics = {
    'Places_LT': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std':[0.229, 0.224, 0.225]
    }
}

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            RandomResizedCropAndInterpolation((224, 224)),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 10),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]

# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.targets)
        
    def __sample__(self, index):

        path = self.img_path[index]
        label = self.targets[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    
    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        img, target = self.__sample__(idx)

        return  {'x_lb':  img, 'y_lb': target}


def get_places(args, alg, name, num_classes, data_dir='./data'):
    # print(alg)
    train_txt_path = 'assets/Places_LT/Places_LT_train.txt'
    val_txt_path = 'assets/Places_LT/Places_LT_val.txt'
    test_txt_path = 'assets/Places_LT/Places_LT_test.txt'

    # print(os.listdir('assets/Places_LT/'))

    rgb_mean, rgb_std = RGB_statistics['Places_LT']['mean'], RGB_statistics['Places_LT']['std']

    train_transform = get_data_transform('train', rgb_mean, rgb_std, 'Places_LT')
    val_transform = get_data_transform('val', rgb_mean, rgb_std, 'Places_LT')
    test_transform = get_data_transform('test', rgb_mean, rgb_std, 'Places_LT')

    trainset = LT_Dataset(data_dir, train_txt_path, train_transform)
    valset = LT_Dataset(data_dir, val_txt_path, val_transform)
    testset = LT_Dataset(data_dir, test_txt_path, test_transform)

    print(f'Dataset size: train {len(trainset)}, val {len(valset)}')

    return trainset, valset, testset
    
