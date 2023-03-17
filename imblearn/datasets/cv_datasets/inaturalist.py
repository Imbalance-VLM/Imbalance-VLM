import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
from imblearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode



# Image statistics
RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
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
            RandAugment(3, 9),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key == 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
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


# Load datasets
def load_data(data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True):

    if phase == 'train_plain':
        txt_split = 'train'
    elif phase == 'train_val':
        txt_split = 'val'
        phase = 'train'
    else:
        txt_split = phase
    txt = './data/%s/%s_%s.txt'%(dataset, dataset, txt_split)
    # txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

    print('Loading data from %s' % (txt))

    if dataset == 'iNaturalist18':
        print('===> Loading iNaturalist18 statistics')
        key = 'iNaturalist18'
    else:
        key = 'default'
    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']

    if phase not in ['train', 'val']:
        transform = get_data_transform('test', rgb_mean, rgb_std, key)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std, key)

    print('Use data transformation:', transform)

    set_ = LT_Dataset(data_root, txt, transform)
    print(len(set_))
    if phase == 'test' and test_open:
        open_txt = './data/%s/%s_open.txt'%(dataset, dataset)
        print('Testing with opensets from %s'%(open_txt))
        open_set_ = LT_Dataset('./data/%s/%s_open'%(dataset, dataset), open_txt, transform)
        set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler: ', sampler_dic['sampler'])
        # print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                           num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)


def get_inaturalist(args, alg, name, num_classes, data_dir='./data'):
    # print(alg)
    train_txt_path = 'assets/iNaturalist18/iNaturalist18_train.txt'
    val_txt_path = 'assets/iNaturalist18/iNaturalist18_val.txt'

    # print(os.listdir('assets/iNaturalist18/'))

    rgb_mean, rgb_std = RGB_statistics['iNaturalist18']['mean'], RGB_statistics['iNaturalist18']['std']

    train_transform = get_data_transform('train', rgb_mean, rgb_std, 'iNaturalist18')
    val_transform = get_data_transform('val', rgb_mean, rgb_std, 'iNaturalist18')

    trainset = LT_Dataset(data_dir, train_txt_path, train_transform)
    valset = LT_Dataset(data_dir, val_txt_path, val_transform)

    return trainset, valset
    
