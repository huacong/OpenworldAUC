import os
from torch.utils.data import Dataset
from utils.util_data import (read_split, 
                             subsample_classes, 
                             generate_fewshot_dataset, 
                             read_image,
                             read_json,
                             imb_split, 
                             write_json,
                             get_lab2cname)

template = ['a photo of a person doing {}.']

class UCF101(Dataset):
    dataset_dir = 'ucf101'

    def __init__(self, root, num_shots, subsample, transform=None, type='train', seed=0, imb_domain = 'base'):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'UCF-101-midframes')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_UCF101.json')

        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        train = generate_fewshot_dataset(train, num_shots=num_shots, seed=seed)

        self.subsample = subsample
        # self.imb_domain = imb_domain
        # self.imb_factor = 0.2
        # test = imb_split(test, imb_domain=self.imb_domain, imb_factor=self.imb_factor)
        train, val, test = subsample_classes(train, val, test, subsample=self.subsample)
        dataset = {'train' : train, 'val' : val, 'test' : test}
        self.data_source = dataset[type]
        self.label2cname, self.cname2label, self.classnames = get_lab2cname(self.data_source)
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        image = read_image(item.impath)
        if self.transform:
            image = self.transform(image)
        
        return image, item.label, item.classname