import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


def cointoss(p):
    return random.random() < p


class vimeo_septuplet(Dataset):
    def __init__(self, db_dir, random_crop=None, resize=None, augment_s=True, augment_t=True):
        db_dir += '/sequences'
        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t

        transform_list = []
        if resize is not None:
            transform_list += [transforms.Resize(resize)]

        transform_list += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

        self.folder_list = [(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))]
        self.triplet_list = []
        for folder in self.folder_list:
            self.triplet_list += [(folder + '/' + f) for f in listdir(folder) if isdir(join(folder, f))]

        self.triplet_list = np.array(self.triplet_list)
        self.file_len = len(self.triplet_list)

    def __getitem__(self, index):
        intRandIdx = random.choice([3, 4, 5, 6, 7])
        rawInput = Image.open(self.triplet_list[index] + "/im"+str(intRandIdx)+".png")
        rawRef = Image.open(self.triplet_list[index] + "/im"+str(intRandIdx-2)+".png")

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawInput, output_size=self.random_crop)
            rawInput = TF.crop(rawInput, i, j, h, w)
            rawRef = TF.crop(rawRef, i, j, h, w)

        if self.augment_s:
            if cointoss(0.5):
                rawInput = TF.hflip(rawInput)
                rawRef = TF.hflip(rawRef)
            if cointoss(0.5):
                rawInput = TF.vflip(rawInput)
                rawRef = TF.vflip(rawRef)

        tenInput = self.transform(rawInput)
        tenRef = self.transform(rawRef)

        if self.augment_t:
            if cointoss(0.5):
                return tenInput, tenRef
            else:
                return tenRef, tenInput
        else:
            return tenInput, tenRef

    def __len__(self):
        return self.file_len
