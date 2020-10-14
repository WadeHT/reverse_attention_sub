from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
import torch
import random
import re

"""define dataloader"""


class real_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.x = [str(filename) for filename in self.root_dir.glob('*')]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img = cv2.imread(self.x[index], 0)
        img = cv2.equalizeHist(img)
        if self.transform:
            img = self.transform(img)
        return img


class sim_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.golden = sorted([str(filename) for filename in self.root_dir.glob('*') if str(filename).find("_g.png") >= 0])
        self.defect = sorted([str(filename) for filename in self.root_dir.glob('*') if str(filename).find("_d.png") >= 0])
        self.target = sorted([str(filename) for filename in self.root_dir.glob('*') if str(filename).find("_t.png") >= 0])

    def __len__(self):
        return len(self.golden)

    def __getitem__(self, index, rand_same=0.5):
        golden = cv2.imread(self.golden[index], 0)
        defect = cv2.imread(self.defect[index], 0)
        target = cv2.imread(self.target[index], 0)

        # equalizeHist
        golden = cv2.equalizeHist(golden)
        defect = cv2.equalizeHist(defect)

        # random defect brightness
        defect = defect + np.random.normal(0, 5) * target

        # random brightness
        golden_1 = random.uniform(0.25, 1.75) * golden + random.randint(-25, 25)
        golden_2 = random.uniform(0.25, 1.75) * golden + random.randint(-25, 25)
        defect = random.uniform(0.25, 1.75) * defect + random.randint(-25, 25)

        # make img in 0 ~ 255
        golden_1[golden_1 > 255], golden_1[golden_1 < 0] = 255, 0
        golden_2[golden_2 > 255], golden_2[golden_2 < 0] = 255, 0
        defect[defect > 255], defect[defect < 0] = 255, 0

        if self.transform:
            golden_1 = self.transform(golden_1.astype(np.uint8))
            golden_2 = self.transform(golden_2.astype(np.uint8))
            defect = self.transform(defect.astype(np.uint8))
            target = self.transform(target.astype(np.uint8))

        if random.uniform(0, 1) <= rand_same:
            return golden_1, defect, target
        else:
            return golden_1, golden_2, torch.zeros_like(target)


class test_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.golden = sorted([str(filename) for filename in self.root_dir.glob('*') if str(filename).find("_g.png") >= 0])
        self.defect = sorted([str(filename) for filename in self.root_dir.glob('*') if str(filename).find("_d.png") >= 0])

    def __len__(self):
        return len(self.golden)

    def __getitem__(self, index):
        golden = cv2.imread(self.golden[index], 0)
        defect = cv2.imread(self.defect[index], 0)

        golden = cv2.equalizeHist(golden)
        defect = cv2.equalizeHist(defect)

        if self.transform:
            golden = self.transform(golden)
            defect = self.transform(defect)
        return golden, defect


"""HERE IS TESTING"""

# # Commented out IPython magic to ensure Python compatibility.
# from google.colab import drive
# drive.mount('/content/drive')
# # %cd /content/drive/My Drive/研究所/lab/wnc/cycle_GAN
# !ls

# from torchvision import transforms
# from google.colab.patches import cv2_imshow
# import numpy as np

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# real_dataset = patch_real_dataset(root_dir="./patch_dataset_20200724", transform=transform)
# real_dataset[0].size()

if __name__ == "__main__":
    dataset = real_dataset(root_dir="../dataset/real")
    img = dataset[0]
    print(len(dataset), img.shape)

    dataset = sim_dataset(root_dir="../dataset/sim")
    golden, defect, target = dataset[0]
    print(len(dataset), golden.shape, defect.shape, target.shape, np.max(target))
    cv2.imshow('My Image', golden)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('My Image', defect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('My Image', target*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
