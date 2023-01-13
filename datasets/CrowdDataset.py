from torch.utils.data import Dataset
import os
from torchvision import transforms
from torchvision.datasets.folder import make_dataset, default_loader
import numpy as np
import re
from PIL import Image, ImageOps
import pandas as pd

def _pluck(root):
    img_path = root + '/img'
    ret = [filename for filename in os.listdir(img_path) \
                       if os.path.isfile(os.path.join(img_path, filename))]
    return ret

class CrowdCluster(Dataset):
    def __init__(self, root_dir, num_domains):
        self.root_dir = root_dir
        self.loader = default_loader
        self.num_domains = num_domains

        self.load_dataset()

        self.img_path = self.root_dir + '/img'
        self.gt_path = self.root_dir + '/den'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.data_files.sort(key = lambda i:int(re.match(r'(\d+)',i).group()))
        self.num_samples = len(self.data_files)
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):


        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)
        img = self.transform(img)
        return fname, img

    def __len__(self):
        return len(self.images)

    def load_dataset(self):
        self.domains = np.zeros(0)

        self.images = _pluck(self.root_dir)  # the directories of all training images
        self.num_train = len(self.images)
        print(self.__class__.__name__, "dataset loaded")
        print("  OrdData   | # ids | # images")
        print("  ---------------------------")
        print("  train     | {:5d}"
              .format(self.num_train))

        self.clusters = np.zeros(self.num_train, dtype=np.int64)



    def set_cluster(self, cluster_list):
        if len(cluster_list) != len(self.images):
            raise ValueError("The length of cluster_list must to be same as self.images")
        else:
            self.subdomains = []
            self.clusters = cluster_list
            for i in range(self.num_domains):
                idx = np.where(cluster_list == i)

                sub_image = list(np.array(self.images)[idx])
                sub_dataset = Sub_Dataset(self.root_dir, sub_image, i)
                self.subdomains.append(sub_dataset)

            for i in range(self.num_domains):
                if i == 0:
                    print("  OrdData   | # images  ", end="")

                elif i == self.num_domains-1:
                    print("| # images  ")
                else:
                    print("| # images  ", end="")


            for i in range(self.num_domains):
                if i == 0:
                    print("  ----------------------", end="")
                elif i == self.num_domains-1:
                    print("------------")
                else:
                    print("------------", end="")

            for i in range(self.num_domains):
                if i == 0:
                    print("  train     | {:5d}     ".format(self.subdomains[i].__len__()), end="")
                elif i == self.num_domains-1:
                    print("| {:5d}     ".format(self.subdomains[i].__len__()))
                else:
                    print("| {:5d}     ".format(self.subdomains[i].__len__()), end="")
    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',', header=None).values

        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)
        return img, den

class Sub_Dataset(Dataset):
    def __init__(self, root, images, label):
        super(Sub_Dataset, self).__init__()
        self.root = root
        self.train = images
        self.label = label

    def __len__(self):
        return len(self.train)