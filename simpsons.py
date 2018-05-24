import scipy
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def getPaths(root):
    for sub_dir in os.listdir(root):
        sub_path = os.path.join(root,sub_dir)
        for file in os.listdir(sub_path):
            if 'jpg' in file:
                yield(os.path.join(sub_path,file))
def readimg(l):
    im = cv2.imread(l)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def resize(img,size):
    return cv2.resize(img, (size, size)) 

def edgeExtract(img):
    edges =  cv2.Canny(img,300,300)
    #edges = cv2.dilate(edges,kernel = np.ones((5,5),np.uint8),iterations = 1)
    return edges 

class Simpsons(Dataset):
    """Simpsons dataset."""
    def __init__(self, root_dir, size, transform = None, hole_size=0, n_holes = 0):
        
        def randomHoles(x):
            r = hole_size
            for _ in range(n_holes):
                i = np.random.choice(range(x.shape[0]-r))
                j = np.random.choice(range(x.shape[1]-r))
                x[i:i+r,j:j+r] = 0
            return x
        
        self.paths = list(getPaths(root_dir))
        self.size = size
        self.img_transform = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])
        self.edge_transforms = transforms.Compose([
                   transforms.Lambda(randomHoles),
                   transforms.ToTensor(),
               ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = resize(readimg(self.paths[idx]), self.size)
        return self.img_transform(img), self.edge_transforms(edgeExtract(img)[:,:,None])