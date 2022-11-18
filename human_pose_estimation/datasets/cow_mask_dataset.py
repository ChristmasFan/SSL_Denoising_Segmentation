import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from utils.utils import generate_cow_mask

class CowMaskGenerator(torch.utils.data.Dataset):
    def __init__(self, crop_size=(256, 256), method='cut', sigmas=None):
        """
        Generates Cow Masks only
        """
        self.crop_size = crop_size
        self.method = method
        if sigmas is None:
            self.sigmas = (13, 15, 17, 19, 21, 23, 25)
        else:
            self.sigmas = sigmas

    def __getitem__(self, idx):
        cow_size = self.crop_size
        sigma = np.random.choice(self.sigmas)
        mask = generate_cow_mask(size=cow_size, sigma=sigma, method=self.method)
        mask = np.expand_dims(mask, axis=0)

        return mask

    def __len__(self):
        return 2000000000
