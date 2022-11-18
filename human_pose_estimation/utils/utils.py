import os
import random
import numpy as np
import torch
from torchvision import transforms
from shutil import copyfile
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

def set_deterministic(seed=2408):
    # settings based on https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True  # Training damit halb so schnell
    #torch.set_deterministic(True)  # nicht möglich
    #torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

def seed_worker(worker_id):
    # taken from https://pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def denormalize_image(img):
    if len(img.shape) != 3:
        # TODO: Vectorize
        img = img.permute(0, 2, 3, 1)
        for i in range(len(img)):
            img[i] = img[i] * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        #raise NotImplementedError
    else:
        img = img.permute(1,2,0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return img

def create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'code')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

def generate_cow_mask(size, sigma, method):
    cow_mask = np.random.uniform(low=0.0, high=1.0, size=size)
    cow_mask_gauss = gaussian_filter(cow_mask, sigma=sigma)

    mean = np.mean(cow_mask_gauss)
    std = np.std(cow_mask_gauss)
    #thresh = mean + perturbation*std
    if method == "mix":
        cow_mask_final = (cow_mask_gauss < mean).astype(np.int32)
    elif method == "cut":
        offset = np.random.uniform(low=0.5, high=1.0, size=())
        cow_mask_final = (cow_mask_gauss < mean+offset*std).astype(np.int32)
    else:
        raise NotImplementedError

    return cow_mask_final