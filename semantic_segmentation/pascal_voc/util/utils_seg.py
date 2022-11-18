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


def iou(prediction, labels, n_class):
    intersections, unions = [], []
    for cls in range(n_class):
        TP = torch.sum(((prediction == cls) & (labels == cls)))
        FP = torch.sum(((prediction == cls) & (labels != cls) & (labels != n_class)))   # remove Background class
        FN = torch.sum(((prediction != cls) & (labels == cls)))
        intersection = TP
        union = (TP + FP + FN)
        if union == 0:
            intersections.append(0)
            unions.append(0)
            # if there is no ground truth, do not include in evaluation
        else:
            intersections.append(intersection.cpu().numpy())
            unions.append(union.cpu().numpy())
            # Append the calculated IoU to the list ious

    return intersections, unions

def std_augmentation(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2):
    augment = [
        transforms.ColorJitter(brightness=brightness,
                               contrast=contrast,
                               saturation=saturation,
                               hue=hue)
    ]
    tfs = transforms.Compose(augment)
    return tfs

def std_augmentation_jitter_flip(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2):
    color_jitter = transforms.ColorJitter(brightness=brightness,
                                          contrast=contrast,
                                          saturation=saturation,
                                          hue=hue)
    tfs = transforms.Compose([color_jitter,
                              transforms.RandomHorizontalFlip(p=0.5)])
    return tfs


def advanced_augmentation(input_shape, s=1.0):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([color_jitter,
                                          #transforms.RandomGrayscale(p=0.2),
                                          #GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return data_transforms

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

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=19, ignore_index=19):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

        print("WARNING: SCE runs only with IGNORE INDEX = 19")

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes+1).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot[:,:,:,:self.num_classes].permute(0,3,1,2)), dim=1))
        #rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss



def create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'code')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

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

def symmetric_cross_entropy(y_true, y_pred, alpha, beta):
    y_true_1 = y_true.float()
    y_pred_1 = y_pred
    y_true_2 = y_true.float()
    y_pred_2 = y_pred
    y_pred_1 = y_pred_1.clamp(1e-7, 1.0) #torch.clip(y_pred_1, 1e-7, 1.0)
    y_true_2 = torch.clamp(y_true_2, min=1e-4, max=0.9) #torch.clip(y_true_2, 1e-4, 1.0)
    return alpha*torch.mean(-torch.sum(y_true_1.permute(0,3,1,2)[:,:19,:,:] * torch.log(y_pred_1), dim = -1)) + beta*torch.mean(-torch.sum(y_pred_2 * torch.log(y_true_2.permute(0,3,1,2)[:,:19,:,:]), dim = -1))

def main():
    for i in tqdm(range(1000)):
        sigma = np.random.choice((13, 15, 17, 19, 21, 23, 25))
        cow_mask = generate_cow_mask(size=(448, 448), sigma=sigma, method="cut")

if __name__ == '__main__':
    main()

