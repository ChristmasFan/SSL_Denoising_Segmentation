import sys

from PIL import Image
from utils.utils import *
import scipy.io
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
#from visualization import *
from utils.targets import create_heatmaps


def load_dataset(path, image_base_path):
    if path[-3:] == "csv":
        annotations = []
        images = []
        with open(path, "r") as csv:
            assertion_num = csv.readline()
            assertion_num = int(assertion_num[1:])
            kp_line = csv.readline()
            cnt = 0
            for line in csv:
                cnt += 1
                splits = line.split(sep=",")
                assert len(splits) == 14 * 2 + 1
                image_name = "im{}.jpg".format(splits[0])
                image_path = os.path.join(image_base_path, image_name)
                images.append(image_path)

                keypoints = np.zeros((14, 3))
                for i in range(14):
                    keypoints[i][0:2] = [float(splits[1 + i * 2]), float(splits[1 + i * 2 + 1])]
                    keypoints[i][2] = 1
                annotations.append(keypoints)
            assert cnt == assertion_num
    elif path[-3:] == "mat":
        mat = scipy.io.loadmat(path)
        keypoints = mat["joints"]
        annotations = np.transpose(keypoints, (2, 0, 1))

        images = sorted(glob.glob(os.path.join(image_base_path, "*.jpg")))

    else:
        return NotImplementedError
    return images, annotations


class SimpleLabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform, input_size=128, output_size=64, return_heatmap=True, location_transform_config=None, sigma=2):
        """
        :param images: A List of image filenames
        :param labels: A List of corresponding labels
        :param transform: A set of PyTorch transformations applied to image using Compose.
        yields Image + Label.
        """
        self.images = images
        self.labels = labels
        self.transforms = transform
        self.num_examples = len(self.images)
        self.input_size = input_size
        self.output_size = output_size
        self.return_heatmap = return_heatmap
        self.location_transforms = location_transform_config
        self.sigma = sigma

    def __getitem__(self, idx):
        # Load the image with PIL
        img_name = self.images[idx]
        img = Image.open(img_name).convert('RGB')
        h, w = img.size
        size = h

        label = np.asarray(self.labels[idx][:,0:2])
        visibility = self.labels[idx][:,2]

        if h > w:
            new_im = Image.new(img.mode, (h, h), (0, 0, 0))
            new_im.paste(img, (0, 0))
            img = new_im
        elif w > h:
            new_im = Image.new(img.mode, (w, w), (0, 0, 0))
            new_im.paste(img, (0, 0))
            img = new_im
            size = w

        if self.location_transforms is not None:
            rot, trans_x, trans_y, flip = np.random.rand(4)
            rot = -self.location_transforms["max_rot"] + 2 * self.location_transforms["max_rot"] * rot
            trans_x = (-self.location_transforms["max_translate"] + 2 * self.location_transforms["max_translate"] * trans_x) * w
            trans_y = (-self.location_transforms["max_translate"] + 2 * self.location_transforms["max_translate"] * trans_y) * h
            flip = self.location_transforms["flip"] * flip

            img = transforms.functional.affine(img, angle=rot, translate=[trans_x, trans_y], scale=1, shear=0)
            label = self.rotate_coords(rot, label, (size, size))
            label[:, 0] += trans_x
            label[:, 1] += trans_y
            if flip > 0.5:
                img = transforms.functional.hflip(img)
                label = self.flip_coords(label, size, self.location_transforms["flip_indices"])

        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        img = self.transforms(img)

        ratio = self.output_size / max(h, w)

        if self.return_heatmap:
            label *= ratio

            heatmap = create_heatmaps(label, self.output_size, visibility, self.sigma)
            heatmap = torch.from_numpy(heatmap)

            return img.float(), heatmap.float(), torch.from_numpy(visibility).float()
        else:
            return img.float(), (label, ratio), torch.from_numpy(visibility).float()

    def __len__(self):
        return self.num_examples

    @staticmethod
    def rotate_coords(angle, coords, size):
        center = (size[0] // 2, size[1] // 2)
        coords = np.copy(coords)
        a = np.radians(angle)
        cosa = np.cos(a)
        sina = np.sin(a)

        coords[:, 0] -= center[0]
        coords[:, 1] -= center[1]
        orig_coords = np.copy(coords)
        coords[:, 0] = orig_coords[:, 0] * cosa - orig_coords[:, 1] * sina
        coords[:, 1] = orig_coords[:, 0] * sina + orig_coords[:, 1] * cosa
        coords[:, 0] += center[0]
        coords[:, 1] += center[1]
        return coords

    @staticmethod
    def scale_coords(scale, coords, size):
        center = (size[0] // 2, size[1] // 2)
        coords[:, 0] -= center[0]
        coords[:, 1] -= center[1]
        coords[:, :2] *= scale
        coords[:, 0] += center[0]
        coords[:, 1] += center[1]
        return coords

    @staticmethod
    def flip_coords(coords, size, flip_config):
        coords[:, 0] = size - coords[:, 0]
        coords = coords[:, flip_config]
        return coords

