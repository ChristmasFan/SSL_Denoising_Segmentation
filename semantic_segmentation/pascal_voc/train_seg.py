import torch
from torch.utils.data import DataLoader
import sys
from torch.utils.tensorboard import SummaryWriter
from models.deep_lab_v3 import *
from datetime import date, datetime
from glob import glob
import segmentation_models_pytorch as smp
from PIL import Image
import torchvision
import datasets.semseg.augmentation as psp_trsform

import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import Image
from glob import glob

#from data import dataset_creator
from util.utils_seg import *

class voc_dset(torch.utils.data.Dataset):
    def __init__(self, images, segmentations, trs_form):
        self.images = images
        self.segmentations = segmentations
        self.transforms = trs_form

    def __getitem__(self, index):
        # load VOC image and its label
        image_path = self.images[index]
        label_path = self.segmentations[index]

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        image, label = self.transforms(image, label)
        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.images)

class SegTrainer:
    def __init__(self, params):
        self.model_name = params['model_name']
        self.path_to_pascal_dataset = params["path_to_pascal_dataset"]
        self.path_to_augmented_pascal_dataset = params["path_to_augmented_pascal_dataset"]
        self.num_labeled_imgs = params['num_labeled_imgs']
        self.total_training_steps = params['total_training_steps']
        self.dataset_seed = params["dataset_seed"]
        self.seed = params["seed"]
        self.save_model_weights = params['saving_weights']
        self.optimizer_alg = params['optimizer_alg']
        self.steps_until_eval = params['steps_until_eval']

        self.num_classes = 21

        if sys.gettrace() is None:
            self.num_workers = 8
            self.debug_mode = False
        else:
            self.num_workers = 0
            self.debug_mode = True

        np.random.seed(self.dataset_seed)

        if self.num_labeled_imgs is not None:
            with open(self.path_to_pascal_dataset + '/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt') as f:
                lines = f.readlines()
            images = [self.path_to_pascal_dataset + "/VOCdevkit/VOC2012/JPEGImages/" + lines[i][:-1] + ".jpg" for i in range(len(lines))]
            labels = [self.path_to_augmented_pascal_dataset + "/SegmentationClassAug/" + lines[i][:-1] + ".png" for i in range(len(lines))]

            indices_labelled = np.unique(np.random.choice(np.arange(len(images)), self.num_labeled_imgs, replace=False))
            labelled_images = [images[i] for i in indices_labelled]
            labelled_segmentations = [labels[i] for i in indices_labelled]
        else:
            all_segmentations = glob(os.path.join(self.path_to_augmented_pascal_dataset + "/SegmentationClassAug/", "*.png"))

            with open(self.path_to_pascal_dataset + '/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt') as f:
                lines = f.readlines()

            basenames = [os.path.basename(all_segmentations[i])[:-4] for i in range(len(all_segmentations))]
            indices_val = []
            for i in range(len(lines)):
                indices_val.append(basenames.index(lines[i][:-1]))

            train_indices = [ha for ha in range(len(all_segmentations)) if ha not in indices_val]
            train_paths = [all_segmentations[i] for i in train_indices]
            val_paths = [all_segmentations[i] for i in indices_val]
            labelled_segmentations = train_paths
            basenames_train = [os.path.basename(train_paths[i])[:-4] for i in range(len(train_paths))]
            labelled_images = [self.path_to_pascal_dataset + "/VOCdevkit/VOC2012/JPEGImages/" + basenames_train[i] + ".jpg" for i in range(len(basenames_train))]

        # Deterministic
        set_deterministic(seed=self.seed)

        trs_form = []
        trs_form.append(psp_trsform.ToTensor())
        trs_form.append(psp_trsform.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
        trs_form.append(psp_trsform.RandResize(scale=[0.5, 2.0]))
        trs_form.append(psp_trsform.RandomHorizontalFlip())
        crop_size, crop_type = [513, 513], "rand"
        trs_form.append(psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=255))

        trs_form = psp_trsform.Compose(trs_form)
        train_dataset = voc_dset(images=labelled_images,
                                 segmentations=labelled_segmentations,
                                 trs_form=trs_form)


        # #######################

        with open(self.path_to_pascal_dataset + '/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt') as f:
            lines = f.readlines()
        val_images = [self.path_to_pascal_dataset + "/VOCdevkit/VOC2012/JPEGImages/" + lines[i][:-1] + ".jpg" for i in range(len(lines))]
        val_segmentations =[self.path_to_augmented_pascal_dataset + "/SegmentationClassAug/" + lines[i][:-1] + ".png" for i in range(len(lines))]

        trs_form_val = []
        trs_form_val.append(psp_trsform.ToTensor())
        trs_form_val.append(psp_trsform.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
        crop_size, crop_type = [513, 513], "center"
        trs_form_val.append(psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=255))

        trs_form_val = psp_trsform.Compose(trs_form_val)
        val_dataset = voc_dset(images=val_images,
                               segmentations=val_segmentations,
                               trs_form=trs_form_val)

        print("NUM WORKER: ", self.num_workers)
        self.labelled_img_loader = DataLoader(dataset=train_dataset,
                                              batch_size=8,
                                              num_workers=self.num_workers,
                                              shuffle=True,
                                              worker_init_fn=seed_worker,
                                              drop_last=True)

        self.val_batch_size = 10
        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=self.val_batch_size,
                                     num_workers=self.num_workers)

        self.n_class = 21
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="none")

        self.model = createDeepLabv3(outputchannels=self.n_class, coco_weights=False)
        self.model_ema = createDeepLabv3(outputchannels=self.n_class, coco_weights=False)

        self.set_up_optimizer()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.model.to(self.device)
        self.model_ema = self.model_ema.to(self.device)

        self.initializes_ema_network(model=self.model, model_ema=self.model_ema)

        for param in self.model_ema.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            print("GPU is available")

        self.model_ema.eval()
        self.model.train()
        print("SETTING EMA MODELS TO EVAL MODE")

        # Tensorboard
        date = datetime.now()
        self.writer = SummaryWriter(
            log_dir="runs/pl/pascal/seg/" + str(self.num_labeled_imgs) + "/" + str(date.year) + "." + str(date.month) + "." + str(date.day) + "-" + str(date.hour) + ":" + str(date.minute) + "-" + self.model_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        create_model_training_folder(
            writer=self.writer,
            files_to_same=["train_seg.py", "models/deep_lab_v3.py", "datasets/cow_mask_dataset.py", "util/utils_seg.py"]
        )

        os.makedirs(os.path.join(self.writer.log_dir, "indices_labelled"), exist_ok=True)
        if self.num_labeled_imgs is not None:
            np.save(os.path.join(self.writer.log_dir, "indices_labelled", "indices_labelled.npy"), indices_labelled)

    def save_model(self, subdir):
        os.makedirs(os.path.join(self.writer.log_dir, "weights", subdir), exist_ok=True)
        torch.save({
            'student_network_state_dict': self.model.state_dict(),
            'teacher_network_state_dict': self.model_ema.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.writer.log_dir, "weights", subdir, self.model_name + ".pth"))

        if subdir == "best":
            os.makedirs(os.path.join(self.writer.log_dir, "weights", "whole_model"), exist_ok=True)
            torch.save(self.model_ema, os.path.join(self.writer.log_dir, "weights", "whole_model", self.model_name + ".pth"))

    def save_class_wise_scores(self, ious):
        os.makedirs(os.path.join(self.writer.log_dir, "class_scores"), exist_ok=True)
        np.save(os.path.join(self.writer.log_dir, "class_scores", "class_scores.npy"), ious)

    def initializes_ema_network(self, model, model_ema):
        # init momentum network as encoder net
        for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(model.buffers(), model_ema.buffers()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def update_ema_network(self, model, model_ema, teacher_ema):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
            param_k.data = param_k.data * teacher_ema + param_q.data * (1. - teacher_ema)

        for param_q, param_k in zip(model.buffers(), model_ema.buffers()):
            param_k.data = param_k.data * teacher_ema + param_q.data * (1. - teacher_ema)

    #@property
    def val_pascal(self, model):
        inters = [0 for i in range(self.n_class)]
        unions = [0 for i in range(self.n_class)]
        true_positives = [0 for i in range(self.n_class)]
        false_positives = [0 for i in range(self.n_class)]
        model.eval()
        for iter, (inputs, labels) in tqdm(enumerate(self.val_loader)):
            inputs, labels = inputs, labels.long()
            inputs = inputs.to(self.device) 
            labels = labels.to(self.device) 

            outputs = model(inputs)
            outputs = outputs['out']

            outputs_argmax = torch.argmax(outputs, dim=1)

            curr_inter, curr_union, curr_tp, curr_fp = [], [], [], []
            for cls in range(self.n_class):
                TP = torch.sum(((outputs_argmax == cls) & (labels == cls)))
                FP = torch.sum(((outputs_argmax == cls) & (labels != cls) & (labels != 255)))  # remove Background class
                FN = torch.sum(((outputs_argmax != cls) & (labels == cls)))
                intersection = TP
                union = (TP + FP + FN)
                if union == 0:
                    curr_inter.append(0)
                    curr_union.append(0)
                    # if there is no ground truth, do not include in evaluation
                else:
                    curr_inter.append(intersection.cpu().numpy())
                    curr_union.append(union.cpu().numpy())
                    # Append the calculated IoU to the list ious
                curr_tp.append(TP.cpu().numpy())
                curr_fp.append(FP.cpu().numpy())

            inters = [inters[p] + curr_inter[p] for p in range(len(inters))]
            unions = [unions[p] + curr_union[p] for p in range(len(unions))]

            true_positives = [true_positives[p] + curr_tp[p] for p in range(len(true_positives))]
            false_positives = [false_positives[p] + curr_fp[p] for p in range(len(false_positives))]

        ious = [inters[p] / unions[p] if unions[p] != 0 else 0 for p in range(len(inters))]
        avg_iou = np.mean(ious)  # sum(inters)/sum(unions)

        return avg_iou, ious

    def train_step_sup(self, images, labels):
        self.optimizer.zero_grad()
        #self.optimizer_decoder.zero_grad()
        labels = labels.long()

        outputs = self.model(images)
        outputs = outputs['out']

        loss = self.criterion_ce(outputs, labels)
        loss = torch.mean(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        #self.optimizer_decoder.step()

        return loss, outputs

    def get_labelled_data_sample(self):
        try:
            (image, labels) = next(self.labelled_img_loader_iter)
        except StopIteration:
            self.labelled_img_loader_iter = iter(self.labelled_img_loader)
            (image, labels) = next(self.labelled_img_loader_iter)

        return image, labels

    def val(self):
        with torch.no_grad():
            avg_iou, ious = self.val_pascal(model=self.model_ema)
            if avg_iou > self.best_iou:
                self.best_iou = avg_iou
                self.best_iou_iteration = self.training_steps
                if self.save_model_weights:
                    self.save_model(subdir="best")
                    self.save_class_wise_scores(ious)
            torch.cuda.empty_cache()

            print('Iterations: ', self.training_steps, 'EMA Validation Avg IOU:', np.round(avg_iou, 4), 'Current Best: ', np.round(self.best_iou, 4))
            print("EMA IOU values for each class are:", *np.round(np.array(ious) * 100, 2))
            print('--------------------------------------------------------------')

            self.writer.add_scalar('val_iou', avg_iou, global_step=self.training_steps)

    def supervised_training_step(self, training_steps):
        images, labels = self.get_labelled_data_sample()

        if training_steps < 2:
            grid = images * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.array([0.229, 0.224, 0.225])), dim=0), dim=-1), dim=-1)
            grid = grid + torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.array([0.485, 0.456, 0.406])), dim=0), dim=-1), dim=-1)
            grid = torchvision.utils.make_grid(grid)
            self.writer.add_image('imgs', grid, global_step=training_steps)

        # Move inputs onto the gpu
        images = images.to(self.device)
        labels = labels.to(self.device)
        images.required_grad = False
        labels.required_grad = False

        self.model.train()
        loss, outputs = self.train_step_sup(images=images, labels=labels)

        self.writer.add_scalar('loss/labeled', loss.detach().cpu().numpy(), global_step=self.training_steps)

    def set_up_optimizer(self):
        if self.optimizer_alg == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        elif self.optimizer_alg == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=1e-4)
        else:
            raise NotImplementedError

    def train(self):
        self.labelled_img_loader_iter = iter(self.labelled_img_loader)

        self.training_steps = 0

        #early_stop = False
        self.training_steps_reset = 0
        self.best_iou_iteration = 0
        self.best_iou = 0

        # Initital training
        while self.training_steps_reset < self.total_training_steps: # and early_stop == False:
            for it in tqdm(range(self.steps_until_eval)):
                self.supervised_training_step(self.training_steps)
                self.update_ema_network(model=self.model, model_ema=self.model_ema, teacher_ema=0.998)
                self.training_steps += 1
                self.training_steps_reset += 1
            self.val()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.cuda.empty_cache()

    params = {"model_name": "Pascal_Imgs=92_Seed=2408",
              'path_to_pascal_dataset': "./datasets/pascal",
              'path_to_augmented_pascal_dataset': "./datasets/pascal_aug",
              'num_labeled_imgs': 92,
              'total_training_steps': 50000,
              'dataset_seed': 2408, # affects the arrangement of labelled images
              'seed': 1234,
              "saving_weights": True,
              'optimizer_alg': "SGD",
              'steps_until_eval': 500,
              }

    print(params)

    trainer = SegTrainer(params)
    trainer.train()

if __name__ == '__main__':
    main()

