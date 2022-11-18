import torch
from torch.utils.data import DataLoader
import sys
from torch.utils.tensorboard import SummaryWriter
from util.utils_seg import *
from datasets.cow_mask_dataset import CowMaskGenerator
import datasets.semseg.augmentation as psp_trsform
from models.deep_lab_v3 import *
from datetime import date, datetime

from PIL import Image
from glob import glob
import PIL

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


class PseudoLabelTrainer:
    def __init__(self, params):
        self.model_name = params['model_name']
        self.path_to_pascal_dataset = params["path_to_pascal_dataset"]
        self.path_to_augmented_pascal_dataset = params["path_to_augmented_pascal_dataset"]
        self.num_labeled_imgs = params['num_labeled_imgs']
        self.full_img_size = params['full_img_size']
        self.teacher_path = params['teacher_path']
        self.warm_up_unsupervised = params['warm_up_unsupervised']
        self.teacher_ema = params['teacher_ema']
        self.loss_function = params['loss_function']
        self.sup_train_steps_mod = params['sup_train_steps_mod']
        self.dataset_seed = params["dataset_seed"]
        self.seed = params["seed"]
        self.save_model_weights = params['saving_weights']
        self.optimizer_alg = params['optimizer_alg']
        self.total_training_steps = params['total_training_steps']
        self.use_cow_mask_mix_perturbation = params['use_cow_mask_mix_perturbation']
        self.steps_until_eval = params['steps_until_eval']

        if str(self.num_labeled_imgs) not in self.teacher_path:
            raise ValueError
        if str(self.dataset_seed) not in self.teacher_path:
            raise ValueError
        self.num_classes = 21

        self.pl_filter_temperature = 1.0

        np.random.seed(self.dataset_seed)
        if self.num_labeled_imgs is not None:
            with open(self.path_to_pascal_dataset + '/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt') as f:
                lines = f.readlines()
            images = [self.path_to_pascal_dataset + "/VOCdevkit/VOC2012/JPEGImages/" + lines[i][:-1] + ".jpg" for i in range(len(lines))]
            labels = [self.path_to_augmented_pascal_dataset + "/SegmentationClassAug/" + lines[i][:-1] + ".png" for i in range(len(lines))]

            indices_labelled = np.unique(np.random.choice(np.arange(len(images)), self.num_labeled_imgs, replace=False))
            labelled_images = [images[i] for i in indices_labelled]
            labelled_segmentations = [labels[i] for i in indices_labelled]

            # Get unlabelled images
            unlabelled_images = images
            for i in range(len(labelled_images)):
                unlabelled_images.remove(labelled_images[i])

            # append images from augmented dataset
            all_segmentations = glob(os.path.join(self.path_to_augmented_pascal_dataset + "/SegmentationClassAug/", "*.png"))
            basenames = [os.path.basename(all_segmentations[i])[:-4] for i in range(len(all_segmentations))]
            for i in range(len(labelled_images)):
                basenames.remove(os.path.basename(labelled_images[i])[:-4])

            with open(self.path_to_pascal_dataset + '/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt') as f:
                lines = f.readlines()
            val_images = [self.path_to_pascal_dataset + "/VOCdevkit/VOC2012/JPEGImages/" + lines[i][:-1] + ".jpg" for i in range(len(lines))]
            val_segmentations = [self.path_to_augmented_pascal_dataset + "/SegmentationClassAug/" + lines[i][:-1] + ".png" for i in range(len(lines))]

            for i in range(len(val_images)):
                basenames.remove(os.path.basename(val_images[i])[:-4])
            self.unlabelled_images = [self.path_to_pascal_dataset + "/VOCdevkit/VOC2012/JPEGImages/" + basenames[i] + ".jpg" for i in range(len(basenames))]
            self.unlabelled_labels = [self.path_to_augmented_pascal_dataset + "/SegmentationClassAug/" + basenames[i] + ".png" for i in range(len(basenames))]
        else:
            raise NotImplementedError

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

        unlabelled_dataset = voc_dset(images=self.unlabelled_images,
                                      segmentations=self.unlabelled_labels,
                                      trs_form=trs_form)

        trs_form_val = []
        trs_form_val.append(psp_trsform.ToTensor())
        trs_form_val.append(psp_trsform.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
        crop_size, crop_type = [513, 513], "center"
        trs_form_val.append(psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=255))

        trs_form_val = psp_trsform.Compose(trs_form_val)
        val_dataset = voc_dset(images=val_images,
                               segmentations=val_segmentations,
                               trs_form=trs_form_val)

        if sys.gettrace() is None:
            self.num_workers = 8
            self.debug_mode = False
        else:
            self.num_workers = 0
            self.debug_mode = True
        print("NUM WORKER: ", self.num_workers)
        self.labelled_img_loader = DataLoader(dataset=train_dataset,
                                              batch_size=8,
                                              num_workers=self.num_workers,
                                              shuffle=True,
                                              worker_init_fn=seed_worker,
                                              drop_last=True)

        self.unlabelled_img_loader = DataLoader(dataset=unlabelled_dataset,
                                                batch_size=8 * 2,
                                                num_workers=self.num_workers,
                                                shuffle=True,
                                                worker_init_fn=seed_worker,
                                                drop_last=True)

        self.val_batch_size = 10
        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=self.val_batch_size,
                                     num_workers=self.num_workers)

        cow_mask_dataset = CowMaskGenerator(crop_size=(513, 513), method="mix")
        self.cow_mask_loader = DataLoader(dataset=cow_mask_dataset,
                                          batch_size=8,
                                          num_workers=self.num_workers,
                                          worker_init_fn=seed_worker)
        self.cow_mask_iter = iter(self.cow_mask_loader)

        self.n_class = 21
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="none")

        self.model = createDeepLabv3(outputchannels=self.n_class, coco_weights=False)
        self.model_ema = createDeepLabv3(outputchannels=self.n_class, coco_weights=False)

        self.set_up_optimizer()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.model.to(self.device)
        self.model_ema = self.model_ema.to(self.device)

        self.initializes_ema_network()

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
            log_dir="runs/pl/pascal/" + str(self.num_labeled_imgs) + "/" + str(date.year) + "." + str(date.month) + "." + str(date.day) + "-" + str(date.hour) + ":" + str(
                date.minute) + "-" + self.model_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.symmetric_cross_entropy = SCELoss(alpha=2, beta=1, num_classes=self.n_class, ignore_index=255)
        create_model_training_folder(
            writer=self.writer,
            files_to_same=["train_pl_iteration.py", "models/deep_lab_v3.py", "datasets/cow_mask_dataset.py", "util/utils_seg.py"]
        )

        # Load the model from previous iteration
        self.teacher = torch.load(self.teacher_path)
        self.teacher = self.teacher.to(self.device)
        # Validate the used model
        if self.debug_mode == False:
            avg_iou, ious, precision = self.val_pascal(model=self.teacher)
            self.teacher_precision = precision
            print("Teacher Performance: ", avg_iou)
            self.writer.add_scalar('teachers/model_for_pl', avg_iou, global_step=1)


    def save_model(self, pl_iteration, subdir):
        os.makedirs(os.path.join(self.writer.log_dir, "weights", str(pl_iteration), subdir), exist_ok=True)
        torch.save({
            'student_network_state_dict': self.model.state_dict(),
            'teacher_network_state_dict': self.model_ema.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.writer.log_dir, "weights", str(pl_iteration), subdir, self.model_name + ".pth"))

        os.makedirs(os.path.join(self.writer.log_dir, "weights", str(pl_iteration), "whole_model"), exist_ok=True)
        torch.save(self.model_ema, os.path.join(self.writer.log_dir, "weights", str(pl_iteration), "whole_model", self.model_name + ".pth"))

    def save_class_wise_scores(self, iteration, ious):
        os.makedirs(os.path.join(self.writer.log_dir, "class_scores", str(iteration)), exist_ok=True)
        np.save(os.path.join(self.writer.log_dir, "class_scores", str(iteration), "class_scores.npy"), ious)

    def initializes_ema_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.model.parameters(), self.model_ema.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.model.buffers(), self.model_ema.buffers()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def update_ema_network(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model.parameters(), self.model_ema.parameters()):
            param_k.data = param_k.data * self.teacher_ema + param_q.data * (1. - self.teacher_ema)

        for param_q, param_k in zip(self.model.buffers(), self.model_ema.buffers()):
            param_k.data = param_k.data * self.teacher_ema + param_q.data * (1. - self.teacher_ema)

    # @property
    def val_pascal(self, model):
        inters = [0 for i in range(self.n_class)]
        unions = [0 for i in range(self.n_class)]
        true_positives = [0 for i in range(self.n_class)]
        false_positives = [0 for i in range(self.n_class)]
        model.eval()
        for iter, (inputs, labels) in tqdm(enumerate(self.val_loader)):
            inputs, labels = inputs, labels.long()
            inputs = inputs.to(self.device)  # Move your inputs onto the gpu
            labels = labels.to(self.device)  # Move your labels onto the gpu

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

        precision = np.array(true_positives) / (np.array(true_positives) + np.array(false_positives))

        return avg_iou, ious, precision

    def train_step_sup(self, images, labels, weight=None, perform_backward=True, loss_function="CE"):
        # self.optimizer_decoder.zero_grad()
        labels = labels.long()

        outputs = self.model(images)
        outputs = outputs['out']

        if loss_function == "CE":
            loss = self.criterion_ce(outputs, labels)
            if weight is not None:
                loss = loss * weight
        elif loss_function == "SCE":
            loss = self.symmetric_cross_entropy(outputs, labels)
            if weight is not None:
                loss = loss * weight
        else:
            raise NotImplementedError
        loss = torch.mean(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

        if perform_backward:
            self.optimizer.step()
            self.optimizer.zero_grad()
        # self.optimizer_decoder.step()

        return loss, outputs

    def get_labelled_data_sample(self):
        try:
            (image, labels) = next(self.labelled_img_loader_iter)
        except StopIteration:
            self.labelled_img_loader_iter = iter(self.labelled_img_loader)
            (image, labels) = next(self.labelled_img_loader_iter)

        return image, labels

    def get_unlabelled_data_sample(self):
        try:
            (image, labels) = next(self.unlabelled_train_img_loader_iter)
        except StopIteration:
            self.unlabelled_train_img_loader_iter = iter(self.unlabelled_img_loader)
            (image, labels) = next(self.unlabelled_train_img_loader_iter)
        return (image, labels)

    def val(self, pl_iteration):
        with torch.no_grad():
            avg_iou, ious, precision = self.val_pascal(model=self.model_ema)
            if avg_iou > self.best_iou:
                self.best_iou = avg_iou
                self.best_iou_iteration = self.training_steps
                if self.save_model_weights:
                    self.save_model(pl_iteration, subdir="best")
                    self.save_class_wise_scores(pl_iteration, ious)
            torch.cuda.empty_cache()

            print('Iterations: ', self.training_steps, 'EMA Validation Avg IOU:', np.round(avg_iou, 4), 'Current Best: ', np.round(self.best_iou, 4))
            print("EMA IOU values for each class are:", *np.round(np.array(ious) * 100, 2))
            print('--------------------------------------------------------------')

            self.writer.add_scalar('val_iou', avg_iou, global_step=self.training_steps)

        return precision

    def supervised_training_step(self):
        images, labels = self.get_labelled_data_sample()

        # Move inputs onto the gpu
        images = images.to(self.device)
        labels = labels.to(self.device)
        images.required_grad = False
        labels.required_grad = False

        self.model.train()
        loss, outputs = self.train_step_sup(images=images, labels=labels)

        self.writer.add_scalar('loss/labeled', loss.detach().cpu().numpy(), global_step=self.training_steps)

    def unsupervised_training_step(self):
        images, _ = self.get_unlabelled_data_sample()

        # Move inputs onto the gpu
        images = images.to(self.device)
        images.required_grad = False

        with torch.no_grad():
            outputs = self.teacher(images)
            outputs = outputs["out"]
            outputs = outputs / self.pl_filter_temperature
            prob_dist = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pseudo_labels = torch.max(prob_dist, dim=1)

        with torch.no_grad():
            outputs = self.model(images)
            outputs = outputs["out"]
            outputs = outputs / self.pl_filter_temperature
            prob_dist = torch.nn.functional.softmax(outputs, dim=1)

            # Grap confidence of student as weight
            confidence_for_pl, _ = torch.max(prob_dist * torch.nn.functional.one_hot(pseudo_labels, num_classes=self.num_classes).permute(0,3,1,2), dim=1)

        with torch.no_grad():
            # Get cow masks
            cow_mask = next(self.cow_mask_iter)
            cow_mask = cow_mask.to(self.device)
            cow_mask.required_grad = False

            temp_in, ssl_target, ssl_weight = [], [], []
            for i in range(0, 8 * 2, 2):
                temp_in.append(images[i] * cow_mask[int(i / 2)] + (1 - cow_mask[int(i / 2)]) * images[i + 1])
                ssl_target.append(pseudo_labels[i] * torch.squeeze(cow_mask[int(i / 2)], dim=1) + (1 - torch.squeeze(cow_mask[int(i / 2)], dim=1)) * pseudo_labels[i + 1])
                ssl_weight.append(confidence_for_pl[i] * torch.squeeze(cow_mask[int(i / 2)], dim=1) + (1 - torch.squeeze(cow_mask[int(i / 2)], dim=1)) * confidence_for_pl[i + 1])

            images = torch.stack(temp_in)
            ssl_target = torch.squeeze(torch.stack(ssl_target))
            ssl_weight = torch.squeeze(torch.stack(ssl_weight))

        self.model.train()

        loss, outputs = self.train_step_sup(images=images, labels=ssl_target, weight=ssl_weight, perform_backward=True)

        used_pixels = torch.sum(torch.where(ssl_target != self.n_class, torch.ones_like(ssl_target), torch.zeros_like(ssl_target)))
        self.writer.add_scalar('infos/used_pixels', used_pixels.cpu().numpy(), global_step=self.training_steps)
        self.writer.add_scalar('loss/unlabeled', loss.detach().cpu().numpy(), global_step=self.training_steps)

    def set_up_optimizer(self):
        if self.optimizer_alg == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        elif self.optimizer_alg == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=1e-4)
        else:
            raise NotImplementedError

    def train(self):
        self.labelled_img_loader_iter = iter(self.labelled_img_loader)
        self.unlabelled_train_img_loader_iter = iter(self.unlabelled_img_loader)
        thresholds = None
        self.training_steps = 0

        # early_stop = False
        self.best_iou_iteration = 0
        self.best_iou = 0

        while self.training_steps < self.total_training_steps:  # and early_stop == False:
            for it in tqdm(range(self.steps_until_eval)):
                if self.training_steps < self.warm_up_unsupervised:
                    self.unsupervised_training_step()
                else:
                    if self.training_steps % self.sup_train_steps_mod == 0:
                        self.supervised_training_step()
                    else:
                        self.unsupervised_training_step()
                self.update_ema_network()
                self.training_steps += 1
            self.val(1)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.cuda.empty_cache()

    params = {"model_name": "Pascal_Iter=1_Imgs=92_Seed=2408", 
              'path_to_pascal_dataset': "./datasets/pascal",
              'path_to_augmented_pascal_dataset': "./datasets/pascal_aug",
              'num_labeled_imgs': 92,
              'teacher_path': None, # specify path to teacher model here
              'full_img_size': True,
              'warm_up_unsupervised': 0,
              'loss_function': "SCE", #"SCE" or "CE"
              'dataset_seed': 2408, # affects the arrangement of labelled images
              'seed': 666,
              'teacher_ema': 0.998,
              'sup_train_steps_mod': 4,
              "saving_weights": True,
              'optimizer_alg': "SGD",
              'total_training_steps': 140000,
              'steps_until_eval': 1000,
              'use_cow_mask_mix_perturbation': True,
              }

    print(params)

    trainer = PseudoLabelTrainer(params)
    trainer.train()


if __name__ == '__main__':
    main()

