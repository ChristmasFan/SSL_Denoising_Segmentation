from torch import optim
from utils.metrics import pck
from models.model import *
from torch.utils.tensorboard import SummaryWriter
#from visualization import *
from datasets.cow_mask_dataset import CowMaskGenerator
from datasets.lsp_dataset import *
from datetime import datetime

class HPE_Trainer:
    def __init__(self, params):
        self.model_name = params['model_name']
        self.lsp_dataset_joints = params["lsp_dataset_joints"]
        self.lsp_dataset_images = params["lsp_dataset_images"]
        self.num_labelled_imgs = params['num_labelled_imgs']
        self.teacher_path = params["teacher_path"]
        self.steps_until_eval = params["steps_until_eval"]
        self.use_cowmask_perturbation = params["use_cowmask_perturbation"]
        self.use_plw = params["use_plw"]
        self.label_type = params["label_type"]
        self.plw_weight_multiplier = params["plw_weight_multiplier"]
        self.use_pseudo_label_filtering = params["use_pseudo_label_filtering"]
        self.plf_top_n = params["plf_top_n"]

        if str(self.num_labelled_imgs) not in self.teacher_path:
            raise ValueError

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("RUNNING ON {}".format(self.device))

        set_deterministic(seed=2408)

        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                              #transforms.RandomInvert(p=0.2),
                                              #transforms.RandomSolarize(threshold=0.8, p=0.2),
                                              #transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.2),
                                              #transforms.RandomAutocontrast(p=0.2),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        location_transform = {
            "flip_indices": [6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 13, 14],
            "max_rot": 10,
            "max_translate": 0.1,
            "flip": 0.5
}

        self.teacher_ema = 0.999
        self.n_joints = 14

        self.mynet = HeatmapResNet50Model(14) 
        self.mynet = self.mynet.to(self.device)

        self.mynet_ema = HeatmapResNet50Model(14)
        self.mynet_ema = self.mynet_ema.to(self.device)
        self.mynet_ema = self.mynet_ema.eval()

        self.initializes_ema_network()

        self.criterion = torch.nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.mynet.parameters(), lr=0.0001)

        self.teacher = torch.load(self.teacher_path)
        self.teacher = self.teacher.eval()
        #optimizer = torch.optim.SGD(mynet.parameters(), lr=2e-3, momentum=0.9, nesterov=True, weight_decay=1e-4)
        self.scaler = torch.cuda.amp.GradScaler()
        #self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 60, 70], gamma=0.1)

        images, labels = load_dataset(self.lsp_dataset_joints, self.lsp_dataset_images)

        val_images = images[1600:]
        val_labels = labels[1600:]
        train_images = images[:1600]
        train_labels = labels[:1600]

        train_images_labelled = train_images[:self.num_labelled_imgs]
        train_labels_labelled = train_labels[:self.num_labelled_imgs]

        unlabelled_traing_images = train_images[self.num_labelled_imgs:1600]
        unlabelled_traing_labels = train_labels[self.num_labelled_imgs:1600]
        unlabelled_traing_images_analysis = unlabelled_traing_images
        unlabelled_traing_labels_analysis = unlabelled_traing_labels
        # For faster Training
        unlabelled_traing_images = unlabelled_traing_images + unlabelled_traing_images + unlabelled_traing_images + unlabelled_traing_images
        unlabelled_traing_labels = unlabelled_traing_labels + unlabelled_traing_labels + unlabelled_traing_labels + unlabelled_traing_labels

        # For faster Training
        train_images_labelled = train_images_labelled + train_images_labelled + train_images_labelled + train_images_labelled + train_images_labelled + train_images_labelled
        train_labels_labelled = train_labels_labelled + train_labels_labelled + train_labels_labelled + train_labels_labelled + train_labels_labelled + train_labels_labelled

        if sys.gettrace() is None:
            num_workers = 16
        else:
            num_workers = 0

        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        batch_size = 8
        if self.use_cowmask_perturbation:
            self.batch_size_unlabelled = 16
        else:
            self.batch_size_unlabelled = 8
        self.image_size = 256
        heatmap_size = 256
        self.sigma = 6

        train_dataset = SimpleLabeledImageDataset(images=train_images_labelled, input_size=self.image_size, labels=train_labels_labelled, transform=transform_train, output_size=heatmap_size,
                                                  location_transform_config=location_transform, sigma=self.sigma)
        unlabelled_dataset = SimpleLabeledImageDataset(images=unlabelled_traing_images, input_size=self.image_size, labels=unlabelled_traing_labels, transform=transform_train, output_size=heatmap_size,
                                                       location_transform_config=location_transform)
        unlabelled_dataset_analysis = SimpleLabeledImageDataset(images=unlabelled_traing_images_analysis, input_size=self.image_size, labels=unlabelled_traing_labels_analysis, transform=transform_train,
                                                       output_size=heatmap_size,
                                                       location_transform_config=location_transform)
        val_dataset = SimpleLabeledImageDataset(images=val_images, input_size=self.image_size, labels=val_labels, transform=transform_val, output_size=heatmap_size, return_heatmap=False)

        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       shuffle=True,
                                       drop_last=True,
                                       worker_init_fn=seed_worker)
        self.unlabelled_train_loader = DataLoader(dataset=unlabelled_dataset,
                                                  batch_size=self.batch_size_unlabelled,
                                                  num_workers=num_workers,
                                                  shuffle=True,
                                                  drop_last=True,
                                                  worker_init_fn=seed_worker)
        self.unlabelled_train_loader_analysis = DataLoader(dataset=unlabelled_dataset_analysis,
                                                           batch_size=self.batch_size_unlabelled,
                                                           num_workers=num_workers,
                                                           shuffle=False,
                                                           drop_last=False,
                                                           worker_init_fn=seed_worker)
        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=False,
                                     drop_last=False,
                                     worker_init_fn=seed_worker)

        self.train_data_iter = iter(self.train_loader)
        self.unlabelled_data_iter = iter(self.unlabelled_train_loader)

        cow_mask_dataset = CowMaskGenerator(crop_size=(self.image_size, self.image_size), method="mix")
        self.cow_mask_loader = DataLoader(dataset=cow_mask_dataset,
                                          batch_size=int(self.batch_size_unlabelled/2),
                                          num_workers=num_workers,
                                          worker_init_fn=seed_worker)
        self.cow_mask_iter = iter(self.cow_mask_loader)

        date = datetime.now()
        self.writer = SummaryWriter(
            log_dir="runs/pl/" + str(self.num_labelled_imgs) + "/" + str(date.year) + "." + str(date.month) + "." + str(date.day) + "-" + str(date.hour) + ":" + str(
                date.minute) + "-" + self.model_name)
        create_model_training_folder(
            writer=self.writer,
            files_to_same=["train_ssl.py", "utils/utils.py", "datasets/lsp_dataset.py", "datasets/cow_mask_dataset.py", "models/model.py"]
        )

        if self.use_pseudo_label_filtering:
            self.threshold, top_n_values = self.plf_calc_joint_thresholds(np.ones(14)*self.plf_top_n)
            print(np.round(top_n_values,2))

        _, _, pck_teacher = self.evaluation(self.teacher)
        print("Performance Teacher: ", pck_teacher)

    def initializes_ema_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.mynet.parameters(), self.mynet_ema.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.mynet.buffers(), self.mynet_ema.buffers()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def update_ema_network(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.mynet.parameters(), self.mynet_ema.parameters()):
            param_k.data = param_k.data * self.teacher_ema + param_q.data * (1. - self.teacher_ema)

        for param_q, param_k in zip(self.mynet.buffers(), self.mynet_ema.buffers()):
            param_k.data = param_k.data * self.teacher_ema + param_q.data * (1. - self.teacher_ema)

    def plf_calc_joint_thresholds(self, top_n):
        if np.max(top_n) > 1.0:
            raise ValueError

        bins = 1000
        # Calculate bins
        bin_edges = np.histogram_bin_edges(np.append(np.arange(0, 1, 1 / (2 * bins), dtype=np.float32), 1.0), bins=bins)
        class_histogram = np.zeros((self.n_joints, bins), dtype=np.int64)
        with torch.no_grad():
            for data in tqdm(self.unlabelled_train_loader_analysis):
                images, _, _ = data
                images = images.to(self.device)

                pseudo_labels = self.teacher(images)

                if self.label_type == "hard":
                    # Get max value and setup new gaussian headmap
                    heatmaps = pseudo_labels.view(pseudo_labels.shape[0], pseudo_labels.shape[1], -1)
                    max_values, max_indices = torch.max(heatmaps, dim=2)
                    confidences = torch.clip(max_values, 0.0, 1.0)

                    class_histogram += [torch.histc(confidences[:,i], bins=bins, min=0.0, max=1.0).cpu().numpy().astype(np.int64) for i in range(self.n_joints)]

        thresholds = np.zeros(self.n_joints)
        bin_thres_edges = np.zeros(self.n_joints)
        top_n_perc_values = np.zeros(self.n_joints)
        for cur_class in range(self.n_joints):
            best_n_perc = [np.sum(class_histogram[cur_class][i:]) / np.sum(class_histogram[cur_class]) * 100.0 for i in range(0, class_histogram.shape[1])]
            index = np.argmin(np.abs(np.array(best_n_perc) - top_n[cur_class] * 100.0))
            top_n_perc_values[cur_class] = best_n_perc[index]

            thresholds[cur_class] = bin_edges[index]
            bin_thres_edges[cur_class] = index
        # plt.plot([np.sum(class_histogram[1][i:i + 10]) for i in range(0, 1000, 10)]), plt.show()
        return thresholds, top_n_perc_values

    def evaluation(self, model):
        annotations = []
        predictions = []

        # again no gradients needed
        with torch.no_grad():
            for data in self.val_loader:
                images, labels, visibility = data
                labels, ratios = labels
                images = images.to(self.device)

                outputs = model(images)

                heatmap_width = outputs.shape[3]
                heatmaps = outputs.view(outputs.shape[0], outputs.shape[1], -1)
                max_values, max_indices = torch.max(heatmaps, dim=2)

                x = max_indices % heatmap_width
                y = torch.div(max_indices, heatmap_width, rounding_mode='floor')

                max_indices = torch.stack((x, y), dim=2)
                max_indices = (max_indices.permute(1, 2, 0) / ratios.cuda()).permute(2, 0, 1)
                prediction = torch.cat((max_indices, max_values[:, :, None]), dim=2)

                annotations.append(labels.cpu().numpy())
                predictions.append(prediction.cpu().numpy())

        annotations = np.concatenate(np.asarray(annotations), axis=0)
        predictions = np.concatenate(np.asarray(predictions), axis=0)
        pck_all_01, pck_joint_01 = pck(annotations, predictions[:, :, :2], [2, 9], 0.1)
        pck_all_02, pck_joint_02 = pck(annotations, predictions[:, :, :2], [2, 9], 0.2)
        return pck_all_01, pck_joint_01, pck_all_02

    def get_labelled_data(self):
        try:
            labelled_data = next(self.train_data_iter)
        except StopIteration:
            self.train_data_iter = iter(self.train_loader)
            labelled_data = next(self.train_data_iter)

        return labelled_data

    def get_unlabelled_data(self):
        try:
            unlabelled_data = next(self.unlabelled_data_iter)
        except StopIteration:
            self.unlabelled_data_iter = iter(self.unlabelled_train_loader)
            unlabelled_data = next(self.unlabelled_data_iter)

        return unlabelled_data

    def train(self):
        best = 0
        self.training_steps = 0
        for epoch in tqdm(range(150)):
            self.mynet.train()
            for i in tqdm(range(self.steps_until_eval), position=0, leave=True):
                labelled_data = self.get_labelled_data()
                unlabelled_data = self.get_unlabelled_data()

                cow_mask = next(self.cow_mask_iter)
                cow_mask = cow_mask.to(self.device)
                cow_mask.required_grad = False

                # get the inputs; data is a list of [inputs, labels]
                inputs_labelled, labels_labelled, _ = labelled_data
                inputs_unlabelled, _, _ = unlabelled_data

                inputs_labelled = inputs_labelled.to(self.device)
                labels_labelled = labels_labelled.to(self.device)

                inputs_unlabelled = inputs_unlabelled.to(self.device)

                with torch.no_grad():
                    pseudo_labels = self.teacher(inputs_unlabelled)

                    if self.label_type == "hard":
                        # Get max value and setup new gaussian headmap
                        heatmap_width = pseudo_labels.shape[3]
                        heatmaps = pseudo_labels.view(pseudo_labels.shape[0], pseudo_labels.shape[1], -1)
                        max_values, max_indices = torch.max(heatmaps, dim=2)

                        x = max_indices % heatmap_width
                        y = torch.div(max_indices, heatmap_width, rounding_mode='floor')

                        max_indices = torch.stack((x, y), dim=2)
                        #max_indices = (max_indices.permute(1, 2, 0) / ratios.cuda()).permute(2, 0, 1)
                        #prediction = torch.cat((max_indices, max_values[:, :, None]), dim=2)

                        heatmap = np.array([create_heatmaps(max_indices[i].cpu().numpy(), self.image_size, torch.ones(14), sigma=self.sigma) for i in range(self.batch_size_unlabelled)])
                        pseudo_labels = torch.from_numpy(heatmap).float()
                        pseudo_labels = pseudo_labels.to(self.device)

                if self.use_pseudo_label_filtering:
                    weights_plf = max_values >= torch.unsqueeze(torch.from_numpy(self.threshold).to(self.device), dim=0)
                    weights_plf = weights_plf.float()
                    weights_plf = torch.unsqueeze(torch.unsqueeze(weights_plf, dim=-1), dim=-1)
                    weights_plf = weights_plf.repeat(1, 1, pseudo_labels.shape[2], pseudo_labels.shape[3])

                if self.use_plw:
                    with torch.no_grad():
                        self.mynet.eval()
                        prediction_student = self.mynet(inputs_unlabelled)
                        # fuck foor loop...
                        haha = prediction_student.cpu().numpy()
                        hm = max_indices.cpu().numpy()
                        weights_plw = np.zeros((self.batch_size_unlabelled, 14))
                        for i in range(self.batch_size_unlabelled):
                            for j in range(14):
                                weights_plw[i,j] = haha[i, j][hm[i,j,1], hm[i,j,0]]
                        weights_plw = np.where(weights_plw < 0, np.zeros_like(weights_plw), weights_plw)
                        weights_plw = weights_plw * self.plw_weight_multiplier
                        weights_plw = torch.from_numpy(weights_plw).float().to(self.device)
                        weights_plw = torch.unsqueeze(torch.unsqueeze(weights_plw, dim=-1), dim=-1)
                        weights_plw = weights_plw.repeat(1,1,pseudo_labels.shape[2],pseudo_labels.shape[3])
                        self.mynet.train()

                if self.use_cowmask_perturbation:
                    ssl_input, ssl_target, weight_plw_train, weights_plf_train = [], [], [], []
                    for i in range(0, self.batch_size_unlabelled, 2):
                        ssl_input.append(inputs_unlabelled[i] * cow_mask[int(i / 2)] + (1 - cow_mask[int(i / 2)]) * inputs_unlabelled[i + 1])
                        ssl_target.append(pseudo_labels[i] * torch.squeeze(cow_mask[int(i / 2)], dim=1) + (1 - torch.squeeze(cow_mask[int(i / 2)], dim=1)) * pseudo_labels[i + 1])
                        if self.use_plw:
                            weight_plw_train.append(weights_plw[i] * torch.squeeze(cow_mask[int(i / 2)], dim=1) + (1 - torch.squeeze(cow_mask[int(i / 2)], dim=1)) * weights_plw[i + 1])
                        if self.use_pseudo_label_filtering:
                            weights_plf_train.append(weights_plf[i] * torch.squeeze(cow_mask[int(i / 2)], dim=1) + (1 - torch.squeeze(cow_mask[int(i / 2)], dim=1)) * weights_plf[i + 1])

                    inputs_unlabelled = torch.stack(ssl_input)
                    pseudo_labels = torch.squeeze(torch.stack(ssl_target))
                    if self.use_plw:
                        weight_plw_train = torch.squeeze(torch.stack(weight_plw_train))
                    if self.use_pseudo_label_filtering:
                        weights_plf_train = torch.squeeze(torch.stack(weights_plf_train))

                inputs = torch.concat([inputs_labelled, inputs_unlabelled])
                labels = torch.concat([labels_labelled, pseudo_labels])
                if self.use_pseudo_label_filtering:
                    used_joints = torch.sum(weights_plf_train) / (weights_plf_train.shape[0] * weights_plf_train.shape[1] * weights_plf_train.shape[2] * weights_plf_train.shape[3])
                    self.writer.add_scalar('plf', used_joints.item(), global_step=self.training_steps)
                    weights_plf_train = torch.concat([torch.ones_like(labels_labelled), weights_plf_train])
                if self.use_plw:
                    weight_plw_train = torch.concat([torch.ones_like(labels_labelled), weight_plw_train])
                    self.writer.add_scalar('plw_weight_mean', torch.mean(weight_plw_train).item(), global_step=self.training_steps)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    prediction = self.mynet(inputs)
                    loss = self.criterion(prediction, labels)
                    
                    if self.use_plw:
                        loss = loss * weight_plw_train
                    if self.use_pseudo_label_filtering:
                        loss = loss * weights_plf_train
                    loss = torch.mean(loss)
                    self.scaler.scale(loss).backward() # loss.backward()
                    self.scaler.step(self.optimizer) #optimizer.step()
                    self.scaler.update()

                self.training_steps += 1
                self.update_ema_network()
                # ------------------------------------------
            #self.scheduler.step()
            self.mynet.eval()
            pck_all_01, pck_joint_01, pck_all_02 = self.evaluation(self.mynet_ema)
            self.writer.add_scalar('pck_all_02', pck_all_02, global_step=self.training_steps)
            if pck_all_02 > best:
                best = pck_all_02
                os.makedirs(os.path.join(self.writer.log_dir, "weights"), exist_ok=True)
                torch.save(self.mynet_ema, os.path.join(self.writer.log_dir, "weights", self.model_name + ".pth"))
                #torch.save(mynet.state_dict(), best_weights_path)

            print("Epoch", epoch, ": PCK: ", np.round(pck_all_01, 4), "Best: ", np.round(best, 4), "PCK 0.2: ", np.round(pck_all_02, 4))
            print("PCK for each joints are:", *np.round(np.array(pck_joint_01), 2))

        print('Finished Training')


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    params = {"model_name": "test",
              'lsp_dataset_joints': "./datasets/LSP/standard/joints.csv",
              'lsp_dataset_images': "./datasets/LSP/standard/images",
              'num_labelled_imgs': 100,
              'steps_until_eval': 200,
              'teacher_path': None,  # specify path to teacher model here
              'use_cowmask_perturbation': True,
              'label_type': 'hard',     # soft or hard
              'use_plw': True,          # pseudo label weighting
              'plw_weight_multiplier': 2,
              'use_pseudo_label_filtering': False,
              'plf_top_n': 0.8,}    # use top n % of samples for each joint during training

    print(params)

    trainer = HPE_Trainer(params)
    trainer.train()

if __name__ == '__main__':
    main()








