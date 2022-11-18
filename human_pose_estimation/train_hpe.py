from torch import optim
from utils.metrics import pck
from models.model import *
from torch.utils.tensorboard import SummaryWriter
#from visualization import *
from datasets.lsp_dataset import *
from datetime import datetime

class HPE_Trainer:
    def __init__(self, params):
        self.model_name = params['model_name']
        self.num_labelled_imgs = params['num_labelled_imgs']
        self.lsp_dataset_joints = params["lsp_dataset_joints"]
        self.lsp_dataset_images = params["lsp_dataset_images"]

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
        self.mynet = HeatmapResNet50Model(14) 
        self.mynet = self.mynet.to(self.device)

        self.criterion = torch.nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.mynet.parameters(), lr=0.0001)
        #optimizer = torch.optim.SGD(mynet.parameters(), lr=2e-3, momentum=0.9, nesterov=True, weight_decay=1e-4)
        self.scaler = torch.cuda.amp.GradScaler()
        #self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 60, 70], gamma=0.1)

        images, labels = load_dataset(self.lsp_dataset_joints, self.lsp_dataset_images)

        val_images = images[1600:]
        val_labels = labels[1600:]
        train_images = images[:1600]
        train_labels = labels[:1600]

        if self.num_labelled_imgs is not None:
            train_images_sub = train_images[:self.num_labelled_imgs]
            train_labels_sub = train_labels[:self.num_labelled_imgs]
            train_images = train_images_sub
            train_labels = train_labels_sub

        # For faster Training
        train_images = train_images + train_images + train_images + train_images + train_images + train_images
        train_labels = train_labels + train_labels + train_labels + train_labels + train_labels + train_labels

        if sys.gettrace() is None:
            num_workers = 16
        else:
            num_workers = 0

        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        batch_size = 16
        image_size = 256
        heatmap_size = 256

        train_dataset = SimpleLabeledImageDataset(images=train_images, input_size=image_size, labels=train_labels, transform=transform_train, output_size=heatmap_size,
                                                  location_transform_config=location_transform, sigma=6)
        val_dataset = SimpleLabeledImageDataset(images=val_images, input_size=image_size, labels=val_labels, transform=transform_val, output_size=heatmap_size, return_heatmap=False)

        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       shuffle=True,
                                       drop_last=True,
                                       worker_init_fn=seed_worker)
        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=False,
                                     drop_last=False,
                                     worker_init_fn=seed_worker)

        date = datetime.now()
        self.writer = SummaryWriter(
            log_dir="runs/pl/" + str(self.num_labelled_imgs) + "/" + str(date.year) + "." + str(date.month) + "." + str(date.day) + "-" + str(date.hour) + ":" + str(
                date.minute) + "-" + self.model_name)
        create_model_training_folder(
            writer=self.writer,
            files_to_same=["train_hpe.py", "utils/utils.py", "datasets/lsp_dataset.py", "datasets/cow_mask_dataset.py", "models/model.py"]
        )

    def evaluation(self):
        annotations = []
        predictions = []

        # again no gradients needed
        with torch.no_grad():
            for data in self.val_loader:
                images, labels, visibility = data
                labels, ratios = labels
                images = images.to(self.device)

                outputs = self.mynet(images)

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

    def train(self):
        best = 0
        #self.evaluation()
        self.training_steps = 0
        for epoch in tqdm(range(100)):  # loop over the dataset multiple times
            self.mynet.train()
            for i, data in tqdm(enumerate(self.train_loader, 0), position=0, leave=True):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, visibility = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                visibility = visibility.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    prediction = self.mynet(inputs)
                    loss = self.criterion(prediction, labels)
                    loss = torch.where(torch.unsqueeze(torch.unsqueeze(visibility, dim=-1), dim=-1) == 0, torch.zeros_like(loss), loss)
                    loss = torch.mean(loss)
                    self.scaler.scale(loss).backward() # loss.backward()
                    self.scaler.step(self.optimizer) #optimizer.step()
                    self.scaler.update()

                self.training_steps += 1

                # ------------------------------------------
            #self.scheduler.step()
            self.mynet.eval()
            pck_all_01, pck_joint_01, pck_all_02 = self.evaluation()
            self.writer.add_scalar('pck_all_01', pck_all_01, global_step=self.training_steps)
            if pck_all_02 > best:
                best = pck_all_02
                os.makedirs(os.path.join(self.writer.log_dir, "weights"), exist_ok=True)
                torch.save(self.mynet, os.path.join(self.writer.log_dir, "weights", self.model_name + ".pth"))

            print("Epoch", epoch, ": PCK: ", np.round(pck_all_01, 4), "Best: ", np.round(best, 4), "PCK 0.2: ", np.round(pck_all_02, 4))
            print("PCK for each joints are:", *np.round(np.array(pck_joint_01), 2))

        print('Finished Training')


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    params = {"model_name": "Baseline",
              'num_labelled_imgs': 100,
              'lsp_dataset_joints': "./datasets/LSP/standard/joints.csv",
              'lsp_dataset_images': "./datasets/LSP/standard/images"}

    print(params)

    trainer = HPE_Trainer(params)
    trainer.train()

if __name__ == '__main__':
    main()











