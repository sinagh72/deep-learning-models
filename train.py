import glob
import os

import lightning.pytorch as pl
import torch
import torchvision.models
from PIL import Image
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch import optim
from torch.utils.data import Dataset, random_split, DataLoader
from torchmetrics import Accuracy, F1Score, AUROC, Precision
from torchvision import transforms
from torchvision.models import VGG16_Weights
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import functional as F


# Define the image transformations for data augmentation and normalization
def get_transformation():
    return transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomRotation(degrees=(-60, 60)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=3)]), p=0.5),
        transforms.RandomApply(
            torch.nn.ModuleList([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.2)]),
            p=0.5),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    ])


# Dataset class for loading and transforming the animal images
class AnimalDatSet(Dataset):
    def __init__(self, data_root, transformation, mode="train"):
        self.data_root = data_root
        self.transformation = transformation
        self.list_imgs = os.listdir(self.data_root)
        self.mode = mode

    def __len__(self):
        return len(self.list_imgs)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        img_name = self.list_imgs[idx]
        img = self.load_img(img_path=self.data_root + "/" + img_name)
        if self.mode != "test":
            img = self.transformation(img)
        label = torch.zeros(1)  # if it's a dog then label = 0
        if "cat" in img_name:
            label = torch.ones(1)

        return {"img": img, "label": label}

    # Load image from the given path
    def load_img(self, img_path):
        img = Image.open(img_path)
        return img


class Model(pl.LightningModule):
    def __init__(self, model, lr, wd):
        super().__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = torch.nn.Sequential(
            model.classifier,
            torch.nn.Linear(1000, 1),
            torch.nn.Sigmoid()
        )
        self.lr = lr
        self.wd = wd
        # Define metrics for binary classification
        self.initialize_metrics()

    def initialize_metrics(self):
        task = "binary"
        self.metrics_list = ["accuracy", "precision", "f1", "auc"]
        self.sessions = ["train", "val", "test"]
        self.classes = [("cat", 1), ("dog", 0)]

        self.train_ac = Accuracy(task=task, average="weighted")
        self.val_ac = Accuracy(task=task, average="weighted")
        self.test_ac = Accuracy(task=task, average="weighted")

        self.train_p = Precision(task=task, average="weighted")
        self.val_p = Precision(task=task, average="weighted")
        self.test_p = Precision(task=task, average="weighted")

        self.train_f1 = F1Score(task=task, average="weighted")
        self.val_f1 = F1Score(task=task, average="weighted")
        self.test_f1 = F1Score(task=task, average="weighted")

        self.train_auc = AUROC(task=task, average="weighted")
        self.val_auc = AUROC(task=task, average="weighted")
        self.test_auc = AUROC(task=task, average="weighted")

        self.metrics = {"train": [self.train_ac, self.train_p, self.train_f1, self.train_auc],
                        "val": [self.val_ac, self.val_p, self.val_f1, self.val_auc],
                        "test": [self.test_ac, self.test_p, self.test_f1, self.test_auc],
                        }
        self.step_output = {"train": [], "val": [], "test": []}
        self.mean_log_keys = ["loss"]

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        return [optimizer], [scheduler]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Explicitly flatten the tensor while retaining the batch dimension
        x = self.classifier(x)
        return x

    # Calculate the loss for the given batch
    def _calculate_loss(self, batch):
        imgs, labels = batch["img"], batch["label"]
        preds = self.forward(imgs)
        loss = F.binary_cross_entropy(preds, labels)
        # preds = preds.argmax(dim=-1) if len(self.classes) == 2 else preds
        return {"loss": loss, "preds": preds, "labels": labels}

    # Training step
    def training_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.step_output["train"].append(output)
        return output["loss"]

    # Actions to perform at the end of each training epoch
    def on_train_epoch_end(self):
        self.stack_update(session="train")

    # Validation step
    def validation_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.step_output["val"].append(output)
        return output["loss"]

    # Actions to perform at the end of each validation epoch
    def on_validation_epoch_end(self):
        self.stack_update(session="val")

    # Test step
    def test_step(self, batch, batch_idx):
        output = self._calculate_loss(batch)
        self.step_output["test"].append(output)
        return output["loss"]

    # Actions to perform at the end of each test epoch
    def on_test_epoch_end(self, ):
        self.stack_update(session="test")

    # Update metrics with predictions and labels
    def update_metrics(self, session, preds, labels):
        for metric in self.metrics[session]:
            metric.update(preds, labels)

    # Compute and log metrics for the given session
    def stack_update(self, session):
        all_preds = torch.cat([out["preds"] for out in self.step_output[session]])
        all_labels = torch.cat([out["labels"] for out in self.step_output[session]])
        log = {}
        for key in self.mean_log_keys:
            log[f"{session}_{key}"] = torch.stack([out[key] for out in self.step_output[session]]).mean()

        self.update_metrics(session=session, preds=all_preds, labels=all_labels)
        res = self.compute_metrics(session=session)
        self.add_log(session, res, log)
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)
        self.restart_metrics(session=session)

        return all_preds, all_labels

    # Compute and log metrics for the given session
    def compute_metrics(self, session):
        res = {}
        for metric, metric_name in zip(self.metrics[session], self.metrics_list):
            res[metric_name] = metric.compute()
        return res

    # Reset metrics for the given session
    def restart_metrics(self, session):
        for metric in self.metrics[session]:
            metric.reset()
        self.step_output[session].clear()  # free memory

    # Add computed metrics to the log
    def add_log(self, session, res, log):
        for metric in self.metrics_list:
            log[session + f"_{metric}"] = res[metric]


if __name__ == "__main__":
    # Load the VGG16 model with pre-trained weights
    model_architecture = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model = Model(model=model_architecture, lr=1e-4, wd=1e-4)

    max_epochs = 100
    torch.set_float32_matmul_precision('medium')
    model_path = "./checkpoints"
    batch_size = 32
    # Create dataset and split it into train, and validation
    cat_dog_dataset = AnimalDatSet(data_root="./dogs-vs-cats (1)/train/train", transformation=get_transformation())
    train_data, val_data, _ = random_split(cat_dog_dataset, [0.8, 0.1, 0.1])
    # Create data loaders for train, and validation
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(train_data, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
    # Set up logging for training progress
    csv_logger = pl_loggers.CSVLogger(save_dir=os.path.join(model_path, "log/"))
    tb_logger = TensorBoardLogger(save_dir=os.path.join(model_path, "tb_log/"), name="vgg16")
    # Define early stopping criteria
    monitor = "val_loss"
    mode = "min"
    early_stopping = EarlyStopping(monitor=monitor, patience=10, verbose=False, mode=mode)
    # Initialize the trainer and start training
    trainer = pl.Trainer(
        default_root_dir=model_path,
        accelerator="gpu",
        max_epochs=max_epochs,
        callbacks=[
            early_stopping,
            ModelCheckpoint(dirpath=model_path, filename="vgg16-{epoch}-{val_loss:.2f}", save_top_k=1,
                            save_weights_only=True, mode=mode, monitor=monitor),
        ],
        logger=[tb_logger, csv_logger],
    )
    # saved_path = glob.glob(model_path + '/vgg16*.ckpt')
    # if bool(saved_path):
    #     model.load_from_checkpoint(checkpoint_path=saved_path[0])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
