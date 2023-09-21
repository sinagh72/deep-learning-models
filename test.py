import torch
import torchvision
from torch.utils.data import random_split, DataLoader
from torchvision.models import VGG16_Weights
import lightning.pytorch as pl

from train import Model, get_transformation, AnimalDatSet

if __name__ == "__main__":
    model_architecture = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model = Model(model=model_architecture, lr=1e-4, wd=1e-4)

    max_epochs = 100
    torch.set_float32_matmul_precision('medium')
    model_path = "./checkpoints"
    batch_size = 32

    cat_dog_dataset = AnimalDatSet(data_root="./dogs-vs-cats (1)/train/train", transformation=get_transformation(), mode="test")
    _, _, test_data = random_split(cat_dog_dataset, [0.8, 0.1, 0.1])

    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
    trainer = pl.Trainer(default_root_dir="./checkpoints",
                         accelerator="gpu",
                         max_epochs=max_epochs,
                         logger=[tb_logger, csv_logger],
                         )