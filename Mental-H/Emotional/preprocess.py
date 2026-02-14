import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import DATA_DIR, BATCH_SIZE, IMG_SIZE
from torchvision import datasets
from torch.utils.data import DataLoader




# Transformations including normalization
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def get_dataloader(data_dir, batch_size=BATCH_SIZE, shuffle=True):

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_dataloader(
    data_dir,
    transform,
    batch_size=64,
    shuffle=True,
    return_labels=False
):
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

    if return_labels:
        labels = [label for _, label in dataset.samples]
        return loader, labels

    return loader

if __name__ == "__main__":
    train_loader = get_dataloader(os.path.join(DATA_DIR, "train"))
    for imgs, labels in train_loader:
        print("Batch imgs:", imgs.shape, "Batch labels:", labels.shape)
        break
