from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import config
import numpy as np
import matplotlib.pyplot as plt
from model import get_model

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap = 'gray')
    print('Checking image pixel values: ', np.min(inp), np.max(inp))
    if title is not None:
        plt.title(title)
    plt.pause(3)  # pause a bit so that plots are updated

def get_loaders():
    AlphabetDataset = ImageFolder('./create_dataset/dataset1/',
                    transform = config.basic_transform
            )

    train_size = int(0.8 * len(AlphabetDataset))
    test_size = len(AlphabetDataset) - train_size

    train_dataset, test_dataset = random_split(AlphabetDataset, [train_size, test_size])
    
    train_loader = DataLoader(
                train_dataset,
                shuffle = True,
                batch_size = config.BATCH_SIZE,
                num_workers = config.NUM_WORKERS,
                pin_memory = True
            )

    test_loader = DataLoader(
                test_dataset,
                shuffle = False,
                batch_size = config.BATCH_SIZE,
                num_workers = config.NUM_WORKERS,
            )
    classes = AlphabetDataset.classes

    model = get_model()

    print('returning dataloaders...')
    return train_loader, test_loader

    for x, y in train_loader:
        print(x.shape, y.shape)
        # imshow(x[0])
        print('Label = ', classes[y[0]])
        out = model(x)
        print(out.shape)
        break
    
    for x, y in test_loader:
        print(x.shape, y.shape)
        # imshow(x[0])
        print('Label = ', classes[y[0]])
        out = model(x)
        print(out.shape)
        break


if __name__ == '__main__':
    get_loaders()
