import torch
from torchvision import transforms

LR = 0.001
MOMENTUM = 0.9
STEP_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 4
BATCH_SIZE = 16
PIN_MEMORY = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

basic_transform = transforms.Compose([
        transforms.ToTensor()
    ])
