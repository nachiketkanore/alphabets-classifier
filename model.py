import torchvision.models as models
import torch.nn as nn

def get_model():

    model = models.resnet18(pretrained = True, progress = True)
    model.fc = nn.Linear(512, 26)
    # print(model.fc)
    return model

get_model()
