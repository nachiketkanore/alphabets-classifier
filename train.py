import torch
from datasets import get_loaders
from model import get_model
import torch.nn as nn
import torch.optim as optim
import config
import time
import copy
from tqdm import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, dataloaders, criterion, optimizer, scheduler, epochs = config.NUM_EPOCHS):
    since = time.time()
    sanity_check = False

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(epochs)):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-'*10)

        # train and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterating over dataloader
            for x, y in tqdm(dataloaders[phase]):
                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)

                optimizer.zero_grad()

                # tracking history of train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(x)
                    idx, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, y)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * x.shape[0]
                running_corrects += torch.sum(pred == y.data)

                if sanity_check == True:
                    break

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_accuracy = running_corrects.double() / len(dataloaders[phase])

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                    phase, epoch_loss, epoch_accuracy
                ))

            if phase == 'test' and epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
                save_model(model, f"./model_{best_acc}.pth")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60
        ))
    print('Best test accuracy: {:.4f}'.format(best_acc))


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap = 'gray')
    if title is not None:
        plt.title(title)
    plt.pause(4)  # pause a bit so that plots are updated


def visualize_model(model, dataloaders, num_images = 10):
    load_model(model)
    model.eval()
    images_done = 0
    
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_done += 1
                ax = plt.subplot(num_images // 2, 2, images_done)
                ax.axis('off')
                ax.set_title('Predicted: {}'.format('abcdefghijklmnopqrstuvwxyz'[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_done == num_images:
                    return


def save_model(model, file_path = './model.pth'):
    print('Saving model')
    torch.save(model.state_dict(), file_path)

def load_model(model):
    file_name = 'best_model.pth'
    import os
    if file_name in os.listdir('./'):
        print('Loading model')
        model.load_state_dict(torch.load('./best_model.pth'))

def main():
    model = get_model()
    load_model(model)
    train_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = config.STEP_SIZE, gamma = 0.1)
    dataloaders = {
        'train': train_loader,
        'test': test_loader
    }

    visualize_model(model, dataloaders)
    print('DONE')
    return
    
    trained_model = train_model(model, dataloaders, criterion, optimizer, scheduler)
    save_model(model)

    print('OK')


if __name__ == '__main__':
    main()
