from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
import PIL
import os 

cudnn.benchmark = True
plt.ion()   # interactive mode



def process_data_dir(data_dir, processed):
    data = json.load(open(data_dir + 'VQA_RAD_Dataset_Public.json'))
    # load images from the folder in a tensor
    train_images = []
    test_images = []

    test_count = 0
    train_count = 0
    for d in data:
        type = d['phrase_type']
        if type[0:4] == 'test':
            test_count += 1
            test_images.append([d['image_name'], d['image_organ']])
            path = processed+"test/"+d['image_organ']
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            os.system(f"cp {data_dir}/images/{d['image_name']} {path}/{d['image_name']}")
        else:
            train_count += 1
            train_images.append([d['image_name'], d['image_organ']])
            path = processed+"train/"+d['image_organ']
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            os.system(f"cp {data_dir}/images/{d['image_name']} {path}/{d['image_name']}")

    print("train images: ", train_count)
    print("test images: ", test_count)
    return train_images, test_images

def get_mean_std(train_images, test_images):
    train_tensor = np.zeros((len(train_images), 224, 224, 3))
    for i in range(len(train_images)):
        img = PIL.Image.open(data_dir + 'images/' + train_images[i][0])
        img = img.resize((224, 224))
        img = np.array(img)
        train_tensor[i] = img

    test_tensor = np.zeros((len(test_images), 224, 224, 3))
    for i in range(len(test_images)):
        img = PIL.Image.open(data_dir + 'images/' + test_images[i][0])
        img = img.resize((224, 224))
        img = np.array(img)
        test_tensor[i] = img

    train_mean = np.mean(train_tensor, axis=(0,1,2)) / 255
    train_std = np.std(train_tensor, axis=(0,1,2)) / 255

    test_mean = train_tensor.mean(axis=(0,1,2)) / 255
    test_std = train_tensor.std(axis=(0,1,2)) / 255

    print("mean and std of train images: ", train_mean, train_std)
    print("mean and std of test images: ", test_mean, test_std)
    return train_mean, train_std, test_mean, test_std


def data_loader(train_mean, train_std, test_mean, test_std):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(test_mean, test_std)
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(processed, x), data_transforms[x]) for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



if __name__ == "__main__":

    data_dir = '../vqa-rad/'
    processed = '../vqa-rad-processed/'
    train_images, test_images = process_data_dir(data_dir, processed)
    train_mean, train_std, test_mean, test_std = get_mean_std(train_images, test_images)
    dataloaders, dataset_sizes, class_names = data_loader(train_mean, train_std, test_mean, test_std)
    

    model_ft = models.resnet152(weights=None)
    num_ftrs = model_ft.fc.in_features


    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=25)