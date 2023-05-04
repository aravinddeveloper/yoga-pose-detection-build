import argparse
import gzip
import json
import logging
import os
import sys
from torch.optim import lr_scheduler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import copy

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from PIL import Image
import io
import pickle
import torch.utils.model_zoo as model_zoo

CLASS = {}

def save_model(model, model_dir):
    print("Saving the model")
    path = os.path.join(model_dir, "model.pth")
    # model_scripted = torch.jit.script(model.cpu().state_dict()) # Export to TorchScript
    # model_scripted.save(path) # Save
    torch.save(model.cpu().state_dict(), path)
    return

def train_model(model, criterion, optimizer, scheduler, dataset_sizes, dataloaders, num_epochs):
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
    
    # save model checkpoint
    save_model(model, args.model_dir)

    # load best model weights
    model.load_state_dict(best_model_wts)

def prepare_data(args):
    global CLASS
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dir_dict = {"train":args.train,"val":args.val}
    image_datasets = {x: datasets.ImageFolder(dir_dict[x],data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(class_names)
    CLASS = {idx:name for idx,name in enumerate(class_names)}
    print(CLASS)
    return dataset_sizes, dataloaders, len(class_names)

def parse_args():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="S",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 1)"
    )
    # parser.add_argument(
    #     "--learning-rate",
    #     type=float,
    #     default=0.001,
    #     metavar="LR",
    #     help="learning rate (default: 0.01)",
    # )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    
    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val", type=str, default=os.environ["SM_CHANNEL_VAL"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    return parser.parse_args()

def model_fn(model_dir):
    model = VGG(make_layers(cfg=cfg))
    # model.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/vgg16-397923af.pth"))
    model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(512, 512),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(512, 5))
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    # model = torch.load('model.pth', map_location=torch.device('cpu'))
    model = model.eval()
   
    return model

def transform_fn(model, request_body, request_content_type,
                    response_content_type):
    """Run prediction and return the output.
    The function
    1. Pre-processes the input request
    2. Runs prediction
    3. Post-processes the prediction output.
    """
    # preprocess
    decoded = Image.open(io.BytesIO(request_body))
    preprocess = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[
                                        0.485, 0.456, 0.406], std=[
                                        0.229, 0.224, 0.225]),
                                    ])
    normalized = preprocess(decoded)
    batchified = normalized.unsqueeze(0)
    
    # predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batchified = batchified.to(device)
    output = model.forward(batchified)
    prob = F.softmax(output,dim=1)
    val,idx = prob.topk(k=1,dim=1)
    print(prob)
    out_class = {0: 'downdog', 1: 'goddess', 2: 'plank', 3: 'tree', 4: 'warrior2'}
    pred_output = {"prediction" : out_class[idx.cpu().detach().numpy()[0][0]], "confidence_score" : val.cpu().detach().numpy()[0][0]}
    pred_output["confidence_score"] = pred_output["confidence_score"].item()
    print(pred_output, response_content_type )
    return json.dumps(pred_output)   

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    args = parse_args()
    print(args.train,args.val)
    dataset_sizes, dataloaders, classes = prepare_data(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_ft = models.vgg16(pretrained=True)
    # model_ft.classifier = nn.Sequential(nn.Linear(25088, 512),
    #                                     nn.ReLU(),
    #                                     nn.Dropout(),
    #                                     nn.Linear(512, 512),
    #                                     nn.ReLU(),
    #                                     nn.Dropout(),
    #                                     nn.Linear(512, classes))
    # model = VGG(make_layers)
    model = VGG(make_layers(cfg=cfg))
    model.load_state_dict(model_zoo.load_url("https://download.pytorch.org/models/vgg16-397923af.pth"))
    model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(512, 512),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(512, classes))
    for param in model.features.parameters():
        param.requires_grad=False
    feature_param_count, classifier_param_count, total_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters() if not p.requires_grad), sum(p.numel() for p in model.parameters())

    print(f"Trainable paramaters - {classifier_param_count} :: Frozen parameters - {feature_param_count} :: Total parameters - {total_param_count}")
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    train_model(model, criterion, optimizer_ft, exp_lr_scheduler,dataset_sizes, dataloaders,args.epochs)
    # train(args)