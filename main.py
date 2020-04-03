import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from sklearn import svm
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

data_transforms = {
    'train':
        transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}

data_dir = 'hw2_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=4) for x in ['train', 'val' , 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class VGG16_Feature_Extraction(torch.nn.Module):
    def __init__(self):
        super(VGG16_Feature_Extraction, self).__init__()
        VGG16_Pretrained = models.vgg16(pretrained=True)
        self.features = VGG16_Pretrained.features
        self.avgpool = VGG16_Pretrained.avgpool
        self.feature_extractor = nn.Sequential(*[VGG16_Pretrained.classifier[i] for i in range(6)])
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.feature_extractor(x)
        return x


model = VGG16_Feature_Extraction()
model = model.to(device)

image_features = {}
image_labels = {}
for phase in ['train', 'test']:
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        model_prediction = model(inputs)
        model_prediction_numpy = model_prediction.cpu().detach().numpy()
        if (phase not in image_features):
            image_features[phase] = model_prediction_numpy
            image_labels[phase] = labels.numpy()
        else:
            image_features[phase] = np.concatenate((image_features[phase], model_prediction_numpy), axis=0)
            image_labels[phase] = np.concatenate((image_labels[phase], labels.numpy()), axis=0)


scaler = StandardScaler()
image_features['train'] = scaler.fit_transform(image_features['train'])
image_features['test'] = scaler.fit_transform(image_features['test'])
#print(image_features['train'].shape)
clf = LinearSVC()
clf.fit(image_features['train'], image_labels['train'])
#print(clf.coef_)
#print(clf.intercept_)

confidence = clf.score(image_features['test'], image_labels['test'])
print('Accuracy of Pre-trained model:', confidence)
y_test_pred = clf.predict(image_features['test'])
print(confusion_matrix(image_labels['test'],y_test_pred))




## Model 1

model = models.vgg16(pretrained=True)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, len(class_names))

num_epochs = 25
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        all_batchs_loss = 0
        all_batchs_corrects = 0
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            all_batchs_loss += loss.item() * inputs.size(0)
            all_batchs_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step()
        epoch_loss = all_batchs_loss / dataset_sizes[phase]
        epoch_acc = all_batchs_corrects.double() / dataset_sizes[phase]

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts , 'best_model_weight.pth')



model = models.vgg16()
num_ftrs = model.classifier[6].in_features
#model.fc = nn.Linear(num_ftrs, 20)
model.classifier[6] = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)
model.load_state_dict(torch.load('best_model_weight.pth'))

model.eval()
phase = 'test'
confusion_matrix = torch.zeros(len(class_names), len(class_names))
all_batchs_corrects = 0
for inputs, labels in dataloaders[phase]:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    all_batchs_corrects += torch.sum(preds == labels.data)

epoch_acc = all_batchs_corrects.double() / dataset_sizes[phase]

print("Accuracy of First Model:", epoch_acc)
print(confusion_matrix)

## Model 2

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4) for x in ['train', 'val' , 'test']}

model = models.vgg16(pretrained=True)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, len(class_names))

num_epochs = 50
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        all_batchs_loss = 0
        all_batchs_corrects = 0
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            all_batchs_loss += loss.item() * inputs.size(0)
            all_batchs_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step()
        epoch_loss = all_batchs_loss / dataset_sizes[phase]
        epoch_acc = all_batchs_corrects.double() / dataset_sizes[phase]

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts , 'second_model_weight.pth')



model = models.vgg16()
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)
model.load_state_dict(torch.load('second_model_weight.pth'))

model.eval()
phase = 'test'
confusion_matrix = torch.zeros(len(class_names), len(class_names))
all_batchs_corrects = 0
for inputs, labels in dataloaders[phase]:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    all_batchs_corrects += torch.sum(preds == labels.data)

epoch_acc = all_batchs_corrects.double() / dataset_sizes[phase]

print("Accuracy of Second Model:", epoch_acc)
print(confusion_matrix)