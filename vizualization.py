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
from PIL import Image 
import cv2

cudnn.benchmark = True
plt.ion()   # interactive mode

#==================================================================
#
#      PREPARE THE DATASET FOR VIZUALIZATION 
#
#==================================================================
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
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

data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#======================================================================
#
#               VIAZUALIZATION FUNCTION 
#
#======================================================================

def visualize_model(model, num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    save_path = "viz"
    counter = 0 
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                image_tensor = inputs.cpu().data[j]
                #print(image_tensor.shape)
                transform = transforms.ToPILImage()
                pil_image = transform(image_tensor)
                #print(pil_image.size)
                #ax = plt.subplot(num_images//2, 2, images_so_far)
                #ax.axis('off')
                #ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                #print(inputs.cpu().data[j])
                #print("\n")
                #plt.imshow(inputs.cpu().data[j])
                #image = np.array(inputs.cpu().data[j])
                #image = np.transpose(image,(2,1,0))
                ##print(image.shape)
                label = class_names[preds[j]]
                #print(label, counter)
                #cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                name = str(counter) + ".jpg"
                #cv2.imwrite(os.path.join(save_path,name), image)
                pil_image.save(os.path.join(save_path,name))


                # convert to numpy arrayh 
                numpy_image = np.array(pil_image)
                print(numpy_image.shape)



                counter = counter + 1
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

#======================================================================
#
#                   LOAD THE MODEL 
#
#======================================================================

model = torch.load("model/resnet34_trained_GTSRB.pth", map_location ='cpu')
model.eval()
visualize_model(model)



















