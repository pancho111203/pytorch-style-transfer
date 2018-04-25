import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from utils.transforms import UnNormalize
from utils.utils import adjust_learning_rate
import numpy as np

# Inputs
original_model = models.vgg19(pretrained=True)
norm_values = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

styleTensorNames = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
contentTensorName = 'conv3_1'

style_image_path = os.path.join(os.path.dirname(__file__), 'data/style_van_gogh.jpg')
content_image_path = os.path.join(os.path.dirname(__file__), 'data/content_bridge.jpg')

imsize = (224, 224)

noise_strength = 0.6

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# Transforms
transform_image = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

detransform_tensor = transforms.Compose([
    transforms.ToPILImage()
])

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = Variable(torch.FloatTensor(mean).view(-1, 1, 1))
        self.std = Variable(torch.FloatTensor(std).view(-1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std

class Model(nn.Module):
    def __init__(self, original_model):
        super(Model, self).__init__()

        self.norm = Normalization(norm_values[0], norm_values[1])
        self.features = nn.Sequential(*list(original_model.features.children())[:-2])

    def forward(self, x):
        norm = self.norm(x)
        return self.features(norm)

model = Model(original_model)

def image_loader(img_path):
    image = Image.open(img_path)
    image = transform_image(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    if torch.cuda.is_available():
        image = image.cuda()
    return image

def image_reconstruct(tensor):
    newT = tensor.data.clone()
    newT = newT.squeeze(0)
    return detransform_tensor(newT)

def generate_noise_image(content_image, noise_strength = noise_strength):
    noise_image = np.random.uniform(-20, 20, (1, 3 ,imsize[0], imsize[1])).astype('float32')
    input_image = noise_image * noise_strength + content_image.data.numpy() * (1 - noise_strength)
    return input_image

style_image = image_loader(style_image_path)
content_image = image_loader(content_image_path)

contentTensor = None
contentTensorTrain = None
styleTensors = [None, None, None, None, None]
styleTensorsTrain = [None, None, None, None, None]
currentRun = None
def feature_capture_hook(module, input, output):
    global contentTensor
    global styleTensors
    global contentTensorTrain
    global styleTensorsTrain

    if currentRun == 'style':
        if module._name == styleTensorNames[0]:
            styleTensors[0] = output.clone()
            styleTensors[0] = styleTensors[0].detach()
            styleTensors[0].requires_grad = False

        elif module._name == styleTensorNames[1]:
            styleTensors[1] = output.clone()
            styleTensors[1] = styleTensors[0].detach()
            styleTensors[1].requires_grad = False

        elif module._name == styleTensorNames[2]:
            styleTensors[2] = output.clone()
            styleTensors[2] = styleTensors[0].detach()
            styleTensors[2].requires_grad = False
            
        elif module._name == styleTensorNames[3]:
            styleTensors[3] = output.clone()
            styleTensors[3] = styleTensors[0].detach()
            styleTensors[3].requires_grad = False
            
        elif module._name == styleTensorNames[4]:
            styleTensors[4] = output.clone()
            styleTensors[4] = styleTensors[0].detach()
            styleTensors[4].requires_grad = False
            

    elif currentRun == 'content':
        if module._name == contentTensorName:
            contentTensor = output.clone()
            contentTensor = contentTensor.detach()
            contentTensor.requires_grad = False

    elif currentRun == 'train':
        if module._name == styleTensorNames[0]:
            styleTensorsTrain[0] = output

        elif module._name == styleTensorNames[1]:
            styleTensorsTrain[1] = output

        elif module._name == styleTensorNames[2]:
            styleTensorsTrain[2] = output
            
        elif module._name == styleTensorNames[3]:
            styleTensorsTrain[3] = output
            
        elif module._name == styleTensorNames[4]:
            styleTensorsTrain[4] = output
        

        if module._name == contentTensorName:
            contentTensorTrain = output

model.features[0]._name = 'conv1_1'
model.features[2]._name = 'conv1_2'
model.features[5]._name = 'conv2_1'
model.features[7]._name = 'conv2_2'
model.features[10]._name = 'conv3_1'
model.features[12]._name = 'conv3_2'
model.features[14]._name = 'conv3_3'
model.features[16]._name = 'conv3_4'
model.features[19]._name = 'conv4_1'
model.features[21]._name = 'conv4_2'
model.features[23]._name = 'conv4_3'
model.features[25]._name = 'conv4_4'
model.features[28]._name = 'conv5_1'
model.features[30]._name = 'conv5_2'
model.features[32]._name = 'conv5_3'
model.features[34]._name = 'conv5_4'

model.features[0].register_forward_hook(feature_capture_hook)
model.features[2].register_forward_hook(feature_capture_hook)
model.features[5].register_forward_hook(feature_capture_hook)
model.features[7].register_forward_hook(feature_capture_hook)
model.features[10].register_forward_hook(feature_capture_hook)
model.features[12].register_forward_hook(feature_capture_hook)
model.features[14].register_forward_hook(feature_capture_hook)
model.features[16].register_forward_hook(feature_capture_hook)
model.features[19].register_forward_hook(feature_capture_hook)
model.features[21].register_forward_hook(feature_capture_hook)
model.features[23].register_forward_hook(feature_capture_hook)
model.features[25].register_forward_hook(feature_capture_hook)
model.features[28].register_forward_hook(feature_capture_hook)
model.features[30].register_forward_hook(feature_capture_hook)
model.features[32].register_forward_hook(feature_capture_hook)
model.features[34].register_forward_hook(feature_capture_hook)

currentRun = 'style'
model(style_image)

currentRun = 'content'
model(content_image)

currentRun = 'train'

criterion = nn.MSELoss()

input_image = Variable(torch.FloatTensor(generate_noise_image(content_image)), requires_grad=True)
if torch.cuda.is_available():
    input_image = input_image.cuda()

# optimizer = optim.SGD([input_image], lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam([input_image], lr = args.lr)
# def train(epoch):
#     model.train()
#     optimizer.zero_grad()

#     model(input_image)

#     #Start with content only
#     loss = criterion(contentTensorTrain, contentTensor)
#     loss.backward()
#     optimizer.step()
#     print('Loss: {}'.format(loss.data[0]))
#     input_image.data.clamp_(-2.5, 2.5)

#for LBFGS optim
optimizer = optim.LBFGS([input_image])
def train(epoch):
    model.train()
    def closure():
        input_image.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_image)
        loss = criterion(contentTensorTrain, contentTensor)
        loss.backward()
        print('Loss: {}'.format(loss.data[0]))
        return loss
    optimizer.step(closure)
    input_image.data.clamp_(0, 1)

import matplotlib.pyplot as plt
def show():
    output_image = image_reconstruct(input_image)
    plt.imshow(output_image)
    plt.show()

def start(startEpoch = 0):
    for epoch in range(startEpoch, startEpoch + 1000):
        # adjust_learning_rate(optimizer, epoch, args.lr, 0.8, 150, 0.2)
        print('epoch {}'.format(epoch))
        train(epoch)

if __name__ == '__main__':
    start(0)
