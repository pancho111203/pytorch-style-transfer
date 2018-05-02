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

styleTensorNames = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
contentTensorNames = ['conv_3']

styleRelativeWeights = [0.2, 0.2, 0.5, 0.5, 0.8]
styleWeight = 100000
contentWeight = 1

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

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Model(nn.Module):
    def __init__(self, original_model, content_image, style_image):
        super(Model, self).__init__()

        self.features = nn.Sequential(Normalization(norm_values[0], norm_values[1]))

        self.content_losses = []
        self.style_losses = []

        i = 0  # increment every time we see a conv
        for layer in original_model.features.eval().children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.features.add_module(name, layer)

            if name in contentTensorNames:
                # add content loss:
                target = self.features(content_image).detach()
                content_loss = ContentLoss(target)
                self.features.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in styleTensorNames:
                # add style loss:
                target_feature = self.features(style_image).detach()
                style_loss = StyleLoss(target_feature)
                self.features.add_module("style_loss_{}".format(i), style_loss)
                self.style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(self.features) - 1, -1, -1):
            if isinstance(self.features[i], ContentLoss) or isinstance(self.features[i], StyleLoss):
                break

        self.features = self.features[:(i + 1)]

    def forward(self, x):
        return self.features(x)


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

model = Model(original_model, content_image, style_image)

input_image = Variable(torch.FloatTensor(generate_noise_image(content_image)), requires_grad=True)
if torch.cuda.is_available():
    input_image = input_image.cuda()

#optimizer = optim.SGD([input_image], lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam([input_image], lr = args.lr)
def train(epoch):
    model.train()
    optimizer.zero_grad()

    model(input_image)

    content_loss = 0
    for ct_loss in model.content_losses:
        content_loss += ct_loss.loss

    style_loss = 0
    i = 0
    for st_loss in model.style_losses:
        style_loss += (st_loss.loss * styleRelativeWeights[i])
        i += 1

    content_loss = content_loss * contentWeight
    style_loss = style_loss * styleWeight

    loss = content_loss + style_loss

    loss.backward()
    optimizer.step()
    print('Content Loss: {}, Style Loss: {}, Total: {}'.format(content_loss, style_loss, loss))
    input_image.data.clamp_(0, 1)

# #for LBFGS optim
# optimizer = optim.LBFGS([input_image])
# def train(epoch):
#     model.train()
#     def closure():
#         input_image.data.clamp_(0, 1)
#         optimizer.zero_grad()
#         model(input_image)

#         content_loss = 0
#         for ct_loss in model.content_losses:
#             content_loss += ct_loss.loss
#         loss = content_loss
#         loss.backward()
#         print('Loss: {}'.format(loss.data[0]))
#         return loss
#     optimizer.step(closure)
#     input_image.data.clamp_(0, 1)

import matplotlib.pyplot as plt
def show():
    output_image = image_reconstruct(input_image)
    plt.imshow(output_image)
    plt.show()

epoch = 0
def go():
    global epoch
    while True:
        # adjust_learning_rate(optimizer, epoch, args.lr, 0.8, 150, 0.2)
        print('epoch {}'.format(epoch))
        train(epoch)
        epoch += 1

if __name__ == '__main__':
    go()
