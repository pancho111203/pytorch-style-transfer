import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from pytorch_utils.transforms import UnNormalize
from pytorch_utils.general import adjust_learning_rate, set_learning_rate, get_learning_rate, LearningRateAdapter
from pytorch_utils.helpers.Metrics import Metrics
import numpy as np
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser(description='Neural style transfer')
argparser.add_argument('--style-image', '-si', type=str, default='data/style_vangogh.jpg')
argparser.add_argument('--content-image', '-ci', type=str, default='data/content_bridge.jpg')
argparser.add_argument('--style-tensors', '-st', type=int, nargs='*', default=[2, 4, 8, 12, 16])
argparser.add_argument('--content-tensors', '-ct', type=int, nargs='*', default=[9])
argparser.add_argument('--style-weight', '-sw', type=float, default=100000.0)
argparser.add_argument('--content-weight', '-cw', type=float, default=1)
argparser.add_argument('--noise-strength', '-n', type=float, default=0.6)
argparser.add_argument('--image-size', '-s', type=int, default=224)
argparser.add_argument('--learning-rate', '-lr', type=float, default=100)
argparser.add_argument('--adaptative-lr', '-adlr', default=False, action='store_true')
argparser.add_argument('--style-relative-weights', '-srw', type=float, nargs='*', default=[0.5, 0.5, 1.5, 3.0, 4.0])
argparser.add_argument('--output-name', '-o', type=str, default=None)
argparser.add_argument('--output-info', '-i', type=str, default=None)
argparser.add_argument('--average-pool', '-avgp', default=False, action='store_true')
args = argparser.parse_args()

# Inputs
original_model = models.vgg19(pretrained=True)
norm_values = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

styleTensorNames = ['conv_{}'.format(tensorIndex) for tensorIndex in args.style_tensors]
contentTensorNames = ['conv_{}'.format(tensorIndex) for tensorIndex in args.content_tensors]

styleWeight = args.style_weight
contentWeight = args.content_weight

styleRelativeWeights = args.style_relative_weights

imsize = (args.image_size, args.image_size)

use_only_noise = True if args.noise_strength >= 1.0 else False
noise_strength = args.noise_strength

style_image_name = args.style_image
content_image_name = args.content_image

def get_output_name():
    info = '' if args.output_info is None else '_{}'.format(args.output_info)
    avgpool = '' if args.average_pool is False else '_avgpool'
    onlynoise = '' if use_only_noise is False else '_onlynoise'
    adapt_lr = '' if args.adaptative_lr is False else '_adaptlr'
    style_img_sn = style_image_name.split('_')[1].split('.')[0]
    content_img_sn = content_image_name.split('_')[1].split('.')[0]
    style_config_str = ''
    for i in range(0, len(styleTensorNames)):
        tensorName = styleTensorNames[i]
        relativeWeigth = styleRelativeWeights[i]
        tensorIndex = tensorName.split('_')[1]

        tensor_config_str = tensorIndex + ':' + str(relativeWeigth)
        style_config_str += tensor_config_str
        if i < len(styleTensorNames) - 1:
            style_config_str += ','

    content_config_str = ''
    for i in range(0, len(contentTensorNames)):
        content_config_str += contentTensorNames[i].split('_')[1]
        if i < len(contentTensorNames) - 1:
            content_config_str += ','

    return str(args.image_size) + 'x' + str(args.image_size) + '_' + style_img_sn + 'x' + str(styleWeight) + '@' + style_config_str + '_' + content_img_sn + 'x' + str(contentWeight) + '@' + content_config_str + avgpool + onlynoise + adapt_lr + info

print(args.adaptative_lr)
filename = args.output_name if args.output_name is not None else get_output_name()
print('Filename: {}'.format(filename))
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'logs')):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'))
metrics = Metrics(os.path.join(os.path.dirname(__file__), 'logs/{}'.format(filename)), save_every=50)

"""
    0 is conv1 (3, 3, 3, 64)
    1 is relu
    2 is conv2 (3, 3, 64, 64)
    3 is relu    
    4 is maxpool
    5 is conv3 (3, 3, 64, 128)
    6 is relu
    7 is conv4 (3, 3, 128, 128)
    8 is relu
    9 is maxpool
    10 is conv5 (3, 3, 128, 256)
    11 is relu
    12 is conv6 (3, 3, 256, 256)
    13 is relu
    14 is conv7 (3, 3, 256, 256)
    15 is relu
    16 is conv8 (3, 3, 256, 256)
    17 is relu
    18 is maxpool
    19 is conv9 (3, 3, 256, 512)
    20 is relu
    21 is conv10 (3, 3, 512, 512)
    22 is relu
    23 is conv11 (3, 3, 512, 512)
    24 is relu
    25 is conv12 (3, 3, 512, 512)
    26 is relu
    27 is maxpool
    28 is conv13 (3, 3, 512, 512)
    29 is relu
    30 is conv14 (3, 3, 512, 512)
    31 is relu
    32 is conv15 (3, 3, 512, 512)
    33 is relu
    34 is conv16 (3, 3, 512, 512)
    35 is relu
    36 is maxpool
    37 is fullyconnected (7, 7, 512, 4096)
    38 is relu
    39 is fullyconnected (1, 1, 4096, 4096)
    40 is relu
    41 is fullyconnected (1, 1, 4096, 1000)
    42 is softmax
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


original_model = original_model.to(device)

style_image_path = os.path.join(os.path.dirname(__file__), style_image_name)
content_image_path = os.path.join(os.path.dirname(__file__), content_image_name)

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
        self.mean = torch.FloatTensor(mean).view(-1, 1, 1)
        self.std = torch.FloatTensor(std).view(-1, 1, 1)
        
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

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
                if args.average_pool:
                    layer = nn.AvgPool2d(2)
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
    image.requires_grad_()
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

def image_reconstruct(tensor):
    newT = tensor.to(torch.device('cpu')).clone()
    newT = newT.squeeze(0)
    return detransform_tensor(newT)

def generate_noise_image(content_image, noise_strength = noise_strength):
    noise_image = torch.tensor(np.random.uniform(-20, 20, (1, 3 ,imsize[0], imsize[1])).astype('float32'))
    noise_image = noise_image.to(device)
    if use_only_noise:
        input_image = noise_image
    else:
        input_image = noise_image * noise_strength + content_image * (1 - noise_strength)
    return input_image.detach()

style_image = image_loader(style_image_path)
content_image = image_loader(content_image_path)

style_image = style_image.to(device)
content_image = content_image.to(device)

model = Model(original_model, content_image, style_image)

input_image = generate_noise_image(content_image)
if torch.cuda.is_available():
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    input_image = input_image.cuda()
    module = model.module
else:
    module = model

input_image.requires_grad_()

optimizer = optim.Adam([input_image], lr = args.learning_rate)
if args.adaptative_lr:
    lr_adapter = LearningRateAdapter(optimizer, args.learning_rate, min_lr=0.01, lr_reduction_percentage=0.7, loss_worsening_count_limit=8)
def train(epoch):
    global input_image
    model.train()
    optimizer.zero_grad()

    model(input_image)

    content_loss = 0
    for ct_loss in module.content_losses:
        content_loss += ct_loss.loss

    style_loss = 0
    i = 0
    for st_loss in module.style_losses:
        style_loss += (st_loss.loss * styleRelativeWeights[i])
        i += 1

    content_loss = content_loss * contentWeight
    style_loss = style_loss * styleWeight

    loss = content_loss + style_loss

    loss.backward()
    optimizer.step()
    print('Content Loss: {}, Style Loss: {}, Total: {}'.format(content_loss, style_loss, loss))
    input_image.data.clamp_(0, 1)
    if args.adaptative_lr:
        lr_adapter.update_loss(loss)
    return (content_loss.item(), style_loss.item(), loss.item())
# #for LBFGS optim
# # Currently not working, learning rate fluctuates a lot and never decreases consistently
# optimizer = optim.LBFGS([input_image])
# def train(epoch):
#     model.train()
#     def closure():
#         input_image.data.clamp_(0, 1)
#         optimizer.zero_grad()
#         model(input_image)

#         content_loss = 0
#         for ct_loss in module.content_losses:
#             content_loss += ct_loss.loss
        
#         style_loss = 0
#         i = 0
#         for st_loss in module.style_losses:
#             style_loss += (st_loss.loss * styleRelativeWeights[i])
#             i += 1

#         content_loss = content_loss * contentWeight
#         style_loss = style_loss * styleWeight

#         loss = content_loss + style_loss
    
#         loss.backward()
#         print('Content Loss: {}, Style Loss: {}, Total: {}'.format(content_loss, style_loss, loss))
#         return content_loss + style_loss
#     optimizer.step(closure)
#     input_image.data.clamp_(0, 1)

def show():
    # f, axarray = plt.subplots(1, 3)
    # axarray[0].imshow(image_reconstruct(content_image.detach()))
    # axarray[0].axis('off')
    # axarray[1].imshow(image_reconstruct(style_image.detach()))
    # axarray[1].axis('off')
    # axarray[2].imshow(image_reconstruct(input_image.detach()))  
    # axarray[2].axis('off')

    plt.imshow(image_reconstruct(input_image.detach()))

    plt.show()

epoch = 0
def go():
    global epoch
    while get_learning_rate(optimizer)[0] > 0.01:
        print('epoch {}'.format(epoch))
        (content_loss, style_loss, total_loss) = train(epoch)
        epoch += 1
        metrics.track({'epoch': epoch, 'content_loss': content_loss, 'style_loss': style_loss, 'total_loss': total_loss })
    metrics.save()
    save()
    show()

def save():
    print('\n\n\nSaving......\n\n\n')
    path = os.path.join(os.path.dirname(__file__), 'outputs/{}.jpg'.format(filename))

    output_img = image_reconstruct(input_image.detach())
    output_img.save(path)

if __name__ == '__main__':
    go()
