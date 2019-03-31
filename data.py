# unzip dataset.zip
import os
import zipfile
if not os.path.exists('dataset'):
    zip = zipfile.ZipFile('dataset.zip', 'r')
    zip.extractall()
    zip.close()

# import torch and torchvision
import torch
import torchvision
import torchvision.transforms as transforms

# set parameters for loading data
batch_size = 8

# set a manual seed for reproducibility of results.
torch.manual_seed(7)

# set parameters for image transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# prepare and load training set
train_set = torchvision.datasets.ImageFolder(root="dataset/train", transform = transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 8, shuffle = True, num_workers = 2)

# prepare and load testing set (we don't make use of dev set)
test_set = torchvision.datasets.ImageFolder(root="dataset/test", transform = True)
test_loader = torch.utils.data.DataLoader(test_set, )

# describe image classes
classes = ['clay', 'gravel', 'loam', 'sand']

# visualizing some images
import matplotlib.pyplot as plt
import numpy as np

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # denormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #fix rgb colors order
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(8)))
