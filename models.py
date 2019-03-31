import torch
import torch.nn as nn
import torch.nn.functional as F

class customCNN(nn.Module):
	def __init__(self):
		super(customCNN, self).__init__()
		# 3 input channels, 15 output channels, 5x5 square convolution
		# kernels
		self.conv1 = nn.Conv2d(3,10,5)
		self.conv2 = nn.Conv2d(10,20,3)
		self.conv3 = nn.Conv2d(20,30,3)
		self.conv4 = nn.Conv2d(30,40,3)
		# affine operations
		self.fc1 = nn.Linear(40*12*12,500)
		self.fc2 = nn.Linear(500,100)
		self.fc3 = nn.Linear(100,20)
		self.fc4 = nn.Linear(20,4)

	def forward(self, x):
		# max pooling over a 2x2 window
		x = F.max_pool2d(F.relu(self.conv1(x)),2)
		x = F.max_pool2d(F.relu(self.conv2(x)),2)
		x = F.max_pool2d(F.relu(self.conv3(x)),2)
		x = F.max_pool2d(F.relu(self.conv4(x)),2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x

	def num_flat_features(self, x):
		# all dimensions except the batch dimension
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features
