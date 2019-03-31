from models import customCNN
from data import train_loader
import torch
import torch.nn as nn
import torchvision
custom_model = customCNN()

"""
sanity check
sample_input = torch.randn(1,3,224,224)
custom_out = custom_model(sample_input)
transfer_out = transfer_model(sample_input)
print(custom_out, transfer_out)
"""
# set up gpu if there is one
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
custom_model.to(device)

# set up loss function and optimization parameters
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(custom_model.parameters(), lr=0.001)

# actual training
for epoch in range(2):  # loop over the training dataset twice

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = custom_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 200 mini-batches
            print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')
