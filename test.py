from train import custom_model
from data import test_loader

# set up gpu if there is one
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = images.to(device)
        outputs = custom_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network test dataset: {}'.format(100 * correct / total))
