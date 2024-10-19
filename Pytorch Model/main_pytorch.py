import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Unpickle function from Toronto dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

# Load CIFAR-10 data
def test_batch(path):
    test_batch = unpickle(f'{path}/test_batch')
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32)
    test_labels = np.array(test_batch[b'labels'])
    return test_data, test_labels

test_data, test_labels = test_batch('./cifar-10-batches-py')

# Load and preprocess data

test_data = torch.tensor(test_data, dtype=torch.float32) / 255.0
test_labels = torch.tensor(test_labels, dtype=torch.long)

class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return (self.transform(image) if self.transform else image, label)

# Define transformations
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Create dataset and DataLoader
test_dataset = CIFAR10Dataset(test_data, test_labels, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.4)

        # Second Convolutional Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.4)

        # Third Convolutional Block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.5)

        # Fourth Convolutional Block (added for deeper network)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.4)

        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)  # Adjusted to match new output size after the fourth block
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout_fc2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # First Convolutional Block
        x = self.pool1(self.dropout1(self.bn2(F.relu(self.conv2(F.relu(self.bn1(self.conv1(x))))))))

        # Second Convolutional Block
        x = self.pool2(self.dropout2(self.bn4(F.relu(self.conv4(F.relu(self.bn3(self.conv3(x))))))))

        # Third Convolutional Block
        x = self.pool3(self.dropout3(self.bn6(F.relu(self.conv6(F.relu(self.bn5(self.conv5(x))))))))

        # Fourth Convolutional Block
        x = self.pool4(self.dropout4(self.bn8(F.relu(self.conv8(F.relu(self.bn7(self.conv7(x))))))))

        # Flatten for Fully Connected Layers
        x = x.view(-1, 512 * 2 * 2)

        # Fully Connected Layers
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(x))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        x = self.fc3(x)

        return x

model = CIFAR10CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Load and evaluate the model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

accuracy_metric = Accuracy(task='multiclass', num_classes=10).to(device)

# Test evaluation
def evaluate_model(model, test_loader, criterion):
    test_running_loss = 0.0
    accuracy_metric.reset()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            accuracy_metric(outputs, labels)

    test_average_loss = test_running_loss / len(test_loader)
    test_accuracy = accuracy_metric.compute() * 100
    return test_average_loss, test_accuracy

test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Confusion matrix visualization
def plot_confusion_matrix(model, test_loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for CIFAR-10. Pytorch model')
    plt.show()

# Plot the confusion matrix
plot_confusion_matrix(model, test_loader)