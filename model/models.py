# %%
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import lovely_tensors

lovely_tensors.monkey_patch()

# %%
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../mnist_data",
        download=True,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
            ]
        ),
    ),
    batch_size=16,
    shuffle=True,
)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../mnist_data",
        download=True,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
            ]
        ),
    ),
    batch_size=16,
    shuffle=True,
)


# %%
class MNISTCLassifierA(nn.Module):
    def __init__(self):
        super(MNISTCLassifierA, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(64, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        print(x)
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = self.dropout1(x)
        # print(x)
        x = x.view(-1, 64 * 20 * 20)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.Softmax()(x)


# %%
MNISTCLassifierB = nn.Sequential(
    *[
        nn.Conv2d(1, 64, kernel_size=8, stride=8),
        nn.ReLU(),
        nn.Dropout2d(p=0.2),
        nn.Conv2d(64, 128, kernel_size=6, stride=6),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=5, stride=5),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(128, 10),
        nn.Softmax()
    ]
)

MNISTCLassifierC = nn.Sequential(
    *[
        nn.Conv2d(1, 64, kernel_size=8, stride=8),
        nn.ReLU(),
        nn.Dropout2d(p=0.2),
        nn.Conv2d(64, 128, kernel_size=6, stride=6),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=5, stride=5),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(128, 10),
        nn.Softmax()
    ]
)


# %%
clf = MNISTCLassifierA()
opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)
ce_loss = torch.nn.CrossEntropyLoss()

loss_history = []
acc_history = []


def train(epoch):
    clf.train()  # set model in training mode (need this because of dropout)

    # dataset API gives us pythonic batching
    for batch_id, (data, label) in enumerate(train_loader):
        data = Variable(data)
        # print("DATA",data)
        target = Variable(label)

        # forward pass, calculate loss and backprop!
        opt.zero_grad()
        preds = clf(data)
        # print(preds)
        # print(target)
        loss = ce_loss(preds, target)
        loss.backward()
        loss_history.append(loss.item())
        opt.step()

        if batch_id % 100 == 0:
            print(loss.item())


# %%
def test(epoch):
    clf.eval()  # set model in inference mode (need this because of dropout)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = Variable(data)
            target = Variable(target)

            output = clf(data)
            test_loss += ce_loss(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    accuracy = 100.0 * correct / len(test_loader.dataset)
    acc_history.append(accuracy)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )


# %%
for epoch in range(0, 3):
    print("Epoch %d" % epoch)
    train(epoch)
    test(epoch)

# %%
