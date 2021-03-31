import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from image_color_channel_display import get_train_and_test_loaders, imshow


class Net(nn.Module):
    """Architecture following this example:
       https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ColorNet(nn.Module):
    """Model: early tensor fusion on individual color features"""
    def __init__(self, r, g, b):
        super(ColorNet, self).__init__()
        self.fc1 = nn.Linear(30, 10)
        self.r = r
        self.g = g
        self.b = b

    def forward(self, x):
        with torch.no_grad():
            k, _, h, w = x.shape  # k is batch size
            zc = torch.zeros((k, 1, h, w))
            oc = torch.ones((k, 1, h, w))  #zero channel and one channel
            r_mask = torch.cat((oc, zc, zc), 1)
            g_mask = torch.cat((zc, oc, zc), 1)
            b_mask = torch.cat((zc, zc, oc), 1)
            rpred = self.r(x * r_mask)
            gpred = self.g(x * g_mask)
            bpred = self.b(x * b_mask)
            y = torch.cat((rpred, gpred, bpred), 1)
        return self.fc1(y)


def get_net(color, save_model=False, num_epochs=5):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_loader, test_loader = get_train_and_test_loaders(color)

    def train():
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 12000 == 11999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 12000))
        print('Finished Training')

    def save():
        """save model.
        load with net.load_state_dict(torch.load(PATH))
        """
        PATH = './cifar_net_' + color + '.pth'
        torch.save(net.state_dict(), PATH)

    train()
    if save_model:
        save()

    return net


def load_color_model(color, path="default"):
    """by default, loads the model saved in './cifar_net_$COLOR.pth
    or: specify an explicit path via path="path/to/model"
    """
    model = Net()
    if color == "fuser":
        r, g, b = load_color_model("red"), load_color_model(
            "green"), load_color_model("blue")
        model = ColorNet(r, g, b)
    if path == "default":
        path = './cifar_net_' + color + '.pth'
    model.load_state_dict(torch.load(path))
    return model


def build_tricolor():
    r = load_color_model("red")
    g = load_color_model("green")
    b = load_color_model("blue")
    net = ColorNet(r, g, b)
    return net


def train_and_save_tricolor(net, save_model=False, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_loader, test_loader = get_train_and_test_loaders("all")

    def train():
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 12000 == 11999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 12000))
        print('Finished Training')

    def save():
        """save model.
        load with net.load_state_dict(torch.load(PATH))
        """
        PATH = './cifar_net_' + "fuser" + '.pth'
        torch.save(net.state_dict(), PATH)

    train()
    if save_model:
        save()

    return net


def train_and_save_all_models(num_epochs=5):
    """create and save 5 models"""
    for color in ["red", "green", "blue", "all"]:
        get_net(color, save_model=True, num_epochs=num_epochs)
    train_and_save_tricolor(build_tricolor(), save_model=True, num_epochs=num_epochs)
    return


def _test_container():
    """Batch of low-rigor tests"""
    def sanity_test(model, test_loader):
        """Test one batch, print ground truth and predictions."""
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                   'horse', 'ship', 'truck')
        test_iter = iter(test_loader)
        images, labels = test_iter.next()
        # check images for debug
        # imshow(torchvision.utils.make_grid(images))
        print('Ground: ',
              ' '.join('%5s' % classes[labels[j]] for j in range(4)))

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ',
              ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    def load_and_test(color):
        """Load a color model and test one batch, printing ground/prediction."""
        model = load_color_model(color)
        train_ldr, test_ldr = get_train_and_test_loaders(color)
        sanity_test(model, test_ldr)

    def sanity_test_all_models():
        """Test one batch for all models. Batch is chosen deterministically."""
        for color in ["red", "green", "blue", "all"]:
            print("Using color: {}".format(color))
            load_test(color)
            print("\n")

    return


def full_test(model, test_loader):
    correct, total = 0, 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Check performance on each class
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy on 10k test images: %.3f %%' % (100 * correct / total))
    for i in range(10):
        print('Accuracy of %5s : %.3f %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))


def full_test_all():
    for color in ["fuser", "red", "green", "blue", "all"]:
        model = load_color_model(color)
        _, test_loader = get_train_and_test_loaders(color)
        print('\nRunning full test suite on color {}:\n'.format(color))
        full_test(model, test_loader)


def main():
    try:
        full_test_all()
    except FileNotFoundError:  # Create new models if they cannot be loaded.
        train_and_save_all_models(num_epochs=1)
        full_test_all()


if __name__ == "__main__":
    main()
