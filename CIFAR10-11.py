from torch import nn
import torch.functional as F

class CIFAR10Network(nn.Module):
    def __init__(self):
        super().__init__()

        # input layer is conv layer that takes in 3x32x32 image(s)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.batch_norm_conv1 = nn.BatchNorm2d(32)

        # convolutional layer 2
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)  # 3x3 kernel with 1 padding, so H,W dimensions do not change
        self.batch_norm_conv2 = nn.BatchNorm2d(32)

        # convolutional layer 3
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.batch_norm_conv3 = nn.BatchNorm2d(64)

        # convolutional layer 4
        self.conv4 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.batch_norm_conv4 = nn.BatchNorm2d(64)

        ## convolutional layer 5
        self.conv5 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.batch_norm_conv5 = nn.BatchNorm2d(128)

        # conv layer 6
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.batch_norm_conv6 = nn.BatchNorm2d(128)

        # conv 7
        self.conv7 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.batch_norm_conv7 = nn.BatchNorm2d(256)

        # conv 8
        self.conv8 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.batch_norm_conv8 = nn.BatchNorm2d(256)

        # conv 9
        self.conv9 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.batch_norm_conv9 = nn.BatchNorm2d(512)

        # max pooling layer (reduce H,W dimensions)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # fully connected Linear layer. Linear layer takes in 1D vector of length in_features
        # So image input must be flattened before going into a Linear layer
        self.fc1 = nn.Linear(in_features=512 * 4 * 4, out_features=500)  # 128*4*4 is C*H*W
        self.batch_norm_fc1 = nn.BatchNorm1d(500)

        # fully connected Linear layer
        self.fc2 = nn.Linear(500, 10)  # 10 output features, for the 10 classes in CIFAR10

        # dropout layer with p=0.5
        self.dropout = nn.Dropout(0.5)
        # no batch normalization after final fc layer

    def forward(self, x):
        x = F.relu(self.batch_norm_conv1(self.conv1(x)))
        x = F.relu(self.batch_norm_conv2(self.conv2(x)))
        x = F.relu(self.batch_norm_conv3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm_conv4(self.conv4(x)))
        x = F.relu(self.batch_norm_conv5(self.conv5(x)))
        x = F.relu(self.batch_norm_conv6(self.conv6(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm_conv7(self.conv7(x)))
        x = F.relu(self.batch_norm_conv8(self.conv8(x)))
        x = F.relu(self.batch_norm_conv9(self.conv9(x)))
        x = self.pool(x)

        # Note: in case of a maxpooling layer and relu activation function,
        #       maxpool(relu(conv(x))) = relu(maxpool(conv(x)))
        #       (relu is an element-wise, monotonically increasing, non-linear function)

        # flatten image input
        x = x.view(-1, 512 * 4 * 4)

        # dropout layer before fully connected Linear layers
        x = self.dropout(x)

        # 1st hidden layer with relu activation function
        x = F.relu(self.batch_norm_fc1(self.fc1(x)))

        # dropout
        x = self.dropout(x)

        # 2nd hidden layer (what activation function?? relu, softmax, log-softmax, ...)
        x = self.fc2(x)

        return x