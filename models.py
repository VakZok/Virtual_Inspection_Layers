from torch import nn
import torch.nn.functional as F


class AudioNet(nn.Module):
    """
    This model is a slightly adjusted version of the AudioNet CAFFE model provided by AudioMNIST repository
    (https://github.com/soerenab/AudioMNIST). The motivation behind creating it lies in the library
    compatibility issues faced when setting up the AudioMNIST repository.
    The model is adjusted in the following ways:
        - model is implemented in PyTorch instead of CAFFE, as PyTorch enables the use of Captum
        library for LRP
        - no label slicer nor any silencing logic, as only the digit class is considered in this work
        - implemented ADAM (with lr=0.001) instead of SDG optimizer (with lr=0.0001), as the model's
        performance did not improve when using SDG
        - implemented early stopping and reconstruction of best model parameters to avoid over-fitting
    """

    def __init__(self):
        super(AudioNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 100, kernel_size=3, stride=1, padding=1)
        self.conv1d_output = None
        self.conv2 = nn.Conv1d(100, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)

        # Pooling layers
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Fully-connected layers
        self.fc7 = nn.Linear(15872, 1024)  # size calculated using dummy input earlier
        self.fc8 = nn.Linear(1024, 512)
        self.fc9 = nn.Linear(512, 10)  # 10 classes

        # Dropout layers
        self.drop7 = nn.Dropout(p=0.5)
        self.drop8 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional and pooling layers
        x = F.relu(self.conv1(x))
        self.conv1d_output = x.detach()  # save output of first convolutional layer for LRP
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = F.relu(self.conv6(x))
        x = self.pool6(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc7(x))
        x = self.drop7(x)
        x = F.relu(self.fc8(x))
        x = self.drop8(x)
        x = self.fc9(x)

        return x


class VIL_Model(nn.Module):  # Virtual Inspection Layer Model
    """
    This model follows the approach of Vielhaben et al. in their paper on Virtual Inspection Layers
    (https://doi.org/10.1016/j.patcog.2024.110309), in which the authors attach two linear layers to
    the input. These layers serve the purpose of transforming the audio (time-domain) input data into different
    formats (like frequency-domain), which Layer-wise Relevancy Propagation (LRP) is utilized on to produce a
    different XAI explanation representation.
    """

    def __init__(self, existing_model, transform_layer, invert_transform_layer):
        super(VIL_Model, self).__init__()
        self.transform_layer = transform_layer
        self.invert_transform_layer = invert_transform_layer
        self.reconstructed_audio = None
        self.existing_model = existing_model

    def forward(self, x):
        x = self.transform_layer(x)
        x = self.invert_transform_layer(x)
        self.reconstructed_audio = x.detach()
        x = self.existing_model(x)
        return x
