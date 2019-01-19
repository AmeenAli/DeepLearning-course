import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is:

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.

    If dropout is used, a dropout layer is added after every ReLU.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param: Dropout probability. Zero means no dropout.
        """
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.dropout = dropout

        blocks = []

        if hidden_features:
            blocks.append(Linear(in_features, hidden_features[0]))
        else:
            blocks.append(Linear(in_features, num_classes))
        layers = [feature for feature in hidden_features] + [num_classes]
        for i in range(len(layers) - 1):
            blocks.append(ReLU())
            if self.dropout > 0:
                blocks.append(Dropout(p=self.dropout))
            blocks.append(Linear(layers[i], layers[i+1]))


        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
   
        prev_channels = in_channels
        for i, channels in enumerate(self.filters):
            layers.append(nn.Conv2d(prev_channels, channels, (3, 3), padding=1))
            layers.append(nn.ReLU())
            if (i + 1) % self.pool_every == 0:
                layers.append(torch.nn.MaxPool2d((2, 2)))
            prev_channels = channels


        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
  
        out_channels, out_h, out_w = in_channels, in_h, in_w
        for i, channels in enumerate(self.filters):
            out_channels = channels
            if (i + 1) % self.pool_every == 0:
                out_h = out_h // 2
                out_w = out_w // 2
        prev_out = out_channels * out_h * out_w
        self.classifier_in_size = prev_out
        for features in self.hidden_dims:
            layers.append(nn.Linear(prev_out, features))
            layers.append(nn.ReLU())
            prev_out = features

        layers.append(nn.Linear(prev_out, self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
   
        N = x.shape[0]
        features = self.feature_extractor.forward(x).view(N, self.classifier_in_size)
        out = self.classifier.forward(features)
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)



    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []

        prev_channels = in_channels
        for i, channels in enumerate(self.filters):
            # The first Convolutional layer has 5x5 kernels.
            if i == 0:
                layers.append(nn.Conv2d(prev_channels, channels, (5, 5), padding=2))
            else:
                layers.append(nn.Conv2d(prev_channels, channels, (3, 3), padding=1))
            layers.append(nn.ReLU())
            if (i + 1) % self.pool_every == 0:
                layers.append(torch.nn.MaxPool2d((2, 2)))
            prev_channels = channels

        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []

        out_channels, out_h, out_w = in_channels, in_h, in_w
        for i, channels in enumerate(self.filters):
            out_channels = channels
            if (i + 1) % self.pool_every == 0:
                out_h = out_h // 2
                out_w = out_w // 2
        prev_out = out_channels * out_h * out_w
        self.classifier_in_size = prev_out
        for features in self.hidden_dims:
            layers.append(nn.Linear(prev_out, features))
            layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU())
            prev_out = features

        layers.append(nn.Linear(prev_out, self.out_classes))

        seq = nn.Sequential(*layers)
        return seq

    # ========================
