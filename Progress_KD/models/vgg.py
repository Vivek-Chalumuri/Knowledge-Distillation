'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm, input_channels=3):
    """
    Helper function to help build the traditional Vgg network
    """
    layers = []
    in_channels = input_channels
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d
            layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
            # Batch norm
            if batch_norm:
                layers.append(nn.BatchNorm2d(x))
            # relu
            layers.append(nn.ReLU(inplace=True))
            in_channels = x
    return nn.Sequential(*layers)

def make_student_layers(cfg, batch_norm, reduce_factor, input_channels=3):
    layers = []
    in_channels = input_channels
    for x in cfg:
        if x == 'M':
            # add back a conpensation convolution network to make block output
            # consistent
            out_channels = in_channels * reduce_factor
            # conv2d
            layers.append(
                nn.Conv2d(in_channels, out_channels,
                kernel_size=1, padding=0))
            # Batch norm
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            # relu
            layers.append(nn.ReLU(inplace=True))

            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # adjust the next layer input channels
            in_channels = in_channels * reduce_factor
        else:
            out_channels = x // reduce_factor
            # conv2d
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            # Batch norm
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            # relu
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

    # as we use in_channels as the storage
    out_channels = in_channels

    return nn.Sequential(*layers), out_channels

# =============================================================================

class Vgg(nn.Module):
    def __init__(self, vgg_name, batch_norm=False, n_classes=10):
        super().__init__()
        self.vgg_name = vgg_name
        self.features = self._make_layers(cfg[vgg_name], batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        #
        self._cross_entropy_loss_fn = nn.CrossEntropyLoss()
        # He Initialization scheme
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, batch_norm):
        return make_layers(cfg, batch_norm)

    def get_loss(self, outputs, labels):
        ret = self._cross_entropy_loss_fn(outputs, labels)
        return ret

class VggStudent(Vgg):

    def __init__(self, vgg_name, reduce_factor=2, batch_norm=False, n_classes=10):
        self.reduce_factor = reduce_factor
        super().__init__(vgg_name, batch_norm, n_classes)

    def _make_layers(self, cfg, batch_norm):
        seq_container, _ = make_student_layers(cfg, batch_norm, self.reduce_factor)
        return seq_container


