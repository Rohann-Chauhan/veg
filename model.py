import torch

import torch.nn as nn


# ==========================================
# VGG CONFIGURATIONS
# ==========================================

vgg11_config = [

    64,
    "M",

    128,
    "M",

    256,
    256,
    "M",

    512,
    512,
    "M",

    512,
    512,
    "M"
]


vgg13_config = [

    64,
    64,
    "M",

    128,
    128,
    "M",

    256,
    256,
    "M",

    512,
    512,
    "M",

    512,
    512,
    "M"
]


vgg16_config = [

    64,
    64,
    "M",

    128,
    128,
    "M",

    256,
    256,
    256,
    "M",

    512,
    512,
    512,
    "M",

    512,
    512,
    512,
    "M"
]


vgg19_config = [

    64,
    64,
    "M",

    128,
    128,
    "M",

    256,
    256,
    256,
    256,
    "M",

    512,
    512,
    512,
    512,
    "M",

    512,
    512,
    512,
    512,
    "M"
]


# ==========================================
# CREATE VGG FEATURE LAYERS
# ==========================================

def get_vgg_layers(
    config,
    batch_norm=True
):

    layers = []

    # RGB image
    in_channels = 3


    for c in config:


        # ------------------------------
        # Max Pooling
        # ------------------------------

        if c == "M":

            layers.append(

                nn.MaxPool2d(

                    kernel_size=2,

                    stride=2
                )
            )


        # ------------------------------
        # Convolution
        # ------------------------------

        else:

            conv2d = nn.Conv2d(

                in_channels,

                c,

                kernel_size=3,

                padding=1
            )


            if batch_norm:

                layers.extend([

                    conv2d,

                    nn.BatchNorm2d(c),

                    nn.ReLU(inplace=True)

                ])


            else:

                layers.extend([

                    conv2d,

                    nn.ReLU(inplace=True)

                ])


            # Next layer input
            in_channels = c


    return nn.Sequential(*layers)


# ==========================================
# VGG MODEL
# ==========================================

class VGG(nn.Module):


    def __init__(
        self,
        feature,
        output_dim
    ):

        super().__init__()


        # Convolution layers
        self.feature = feature


        # Make output exactly 7 x 7
        self.avgpool = nn.AdaptiveAvgPool2d(
            (7, 7)
        )


        # Fully connected layers
        self.classifier = nn.Sequential(

            nn.Linear(
                512 * 7 * 7,
                4096
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.5
            ),


            nn.Linear(
                4096,
                4096
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.5
            ),


            # Cloud / Not Cloud
            nn.Linear(
                4096,
                output_dim
            )
        )


    def forward(self, x):

        # CNN feature extraction
        x = self.feature(x)


        # Resize
        x = self.avgpool(x)


        # Flatten
        x = x.view(
            x.shape[0],
            -1
        )


        # Classification
        x = self.classifier(x)


        return x