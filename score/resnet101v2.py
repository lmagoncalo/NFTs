from torchvision.models import resnet101, ResNet101_Weights
from torch import nn


class ResNet101v2(nn.Module):
    def __init__(self):
        super(ResNet101v2, self).__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2
        self.model = resnet101(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess
