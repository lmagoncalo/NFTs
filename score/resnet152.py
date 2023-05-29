from torchvision.models import resnet152, ResNet152_Weights
from torch import nn


class ResNet152(nn.Module):
    def __init__(self):
        super(ResNet152, self).__init__()
        weights = ResNet152_Weights.DEFAULT
        self.model = resnet152(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess
