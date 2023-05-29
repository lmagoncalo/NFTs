from torchvision.models import resnet50, ResNet50_Weights
from torch import nn


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess
