from torchvision.models import DenseNet201_Weights, densenet201
from torch import nn


class DenseNet201(nn.Module):
    def __init__(self):
        super(DenseNet201, self).__init__()
        weights = DenseNet201_Weights.DEFAULT
        self.model = densenet201(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess
