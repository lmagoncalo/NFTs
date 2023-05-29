from torchvision.models import DenseNet169_Weights, densenet169
from torch import nn


class DenseNet169(nn.Module):
    def __init__(self):
        super(DenseNet169, self).__init__()
        weights = DenseNet169_Weights.DEFAULT
        self.model = densenet169(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess
