from torchvision.models import densenet121, DenseNet121_Weights
from torch import nn


class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        weights = DenseNet121_Weights.DEFAULT
        self.model = densenet121(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess
