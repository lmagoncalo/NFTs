from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn


class EfficientnetB0(nn.Module):
    def __init__(self):
        super(EfficientnetB0, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess

