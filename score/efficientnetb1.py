from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torch import nn


class EfficientnetB1(nn.Module):
    def __init__(self):
        super(EfficientnetB1, self).__init__()
        weights = EfficientNet_B1_Weights.DEFAULT
        self.model = efficientnet_b1(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess

