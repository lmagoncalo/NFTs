from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from torch import nn


class EfficientnetB7(nn.Module):
    def __init__(self):
        super(EfficientnetB7, self).__init__()
        weights = EfficientNet_B7_Weights.DEFAULT
        self.model = efficientnet_b7(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess

