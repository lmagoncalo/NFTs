from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torch import nn


class MobileNetV3Large(nn.Module):
    def __init__(self):
        super(MobileNetV3Large, self).__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT
        self.model = mobilenet_v3_large(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess


