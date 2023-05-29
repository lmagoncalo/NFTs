from torchvision.models import vgg16, VGG16_Weights
from torch import nn


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        weights = VGG16_Weights.DEFAULT
        self.model = vgg16(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess
