from torchvision.models import vgg13, VGG13_Weights
from torch import nn


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        weights = VGG13_Weights.DEFAULT
        self.model = vgg13(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess


