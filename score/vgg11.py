from torchvision.models import vgg11, VGG11_Weights
from torch import nn


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        weights = VGG11_Weights.DEFAULT
        self.model = vgg11(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess


