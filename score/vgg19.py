from torchvision.models import vgg19, VGG19_Weights
from torch import nn


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        weights = VGG19_Weights.DEFAULT
        self.model = vgg19(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess
