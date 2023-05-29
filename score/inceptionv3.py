from torchvision.models import inception_v3, Inception_V3_Weights
from torch import nn


class Inception3(nn.Module):
    def __init__(self):
        super(Inception3, self).__init__()
        weights = inception_v3.DEFAULT
        self.model = Inception_V3_Weights(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (299, 299)

    def get_input_preprocessor(self):
        return self.preprocess

