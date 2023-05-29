from torchvision.models import mnasnet0_5, MNASNet0_5_Weights
from torch import nn


class MNASNet05(nn.Module):
    def __init__(self):
        super(MNASNet05, self).__init__()
        weights = MNASNet0_5_Weights.DEFAULT
        self.model = mnasnet0_5(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess

