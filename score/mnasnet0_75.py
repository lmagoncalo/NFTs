from torchvision.models import mnasnet0_75, MNASNet0_75_Weights
from torch import nn


class MNASNet075(nn.Module):
    def __init__(self):
        super(MNASNet075, self).__init__()
        weights = MNASNet0_75_Weights.DEFAULT
        self.model = mnasnet0_75(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess

