from torchvision.models import mnasnet1_3, MNASNet1_3_Weights
from torch import nn


class MNASNet13(nn.Module):
    def __init__(self):
        super(MNASNet13, self).__init__()
        weights = MNASNet1_3_Weights.DEFAULT
        self.model = mnasnet1_3(weights=weights)
        self.model.eval()

        self.preprocess = weights.transforms()

    def predict(self, batch, explain=False):
        return self.model(batch)

    def get_target_size(self):
        return (224, 224)

    def get_input_preprocessor(self):
        return self.preprocess


