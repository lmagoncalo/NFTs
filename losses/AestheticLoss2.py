import subprocess
from pathlib import Path

import clip
import torch
import torchvision
from torch import nn
from torch.nn import functional as F


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


# if you changed the MLP architecture during training, change it also here:
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


def wget_file(url, out):
    try:
        print(f"Downloading {out} from {url}, please wait")
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)


class AestheticLoss2(nn.Module):
    def __init__(self, model=None, preprocess=None, clip_model="ViT-L/14"):
        super(AestheticLoss2, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.aesthetic_target = 10

        # Available here: https://github.com/christophschuhmann/improved-aesthetic-predictor
        self.model_path = Path("models/sac+logos+ava1-l14-linearMSE.pth")

        if not self.model_path.exists():
            wget_file(
                "https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
                self.model_path)

        self.mlp = MLP(768).to(self.device)
        self.mlp.load_state_dict(torch.load(self.model_path))
        self.target_rating = torch.ones(size=(1, 1)) * self.aesthetic_target
        self.target_rating = self.target_rating.to(self.device)

        self.clip_model = clip_model

        if model is None:
            print(f"Loading CLIP model: {clip_model}")

            self.model, self.preprocess = clip.load(self.clip_model, device=self.device)

            print("CLIP module loaded.")
        else:
            self.model = model
            self.preprocess = preprocess

    def forward(self, img):
        img = img.to(self.device)

        image_features = self.model.encode_image(img)

        aes_rating = -self.mlp(F.normalize(image_features.float(), dim=-1)).to(self.device)

        aes_loss = (aes_rating - self.target_rating).square().mean() * 0.02

        return aes_loss
