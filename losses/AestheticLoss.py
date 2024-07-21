import subprocess
from pathlib import Path

import clip
import torch
import torchvision
from torch import nn
from torch.nn import functional as F


def wget_file(url, out):
    try:
        print(f"Downloading {out} from {url}, please wait")
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)


class AestheticLoss(nn.Module):
    def __init__(self, model=None, preprocess=None, clip_model="ViT-B/16"):
        super(AestheticLoss, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Only available here: https://twitter.com/RiversHaveWings/status/1472346186728173568
        self.model_path = Path("models/ava_vit_b_16_linear.pth")

        if not self.model_path.exists():
            wget_file(
                "https://cdn.discordapp.com/attachments/821173872111517696/921905064333967420/ava_vit_b_16_linear.pth",
                self.model_path)

        layer_weights = torch.load(self.model_path)
        self.ae_reg = nn.Linear(512, 1).to(self.device)
        # self.ae_reg.load_state_dict(torch.load(self.model_path))
        self.ae_reg.bias.data = layer_weights["bias"].to(self.device)
        self.ae_reg.weight.data = layer_weights["weight"].to(self.device)

        self.clip_model = "ViT-B/32"

        if model is None:
            print(f"Loading CLIP model: {clip_model}")

            self.model, self.preprocess = clip.load(self.clip_model, device=self.device)

            print("CLIP module loaded.")
        else:
            self.model = model
            self.preprocess = preprocess

    def forward(self, img):
        img = img.to(self.device)

        # img = torchvision.transforms.functional.resize(img, (224, 224))

        image_features = self.model.encode_image(img)

        aes_rating = -self.ae_reg(F.normalize(image_features.float(), dim=-1)).to(self.device) * 0.02

        return aes_rating
