import os

import clip
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from losses.PerceptualLoss import PerceptualLoss
from losses.PerceptualLoss2 import VGGPerceptualLoss
from losses.loss import CLIPConvLoss
from render import *
from utils import MakeCutouts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = random.randint(0, 9999)

print("Seed:", seed)

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

num_iter = 1001

mk = MakeCutouts(cut_size=224, cutn=20)

# model, preprocess = clip.load("ViT-B/32", device=device)

color = torch.tensor([0.0,  0.0, 0.0, 1.])

# render = GridDrawRenderer()
render = LineDrawRenderer(img_size=224, color=color)
# render = MergeDrawRenderer()

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

target = Image.open("dogcat.jpg").convert("RGB")
target = transform(target)
target = target.unsqueeze(0)
target = target.to(device)

perception_loss = CLIPConvLoss().to(device)
# perception_loss = VGGPerceptualLoss().to(device)

optims = render.get_opts()

for i in tqdm(range(num_iter)):
    for optim in optims:
        for g in optim.param_groups:
            g['lr'] *= 0.9999

    for optim in optims:
        optim.zero_grad()

    img = render.render()
    img = img.to(device)

    # imgs = mk(img)

    losses_dict = perception_loss(img, target)
    # loss = -perception_loss(img, target, feature_layers=[2], style_layers=[0, 1, 2, 3])

    loss = sum(list(losses_dict.values()))
    # Backpropagate the gradients.
    loss.backward()

    for optim in optims:
        optim.step()

    if i % 100 == 0:
        print(i, loss.item())
        # Save the intermediate render.
        render.save_image("results", i)

    render.clip_z()