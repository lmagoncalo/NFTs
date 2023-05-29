import pydiffvg
import torch
from tqdm import tqdm

from render import LineDrawRenderer
from score import VGG11, MobileNetV2, EfficientnetB0
from utils import MakeCutouts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

evos = [VGG11]

evaluators = []

for e in evos:
    evaluators.append(e().to(device))

render = LineDrawRenderer()

optims = render.get_opts()

mk = MakeCutouts(cut_size=224, cutn=50)

for i in tqdm(range(1001)):
    for optim in optims:
        optim.zero_grad()

    img = render.render()

    img = mk(img)

    losses = []
    for evaluator in evaluators:
        processed_img = evaluator.get_input_preprocessor()(img)

        value = evaluator.predict(processed_img).softmax(1)[:, 7].mean()
        losses.append(value)

    losses = torch.stack(losses)
    loss = torch.sum(losses)
    (-loss).backward()

    for optim in optims:
        optim.step()

    if i % 100 == 0:
        print(i, loss.item())
        pydiffvg.save_svg(f"results/{i}.svg", 500, 500, render.shapes, render.shape_groups)

    render.clip_z()
