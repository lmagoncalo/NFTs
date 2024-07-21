import torch
from PIL import Image
from torchvision.transforms import functional as TF

from score import VGG16, MobileNetV2, EfficientnetB0, Inception3, EfficientnetB4, VGG13, VGG11

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

evos = [VGG13, MobileNetV2, Inception3, EfficientnetB0, EfficientnetB4, VGG16, VGG11]

img = Image.open("results/8000.png")

img = TF.pil_to_tensor(img)
img = img.unsqueeze(0).to(device)

print(img.shape)

for evo in evos:
    evaluator = evo().to(device)

    processed_img = evaluator.get_input_preprocessor()(img)

    value = evaluator.predict(processed_img).softmax(1)[:, 954]

    print(evaluator.__class__.__name__, value.item())



