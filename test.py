import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2

    print("x01", x01.shape, "x02", x02.shape)

    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    print("x1", x1.shape, "x2", x2.shape, "x3", x3.shape, "x4", x4.shape)

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    print("x_LL", x_LL.shape, "x_HL", x_HL.shape, "x_LH", x_LH.shape, "x_HH", x4.shape)

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


# i = torch.rand(1, 3, 224, 224)
img = Image.open("dogcat.jpg")
i = (transforms.ToTensor()(img)).unsqueeze(0)
print(i.shape)

o = dwt_init(i)
print(o.shape)

for i in range(o.shape[1]):
    save_image(o[:, i, :, :], f"results/image_{i}.png")

r = iwt_init(o)
print(r.shape)

save_image(r, "results/image_final.png")
