import random

import numpy as np
import pydiffvg
import torch


class GridDrawRenderer:
    def __init__(self, img_size=224):
        super(GridDrawRenderer, self).__init__()

        # self.img_size = img_size

        # 1400 x 350
        # 90

        self.x_num_cells = 90
        self.y_num_cells = 22

        self.x_img_size = self.x_num_cells * 16
        self.y_img_size = self.y_num_cells * 16

        shapes = []
        shape_groups = []
        cell_size = 16
        for r in range(self.y_num_cells):
            cur_y = r * cell_size
            for c in range(self.x_num_cells):
                cur_x = c * cell_size
                p0 = [cur_x, cur_y]
                p1 = [cur_x + cell_size, cur_y + cell_size]

                cell_color = torch.tensor([random.random(), random.random(), random.random(), 1.0])

                path = pydiffvg.Rect(p_min=torch.tensor(p0), p_max=torch.tensor(p1))
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), stroke_color=None,
                                                 fill_color=cell_color)
                shape_groups.append(path_group)

        color_vars = []
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)

        self.shapes = shapes
        self.shape_groups = shape_groups

        self.color_vars = color_vars

    def __str__(self):
        return "griddraw"

    def get_opts(self):
        # Optimizers
        optims = [torch.optim.Adam(self.color_vars, lr=0.01)]
        return optims

    def render(self):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.x_img_size, self.y_img_size, self.shapes,
                                                             self.shape_groups)
        img = render(self.x_img_size, self.y_img_size, 2, 2, 0, None, *scene_args)

        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        return img

    def clip_z(self):
        with torch.no_grad():
            for group in self.shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)

    def save_image(self, folder_name, i):
        """
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.img_size, self.img_size, self.shapes,
                                                             self.shape_groups)
        img = render(self.img_size, self.img_size, 2, 2, 0, None, *scene_args)

        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        pydiffvg.imwrite(img.cpu(), f"results/{i}.png")
        """

        pydiffvg.save_svg(f"results/{folder_name}/{i}.svg", self.x_img_size, self.y_img_size, self.shapes, self.shape_groups)


class BlobDrawRenderer:
    def __init__(self, img_size=224):
        super(BlobDrawRenderer, self).__init__()

        self.img_size = img_size

        radius = 100
        num_segments = 20
        num_control_points = [2] * num_segments
        bias = (int(img_size / 2), int(img_size / 2))
        points = []
        avg_degree = 360 / (num_segments * 3)
        for i in range(0, num_segments * 3):
            point = (np.cos(np.deg2rad(i * avg_degree)),
                     np.sin(np.deg2rad(i * avg_degree)))
            points.append(point)
        points = torch.tensor(points)
        points = (points) * radius + torch.tensor(bias).unsqueeze(dim=0)
        points = points.type(torch.FloatTensor)

        path = pydiffvg.Path(num_control_points=torch.LongTensor(num_control_points), points=points,
                             stroke_width=torch.tensor(0.0), is_closed=True)
        self.shapes = [path]

        polygon_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor(
            [random.random(), random.random(), random.random(), 1.]),
                                            stroke_color=None)
        self.shape_groups = [polygon_group]

        path.points.requires_grad = True
        self.points_vars = [path.points]

        polygon_group.fill_color.requires_grad = True
        self.color_vars = [polygon_group.fill_color]

    def __str__(self):
        return "blobdraw"

    def get_opts(self):
        # Optimizers
        optims = [torch.optim.Adam(self.points_vars, lr=0.1), torch.optim.Adam(self.color_vars, lr=0.01)]
        return optims

    def render(self):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.img_size, self.img_size, self.shapes,
                                                             self.shape_groups)
        img = render(self.img_size, self.img_size, 2, 2, 0, None, *scene_args)

        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        return img

    def clip_z(self):
        with torch.no_grad():
            for group in self.shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)


class LineDrawRenderer:
    def __init__(self, img_size=224, color=None):
        super(LineDrawRenderer, self).__init__()

        self.img_size = img_size

        self.num_strokes = 10

        shapes = []
        shape_groups = []
        num_control_points = torch.tensor([1])

        radius = 0.05

        # Red ones
        for i in range(self.num_strokes):
            p0 = (random.random(), random.random())
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            points = torch.tensor([p0, p1, p2])

            points[:, 0] *= self.img_size
            points[:, 1] *= self.img_size

            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 is_closed=False,
                                 stroke_width=torch.tensor(random.random()))
            shapes.append(path)
            if color is not None:
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                                 stroke_color=color,
                                                 # stroke_color=torch.tensor([255 / 255, 190 / 255, 0 / 255, 1.]),
                                                 fill_color=None)
            else:
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                                 stroke_color=torch.tensor([random.random(), random.random(), random.random(), 1.]),
                                                 # stroke_color=torch.tensor([255 / 255, 190 / 255, 0 / 255, 1.]),
                                                 fill_color=None)
            shape_groups.append(path_group)

        # Black ones
        num_control_points = torch.tensor([1])
        for i in range(self.num_strokes):
            p0 = (random.random(), random.random())
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            points = torch.tensor([p0, p1, p2])

            points[:, 0] *= self.img_size
            points[:, 1] *= self.img_size

            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 is_closed=False,
                                 stroke_width=torch.tensor(random.random()))
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                             stroke_color=torch.tensor([13/255, 13/255, 13/255, 1.]),
                                             fill_color=None)
            shape_groups.append(path_group)

        self.shapes = shapes
        self.shape_groups = shape_groups

        self.points_vars = []
        self.stroke_width_vars = []
        self.color_vars = []
        for path in shapes:
            path.points.requires_grad = True
            self.points_vars.append(path.points)

            path.stroke_width.requires_grad = True
            self.stroke_width_vars.append(path.stroke_width)

        if color is None:
            for group in shape_groups[:self.num_strokes]:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)

    def __str__(self):
        return "linedraw"

    def get_opts(self):
        # Optimizers
        optims = [torch.optim.Adam(self.points_vars, lr=1.0), torch.optim.Adam(self.stroke_width_vars, lr=0.1)]

        if len(self.color_vars) > 0:
            optims.append(torch.optim.Adam(self.color_vars, lr=0.01))

        return optims

    def save_image(self, folder_name, i):
        """
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.img_size, self.img_size, self.shapes,
                                                             self.shape_groups)
        img = render(self.img_size, self.img_size, 2, 2, 0, None, *scene_args)

        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        pydiffvg.imwrite(img.cpu(), f"results/{i}.png")
        """

        pydiffvg.save_svg(f"{folder_name}/{i}.svg", self.img_size, self.img_size, self.shapes, self.shape_groups)

    def render(self):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.img_size, self.img_size, self.shapes,
                                                             self.shape_groups)
        img = render(self.img_size, self.img_size, 2, 2, 0, None, *scene_args)

        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        return img

    def clip_z(self):
        for i in range(0, self.num_strokes):
            self.shapes[i].stroke_width.data.clamp_(5.0, 20.0)

        for i in range(self.num_strokes, len(self.shapes)):
            self.shapes[i].stroke_width.data.clamp_(0.5, 1.)


class MergeDrawRenderer:
    def __init__(self, img_size=224):
        super(MergeDrawRenderer, self).__init__()

        self.img_size = img_size

        self.num_strokes = 10

        radius = torch.tensor(random.random()) * self.img_size
        center = torch.tensor([random.random(), random.random()]) * self.img_size
        ellipse = pydiffvg.Circle(radius=radius, center=center)

        self.shapes = [ellipse]

        polygon_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor([242/255, 71/255, 56/255, 1.]),
                                            stroke_color=None)
        self.shape_groups = [polygon_group]

        # Black ones
        num_control_points = torch.tensor([0])
        for i in range(self.num_strokes):
            points = torch.tensor([[random.random(), random.random()],  # base
                                   [random.random(), random.random()]])  # base

            points[:, 0] *= self.img_size
            points[:, 1] *= self.img_size

            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 is_closed=False,
                                 stroke_width=torch.tensor(random.random()))
            self.shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                             stroke_color=torch.tensor([13/255, 13/255, 13/255, 1.]),
                                             fill_color=None)
            self.shape_groups.append(path_group)

        self.points_vars = []
        self.stroke_width_vars = []

        self.shapes[0].radius.requires_grad = True
        self.points_vars.append(self.shapes[0].radius)
        self.shapes[0].center.requires_grad = True
        self.points_vars.append(self.shapes[0].center)

        for path in self.shapes[1:]:
            path.points.requires_grad = True
            self.points_vars.append(path.points)
        for path in self.shapes[1:]:
            path.stroke_width.requires_grad = True
            self.stroke_width_vars.append(path.stroke_width)

    def __str__(self):
        return "linedraw"

    def get_opts(self):
        # Optimizers
        optims = [torch.optim.Adam(self.points_vars, lr=1.0), torch.optim.Adam(self.stroke_width_vars, lr=0.1)]
        return optims

    def render(self):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.img_size, self.img_size, self.shapes,
                                                             self.shape_groups)
        img = render(self.img_size, self.img_size, 2, 2, 0, None, *scene_args)

        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        return img

    def clip_z(self):
        # for i in range(0, self.num_strokes):
        #     self.shapes[i].stroke_width.data.clamp_(5.0, 20.0)

        # for i in range(self.num_strokes, len(self.shapes)):
        for i in range(0, self.num_strokes):
            self.shapes[i].stroke_width.data.clamp_(0.5, 1.)


class ColorDrawRenderer:
    def __init__(self, img_size=224):
        super(ColorDrawRenderer, self).__init__()

        self.img_size = img_size

        cell_color = torch.tensor([random.random(), random.random(), random.random(), 1.0])

        path = pydiffvg.Rect(p_min=torch.tensor([0, 0]), p_max=torch.tensor([self.img_size, self.img_size]))
        shapes = [path]
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), stroke_color=None,
                                         fill_color=cell_color)
        shape_groups = [path_group]

        color_vars = []
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)

        self.shapes = shapes
        self.shape_groups = shape_groups

        self.color_vars = color_vars

    def __str__(self):
        return "colordraw"

    def get_opts(self):
        # Optimizers
        optims = [torch.optim.Adam(self.color_vars, lr=0.01)]
        return optims

    def render(self):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.img_size, self.img_size, self.shapes,
                                                             self.shape_groups)
        img = render(self.img_size, self.img_size, 2, 2, 0, None, *scene_args)

        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        return img

    def clip_z(self):
        with torch.no_grad():
            for group in self.shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)

    def save_image(self, i):
        pydiffvg.save_svg(f"results/{i}.svg", self.img_size, self.img_size, self.shapes, self.shape_groups)