"""
import random

import pydiffvg
import torch


class LineDrawRenderer:
    def __init__(self, num_lines=4, img_size=224, ):
        super(LineDrawRenderer, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_lines = num_lines
        self.img_size = img_size

        self.max_width = 2 * self.img_size / 100
        self.min_width = 0.5 * self.img_size / 100

        self.stroke_length = 4

        shapes = []
        shape_groups = []

        # Initialize Random Curves
        for i in range(self.num_lines):
            num_segments = self.stroke_length
            num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                p1 = (random.random(), random.random())
                p2 = (random.random(), random.random())
                p3 = (random.random(), random.random())
                points.append(p1)
                points.append(p2)
                points.append(p3)
            points = torch.tensor(points)
            points[:, 0] *= self.img_size
            points[:, 1] *= self.img_size
            path = pydiffvg.Path(num_control_points=num_control_points, points=points,
                                 stroke_width=torch.tensor(self.max_width / 10), is_closed=False)
            shapes.append(path)
            s_col = [random.random(), random.random(), random.random(), 1.0]
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
                                             stroke_color=torch.tensor(s_col))
            shape_groups.append(path_group)

        color_vars = []
        points_vars = []
        stroke_width_vars = []
        for path in shapes[1:]:
            path.points.requires_grad = True
            points_vars.append(path.points)
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)

        for group in shape_groups:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)

        self.points_vars = points_vars
        self.stroke_width_vars = stroke_width_vars
        self.color_vars = color_vars
        self.shapes = shapes
        self.shape_groups = shape_groups

    def __str__(self):
        return "linedraw"

    def get_opts(self):
        # Optimizers
        points_optim = torch.optim.Adam(self.points_vars, lr=1.0)
        width_optim = torch.optim.Adam(self.stroke_width_vars, lr=0.1)
        color_optim = torch.optim.Adam(self.color_vars, lr=0.01)
        opts = [points_optim, width_optim, color_optim]
        return opts

    def render(self):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.img_size, self.img_size, self.shapes, self.shape_groups)
        img = render(self.img_size, self.img_size, 2, 2, 0, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        return img

    def clip_z(self):
        with torch.no_grad():
            for path in self.shapes:
                path.stroke_width.data.clamp_(self.min_width, self.max_width)
            for group in self.shape_groups[1:]:
                group.stroke_color.data.clamp_(0.0, 1.0)
"""

import random

import numpy as np
import pydiffvg
import torch


class LineDrawRenderer:
    def __init__(self, img_size=224):
        super(LineDrawRenderer, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.img_size = img_size

        self.num_cells = 16

        shapes = []
        shape_groups = []
        colors = []
        cell_size = int(self.img_size / self.num_cells)
        for r in range(self.num_cells):
            cur_y = r * cell_size
            for c in range(self.num_cells):
                cur_x = c * cell_size
                p0 = [cur_x, cur_y]
                p1 = [cur_x + cell_size, cur_y + cell_size]

                cell_color = torch.tensor([random.random(), random.random(), random.random(), 1.0])
                colors.append(cell_color)

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
        return "linedraw"

    def get_opts(self):
        # Optimizers
        optims = [torch.optim.Adam(self.color_vars, lr=0.01)]
        return optims

    def render(self):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.img_size, self.img_size, self.shapes, self.shape_groups)
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
