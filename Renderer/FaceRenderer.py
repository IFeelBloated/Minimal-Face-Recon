import torch.nn as nn
import numpy as np
from .ParametricFaceModel import ParametricFaceModel
from .MeshRenderer import MeshRenderer

class FaceRenderer(nn.Module):
    def __init__(self, bfm_path, resolution):
        super(FaceRenderer, self).__init__()

        camera_distance = 10.
        focal = 1015 / 224 * resolution
        center = resolution / 2
        rasterize_size = resolution
        fov = 2 * np.arctan(center / focal) * 180 / np.pi
        znear = 5.0
        zfar = 15.0

        self.FaceModel = ParametricFaceModel(bfm_path, camera_distance=camera_distance, focal=focal, center=center)
        self.Renderer = MeshRenderer(fov, znear, zfar, rasterize_size, use_opengl=False)

    def forward(self, x):
        face_vertex, face_texture, face_color, landmark = self.FaceModel(x)
        return self.Renderer(face_vertex, self.FaceModel.face_buf, feat=face_color)