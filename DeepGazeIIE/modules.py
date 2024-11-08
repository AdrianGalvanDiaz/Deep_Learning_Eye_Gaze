import functools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GaussianFilterNd

def encode_scanpath_features(x_hist, y_hist, size, device=None, include_x=True, include_y=True, include_duration=False):
    """
    Codifica características de scanpath como en el paper original.
    """
    assert include_x
    assert include_y
    assert not include_duration

    height = size[0]
    width = size[1]

    xs = torch.arange(width, dtype=torch.float32).to(device)
    ys = torch.arange(height, dtype=torch.float32).to(device)
    YS, XS = torch.meshgrid(ys, xs, indexing='ij')

    XS = torch.repeat_interleave(
        torch.repeat_interleave(
            XS[np.newaxis, np.newaxis, :, :],
            repeats=x_hist.shape[0],
            dim=0,
        ),
        repeats=x_hist.shape[1],
        dim=1,
    )

    YS = torch.repeat_interleave(
        torch.repeat_interleave(
            YS[np.newaxis, np.newaxis, :, :],
            repeats=y_hist.shape[0],
            dim=0,
        ),
        repeats=y_hist.shape[1],
        dim=1,
    )

    XS -= x_hist.unsqueeze(2).unsqueeze(3)
    YS -= y_hist.unsqueeze(2).unsqueeze(3)

    distances = torch.sqrt(XS**2 + YS**2)

    return torch.cat((XS, YS, distances), axis=1)

class FeatureExtractor(torch.nn.Module):
    """
    Extractor de características exactamente como en el paper.
    """
    def __init__(self, features, targets):
        super().__init__()
        self.features = features
        self.targets = targets
        self.outputs = {}

        for target in targets:
            layer = dict([*self.features.named_modules()])[target]
            layer.register_forward_hook(self.save_outputs_hook(target))

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self.outputs[layer_id] = output.clone()
        return fn

    def forward(self, x):
        self.outputs.clear()
        self.features(x)
        return [self.outputs[target] for target in self.targets]

class Finalizer(nn.Module):
    """
    Transformador de readout a predicción de mirada, idéntico al original.
    """
    def __init__(self, sigma, kernel_size=None, learn_sigma=False,
                 center_bias_weight=1.0, learn_center_bias_weight=True,
                 saliency_map_factor=4):
        super(Finalizer, self).__init__()
        
        self.saliency_map_factor = saliency_map_factor
        self.gauss = GaussianFilterNd(
            [2, 3], 
            sigma, 
            truncate=3, 
            trainable=learn_sigma
        )
        self.center_bias_weight = nn.Parameter(
            torch.Tensor([center_bias_weight]), 
            requires_grad=learn_center_bias_weight
        )

    def forward(self, readout, centerbias):
        """Aplica los pasos de finalización al readout dado"""
        
        # Redimensionar centerbias
        downscaled_centerbias = F.interpolate(
            centerbias.view(centerbias.shape[0], 1, centerbias.shape[1], centerbias.shape[2]),
            scale_factor=1/self.saliency_map_factor,
            recompute_scale_factor=False,
        )[:, 0, :, :]

        # Redimensionar readout
        out = F.interpolate(
            readout,
            size=[downscaled_centerbias.shape[1], downscaled_centerbias.shape[2]]
        )

        # Aplicar filtro gaussiano
        out = self.gauss(out)

        # Remover dimensión de canal
        out = out[:, 0, :, :]

        # Añadir al center bias
        out = out + self.center_bias_weight * downscaled_centerbias

        # Redimensionar a tamaño original
        out = F.interpolate(
            out[:, np.newaxis, :, :], 
            size=[centerbias.shape[1], centerbias.shape[2]]
        )[:, 0, :, :]

        # Normalizar
        out = out - out.logsumexp(dim=(1, 2), keepdim=True)

        return out

class DeepGazeIIIMixture(torch.nn.Module):
    """
    Implementación exacta del modelo de mezcla DeepGazeIII.
    """
    def __init__(self, features, saliency_networks, scanpath_networks, 
                 fixation_selection_networks, finalizers, downsample=2, 
                 readout_factor=2, saliency_map_factor=2, included_fixations=-2):
        super().__init__()
        
        self.downsample = downsample
        self.readout_factor = readout_factor
        self.saliency_map_factor = saliency_map_factor
        self.included_fixations = included_fixations

        self.features = features
        # Congelar parámetros del extractor
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

        self.saliency_networks = nn.ModuleList(saliency_networks)
        self.scanpath_networks = nn.ModuleList(scanpath_networks)
        self.fixation_selection_networks = nn.ModuleList(fixation_selection_networks)
        self.finalizers = nn.ModuleList(finalizers)

    def forward(self, x, centerbias, x_hist=None, y_hist=None, durations=None):
        orig_shape = x.shape
        x = F.interpolate(x, scale_factor=1/self.downsample)
        x = self.features(x)

        readout_shape = [
            math.ceil(orig_shape[2]/self.downsample/self.readout_factor),
            math.ceil(orig_shape[3]/self.downsample/self.readout_factor)
        ]
        x = [F.interpolate(item, readout_shape) for item in x]
        x = torch.cat(x, dim=1)

        predictions = []
        readout_input = x

        for saliency_network, scanpath_network, fixation_selection_network, finalizer in zip(
            self.saliency_networks, self.scanpath_networks, 
            self.fixation_selection_networks, self.finalizers
        ):
            x = saliency_network(readout_input)

            if scanpath_network is not None:
                scanpath_features = encode_scanpath_features(
                    x_hist, y_hist, 
                    size=(orig_shape[2], orig_shape[3]), 
                    device=x.device
                )
                scanpath_features = F.interpolate(scanpath_features, readout_shape)
                y = scanpath_network(scanpath_features)
            else:
                y = None

            x = fixation_selection_network((x, y))
            x = finalizer(x, centerbias)
            predictions.append(x[:, np.newaxis, :, :])

        predictions = torch.cat(predictions, dim=1) - np.log(len(self.saliency_networks))
        prediction = predictions.logsumexp(dim=1)
        return prediction