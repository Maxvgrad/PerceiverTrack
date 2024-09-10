# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from typing import Dict, List, Set

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from .position_encoding import build_position_encoding
from ..util.misc import NestedTensor, is_main_process


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 return_interm_layers: bool, layers_used: Set = None):
        super().__init__()
        if layers_used is None:
            layers_used = {'layer1', 'layer2', 'layer3', 'layer4'}
        for name, parameter in backbone.named_parameters():
            if not train_backbone or not any(layer in name for layer in layers_used):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [4, 8, 16, 32]
            self.num_channels = [256, 512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 layers_used: Set[str] = None):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        super().__init__(backbone, train_backbone,
                         return_interm_layers, layers_used)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for x in xs.values():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class IntermediateLayerGetterBackbone(nn.Module):

    def __init__(self, layer: str, backbone: Backbone):
        super().__init__()
        self.layer = layer if layer is not None else '0'
        self.num_channels = [256, 512, 1024, 2048][int(layer)] if layer is not None else backbone.num_channels[-1] # TODO: hack trackformer moved to list for numb of channels
        self.backbone = backbone

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone.forward(tensor_list)
        return xs[self.layer]


def build_backbone(args):
    if not hasattr(args, 'model') or args.model == 'detr':
        position_embedding = build_position_encoding(args)
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks or (args.num_feature_levels > 1)
        backbone = Backbone(args.backbone,
                            train_backbone,
                            return_interm_layers,
                            args.dilation)
        model = Joiner(backbone, position_embedding)
        return model
    elif args.model == 'perceiver':
        if args.backbone == 'resnet50':
            train_backbone = args.lr_backbone > 0
            return_interm_layers = bool(args.interm_layer)
            layers_used = {f'layer{i}' for i in range(1, int(args.interm_layer) + 2)} if return_interm_layers else None
            print(f'interm_layer: {args.interm_layer}, return_interm_layers: {return_interm_layers}, layers_used: {layers_used}')
            backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation,
                                layers_used=layers_used)
            # For perceiver models we return backbone's feature output from particular layer
            return IntermediateLayerGetterBackbone(layer=args.interm_layer, backbone=backbone)

    raise NotImplementedError('Backbone {} not implemented'.format(args.backbone))
