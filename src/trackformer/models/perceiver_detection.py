import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from trackformer.models import build_backbone
from trackformer.models.perceiver import Perceiver
from trackformer.util.misc import NestedTensor, nested_tensor_from_tensor_list


class PerceiverDetection(nn.Module):

    def __init__(self, backbone, perceiver, classification_head):
        super().__init__()
        self.backbone = backbone
        self.perceiver = perceiver
        self.classification_head = classification_head
        # Compatibility with TrackingBaseModel
        self.num_queries = perceiver.latents.shape[0]
        self.hidden_dim = perceiver.latents.shape[1]
        self.overflow_boxes = False

    def forward(self, samples: NestedTensor, targets: list = None, latents: Tensor = None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features_nested_tensor = self.backbone(samples)
        src, mask = features_nested_tensor.decompose()
        src = src.permute(0, 2, 3, 1)
        assert mask is not None
        if latents is None and (targets and '_hs_embed' in targets[0]):
            latents = torch.stack([target['_hs_embed'] for target in targets]).to(src.device)

        hs = self.perceiver(
            data=src,
            mask=mask,
            return_embeddings=True,
            latents=latents
        )
        out = self.classification_head(hs)

        # TODO: double check if normilization should be disabled
        out['hs_embed'] = hs

        return (
            out,
            targets,
            features_nested_tensor,
            None,  # Memory, is an output from encoder
            hs
        )


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ObjectDetectionHead(nn.Module):
    def __init__(self, num_classes, num_latents, latent_dim):
        """ Initializes the model.
        Parameters:
            num_classes: number of object classes
            num_latents: number of object queries, ie detection slot. This is the maximal number of objects
                         model can detect in a single image. For COCO, we recommend 100 queries.
            latent_dim: dimension of the latent object query.
        """
        super().__init__()
        self.num_queries = num_latents
        self.class_embed = nn.Linear(latent_dim, num_classes + 1)
        self.bbox_embed = MLP(latent_dim, latent_dim, 4, 3)

    def forward(self, hs: Tensor):
        """Forward pass of the ObjectDetectionHead.
            Parameters:
                - hs: Tensor
                    Hidden states from the model, of shape [batch_size x num_queries x latent_dim].

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
        """
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out


def build_model(args, matcher, num_classes):

    device = torch.device(args.device)

    backbone = build_backbone(args)

    num_freq_bands = args.num_freq_bands
    fourier_channels = 2 * ((num_freq_bands * 2) + 1)

    perceiver = Perceiver(
        input_channels=backbone.num_channels,  # number of channels for each token of the input
        input_axis=2,  # number of axis for input data (2 for images, 3 for video)
        num_freq_bands=num_freq_bands,  # number of freq bands, with original value (2 * K + 1)
        max_freq=args.max_freq,  # maximum frequency, hyperparameter depending on how fine the data is
        depth=args.enc_layers,  # depth of net. The shape of the final attention mechanism will be:
        #   depth * (cross attention -> self_per_cross_attn * self attention)
        num_latents=args.num_queries,
        # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim=args.hidden_dim,  # latent dimension
        cross_heads=args.enc_nheads_cross,  # number of heads for cross attention. paper said 1
        latent_heads=args.nheads,  # number of heads for latent self attention, 8
        cross_dim_head=(backbone.num_channels + fourier_channels) // args.enc_nheads_cross,
        # number of dimensions per cross attention head
        latent_dim_head=args.hidden_dim // args.nheads,  # number of dimensions per latent self attention head
        num_classes=-1,  # NOT USED. output number of classes.
        attn_dropout=args.dropout,
        ff_dropout=args.dropout,
        weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
        fourier_encode_data=True,
        # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
        self_per_cross_attn=args.self_per_cross_attn,  # number of self attention blocks per cross attention
        final_classifier_head=False  # mean pool and project embeddings to number of classes (num_classes) at the end
    )

    classifier_head = ObjectDetectionHead(
        num_classes=num_classes,
        num_latents=args.num_queries,
        latent_dim=args.hidden_dim
    )

    return backbone, perceiver, classifier_head
