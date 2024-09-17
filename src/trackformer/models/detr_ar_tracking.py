from collections import deque

import torch
import torch.nn as nn

from .deformable_detr import DeformableDETR
from .detr import DETR
from ..util.misc import NestedTensor


class DETRArTrackingBase(nn.Module):

    def __init__(self,
                 obj_detector_post,
                 track_obj_score_threshold: float = 0.4,
                 ):
        self._obj_detector_post = obj_detector_post
        self._track_obj_score_threshold = track_obj_score_threshold

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        src, mask = samples.decompose()

        if len(src.shape) < 5:
            # samples without a time dimension
            raise NotImplementedError("Not implemented yet samples without a time dimension.")

        src = src.permute(1, 0, 2, 3, 4)  # change dimension order from BT___ to TB___

        result = {'pred_logits': [], 'pred_boxes': []}
        latents = None
        max_num_of_frames_lookback = 0 if self.training else self._max_num_of_frames_lookback
        # 0 index in deque is kept for current timestamp's output latents
        # indecision from 1 to max_num_of_frames_lookback reserved for latents which was produced by dripping frame
        # index is equal to how many times frame was dropped in a row
        # e.g. if index is 3 then latent from timestamp-3 was fed into the model 3 times without frame input
        latent_deque = deque(maxlen=max_num_of_frames_lookback + 1)
        targets_flat = []
        hs_embeds = []
        orig_size = torch.stack([t[-1]["orig_size"] for t in targets], dim=0).to(src.device)

        for timestamp, batch in enumerate(src):
            current_targets = [target_list[timestamp] for target_list in targets]

            if self.training:
                frame_keep_mask = [t['keep_frame'] for t in current_targets]
                frame_keep_mask = torch.tensor(frame_keep_mask, device=batch.device)
                frame_keep_mask = frame_keep_mask.view(-1, 1, 1, 1)
                batch = batch * frame_keep_mask
            else:
                for current_target in current_targets:
                    current_target['consecutive_frame_skip_number'] = torch.tensor(0, device=batch.device)

            if len(hs_embeds) > 0:
                for i, current_target in enumerate(current_targets):
                    current_target['track_query_hs_embeds'] = hs_embeds[i][0]
                    current_target['track_query_boxes'] = hs_embeds[i][1]

            out, targets_resp, features, memory, hs = self.super().forward(
                samples=batch, targets=current_targets
            )

            post_process_results = self._obj_detector_post['bbox'](out, orig_size)

            hs_embeds = []

            for i, post_process_result in enumerate(post_process_results):

                track_scores = post_process_result['scores']

                track_keep = torch.logical_and(
                    track_scores > self.track_obj_score_thresh,
                    post_process_result['labels'][:] == 0
                )

                hs_embeds.append(
                    (out['hs_embed'][i][track_keep], post_process_result['boxes'][track_keep])
                )

            latent_deque.appendleft(out['hs_embed'])

            if 'boxes' in current_targets[0] or self._debug:
                # frame has annotations then include it in output
                result['pred_logits'].append(out['pred_logits'])
                result['pred_boxes'].append(out['pred_boxes'])
                targets_flat.extend(current_targets)

            for num_frames_lookback in range(1, 1 + max_num_of_frames_lookback):
                if num_frames_lookback == len(latent_deque):
                    # In the beginning of the sequence there's not enough latents for lookback
                    break

                latents = latent_deque[num_frames_lookback]

                zero_batch = torch.zeros_like(batch)

                out, *_ = super().forward(
                    samples=zero_batch, targets=current_targets, latents=latents
                )

                latent_deque[num_frames_lookback] = out['hs_embed']

                if 'boxes' in current_targets[0]:
                    current_targets = current_targets.copy()
                    current_targets = [current_target.copy() for current_target in current_targets]
                    for current_target in current_targets:
                        current_target['consecutive_frame_skip_number'] = torch.tensor(num_frames_lookback, device=batch.device)

                    result['pred_logits'].append(out['pred_logits'])
                    result['pred_boxes'].append(out['pred_boxes'])
                    targets_flat.extend(current_targets)

        result['pred_logits'] = torch.cat(result['pred_logits'], dim=0)
        result['pred_boxes'] = torch.cat(result['pred_boxes'], dim=0)
        return result, targets_flat


class DETRArTracking(DETRArTrackingBase, DETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRArTrackingBase.__init__(self, **tracking_kwargs)


class DeformableDETRArTracking(DETRArTrackingBase, DeformableDETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRArTrackingBase.__init__(self, **tracking_kwargs)
