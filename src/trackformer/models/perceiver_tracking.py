from collections import deque

import torch
from torch import Tensor

from trackformer.models.perceiver_detection import PerceiverDetection
from trackformer.util.misc import NestedTensor


class PerceiverTracking(PerceiverDetection):
    def __init__(self, max_num_of_frames_lookback, tracking_kwargs, detection_model_kwargs):
        PerceiverDetection.__init__(self, **detection_model_kwargs)
        self._max_num_of_frames_lookback = max_num_of_frames_lookback

    def forward(self, samples: NestedTensor, targets: list = None, latents: Tensor = None):
        src, mask = samples.decompose()

        if len(src.shape) < 5:
            # samples without a time dimension
            return super().forward(samples, targets)

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

            out, *_ = super().forward(
                samples=batch, latents=latents
            )
            latents = out['hs_embed']
            latent_deque.appendleft(latents)

            if 'boxes' in current_targets[0]:
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
