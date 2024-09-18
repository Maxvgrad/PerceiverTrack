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
                 max_num_of_frames_lookback: int = 0,
                 **kwargs
                 ):
        self._obj_detector_post = obj_detector_post
        self._track_obj_score_threshold = track_obj_score_threshold
        self._max_num_of_frames_lookback = max_num_of_frames_lookback
        self._debug = False

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
        output_deque = deque(maxlen=max_num_of_frames_lookback + 1)
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

            current_targets = self.populate_targets_with_query_hs_and_reference_boxes(current_targets, hs_embeds)

            out, targets_resp, features, memory, hs = super().forward(
                samples=batch, targets=current_targets
            )

            hs_embeds = self.filter_hs_embeds(orig_size, out)

            output_deque.appendleft(out)

            if 'boxes' in current_targets[0] or self._debug:
                # frame has annotations then include it in output
                result['pred_logits'].append(out['pred_logits'])
                result['pred_boxes'].append(out['pred_boxes'])
                targets_flat.extend(current_targets)

            for num_frames_lookback in range(1, 1 + max_num_of_frames_lookback):
                if num_frames_lookback == len(output_deque):
                    # In the beginning of the sequence there's not enough latents for lookback
                    break

                hs_embeds = self.filter_hs_embeds(orig_size, output_deque[num_frames_lookback])
                current_targets = self.populate_targets_with_query_hs_and_reference_boxes(current_targets, hs_embeds)

                zero_batch = torch.zeros_like(batch)

                out, *_ = super().forward(
                    samples=zero_batch, targets=current_targets
                )

                output_deque[num_frames_lookback] = out

                if 'boxes' in current_targets[0]:
                    current_targets = current_targets.copy()
                    current_targets = [current_target.copy() for current_target in current_targets]
                    for current_target in current_targets:
                        current_target['consecutive_frame_skip_number'] = torch.tensor(num_frames_lookback, device=batch.device)

                    result['pred_logits'].append(out['pred_logits'])
                    result['pred_boxes'].append(out['pred_boxes'])
                    targets_flat.extend(current_targets)

        min_size = min([logit.shape[0] for logit in result['pred_logits']])

        filtered_logits = []
        filtered_boxes = []

        for logits, boxes in zip(result['pred_logits'], result['pred_boxes']):
            post_process_results = self._obj_detector_post['bbox'](
                {'pred_boxes': boxes, 'pred_logits': logits},
                orig_size
            )

            labels = torch.stack([post_process_result['labels'] for post_process_result in post_process_results]) # Corresponding labels

            # Step 1: Create a mask for elements where label != 0
            remove_mask = labels != 0

            for b in range(logits.shape[0]):  # Iterate over each batch

                # Get the number of elements for the current batch
                num_elements = logits[b].shape[0]

                # Get the indices for the current batch
                batch_remove_mask = remove_mask[b]
                batch_remove_indices = torch.where(batch_remove_mask)[0]  # Indices where label != 0 for this batch

                # If the number of elements to remove exceeds (num_elements - min_size), keep only up to the threshold
                if batch_remove_indices.numel() > (num_elements - min_size):
                    # Limit the number of indices to remove
                    excess_elements_to_remove = batch_remove_indices[-(num_elements - min_size):]  # Keep only the excess
                    batch_final_mask = torch.ones(num_elements, dtype=torch.bool,
                                                  device=logits.device)  # Start with all True
                    batch_final_mask[excess_elements_to_remove] = False  # Set False for excess elements with label != 0

                    # Filter the logits and boxes based on the mask
                    logits_filtered_batch = logits[b][batch_final_mask]
                    boxes_filtered_batch = boxes[b][batch_final_mask]
                else:
                    # If no excess, retain all elements
                    logits_filtered_batch = logits[b]
                    boxes_filtered_batch = boxes[b]
                filtered_logits.append(logits_filtered_batch)
                filtered_boxes.append(boxes_filtered_batch)

        result['pred_logits'] = torch.cat(filtered_logits, dim=0)
        result['pred_boxes'] = torch.cat(filtered_boxes, dim=0)
        return result, targets_flat

    def populate_targets_with_query_hs_and_reference_boxes(self, current_targets, hs_embeds):
        current_targets = current_targets.copy()
        if len(hs_embeds) > 0:
            for i, current_target in enumerate(current_targets):
                current_target['track_query_hs_embeds'] = hs_embeds[i][0]
                current_target['track_query_boxes'] = hs_embeds[i][1]

        return current_targets

    def filter_hs_embeds(self, orig_size, out):
        post_process_results = self._obj_detector_post['bbox'](out, orig_size)
        hs_embeds = []
        for i, post_process_result in enumerate(post_process_results):
            track_scores = post_process_result['scores']

            track_keep = torch.logical_and(
                track_scores > self._track_obj_score_threshold,
                post_process_result['labels'][:] == 0
            )

            hs_embeds.append(
                (out['hs_embed'][i][track_keep], post_process_result['boxes'][track_keep])
            )
        return hs_embeds


class DETRArTracking(DETRArTrackingBase, DETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRArTrackingBase.__init__(self, **tracking_kwargs)


class DeformableDETRArTracking(DETRArTrackingBase, DeformableDETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRArTrackingBase.__init__(self, **tracking_kwargs)
