from abc import abstractmethod

import torch
import torch.nn as nn

from .deformable_detr import DeformableDETR
from .detr import DETR
from .perceiver_detection import PerceiverDetection
from ..util.misc import NestedTensor, nested_tensor_from_tensor_list


class DETRArTrackingBase(nn.Module):

    def __init__(self,
                 obj_detector_post,
                 track_obj_score_threshold: float = 0.4,
                 max_num_of_frames_lookback: int = 0,
                 disable_propagate_track_query_experiment: bool = False,
                 **kwargs
                 ):
        self._obj_detector_post = obj_detector_post
        self._track_obj_score_threshold = track_obj_score_threshold
        self._max_num_of_frames_lookback = max_num_of_frames_lookback
        self._debug = False
        self._disable_propagate_track_query_experiment = disable_propagate_track_query_experiment

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        src, mask = samples.decompose()

        if len(src.shape) < 5:
            # samples without a time dimension
            raise NotImplementedError("Not implemented yet samples without a time dimension.")

        src = src.permute(1, 0, 2, 3, 4)  # change dimension order from BT___ to TB___

        result = {'pred_logits': [], 'pred_boxes': []}
        targets_flat = []
        orig_size = torch.stack([t[-1]["orig_size"] for t in targets], dim=0).to(src.device)

        out_baseline = None
        out_blind = None

        for timestamp, batch in enumerate(src):
            current_targets_base = [target_list[timestamp] for target_list in targets]
            has_ground_truth = 'boxes' in current_targets_base[0]

            if self.training:
                frame_keep_mask = [t['keep_frame'] for t in current_targets_base]

                batch = nested_tensor_from_tensor_list(batch)

                for keep_frame, img, m in zip(frame_keep_mask, batch.tensors, batch.mask):
                    if not keep_frame:  # If the frame should be dropped
                        img[:, :, :] = 0  # Zero out the entire image
                        m[:, :] = True  # Set the mask to True, marking the frame as dropped

                batch = NestedTensor(batch.tensors, batch.mask)

            if self._disable_propagate_track_query_experiment:
                # Experiment: No track query propagation
                out_no_track_query_prop, targets_resp, features, memory, hs = super().forward(
                    samples=batch, targets=current_targets_base
                )

                if has_ground_truth:
                    # If we have a ground truth for this timestamp
                    self.populate_results(
                        batch.device, current_targets_base, out_no_track_query_prop, result,
                        targets_flat, timestamp, 'no_track_query_propagation'
                    )

            current_targets_baseline = current_targets_base
            # Experiment: Baseline
            if out_baseline:
                # Populate a previous hidden state
                hs_embeds_prev, num_track_queries_reused_prev = self.filter_hs_embeds(orig_size, out_baseline)
                current_targets_baseline = self.populate_targets_with_query_hs_and_reference_boxes(
                    current_targets_base, hs_embeds_prev, num_track_queries_reused_prev)

            out_baseline, targets_resp, features, memory, hs = super().forward(
                samples=batch, targets=current_targets_baseline
            )

            if has_ground_truth:
                # If we have a ground truth for this timestamp
                self.populate_results(
                    batch.device, current_targets_baseline, out_baseline, result, targets_flat, timestamp, 'baseline'
                )

            # Evaluation experiments
            if not self.training:

                # Experiment: blind
                if timestamp > 0:
                    current_targets_blind = current_targets_base
                    if out_blind:
                        hs_embeds, num_track_queries_reused = self.filter_hs_embeds(orig_size, out_blind)
                        current_targets_blind = self.populate_targets_with_query_hs_and_reference_boxes(
                            current_targets_blind, hs_embeds, num_track_queries_reused)

                    # Experiment where the model is supposed to receive input at every timestamp.
                    # We assume the input is not available, so we feed a zero image as input.
                    # This allows us to evaluate how well the model can predict a new object position without actual input.
                    zero_batch = torch.zeros_like(batch)
                    zero_samples = nested_tensor_from_tensor_list(zero_batch)
                    zero_mask = torch.ones(zero_samples.mask.shape, dtype=torch.bool, device=batch.device)
                    zero_samples = NestedTensor(zero_samples.tensors, zero_mask)

                    out_blind, *_ = super().forward(
                        samples=zero_samples, targets=current_targets_blind
                    )

                    if has_ground_truth:
                        self.populate_results(
                            batch.device, current_targets_blind, out_blind, result, targets_flat, timestamp,
                            'blind'
                        )

                    # Experiment: gap
                    if timestamp > 1 and has_ground_truth:
                        # Experiment to evaluate how well the model can predict an object's position after a time gap
                        # where the input was either skipped or replaced with zero images due to missing data.
                        out_gap, *_ = super().forward(
                            samples=batch, targets=current_targets_blind
                        )

                        self.populate_results(
                            batch.device, current_targets_blind, out_gap, result, targets_flat, timestamp,
                            'gap'
                        )

        # Prepare data for stacking
        min_size = min([logit.shape[1] for logit in result['pred_logits']])  # Minimum number of queries

        filtered_logits = []
        filtered_boxes = []

        for logits, boxes in zip(result['pred_logits'], result['pred_boxes']):  # Iteration over time
            post_process_results = self._obj_detector_post['bbox'](
                {'pred_boxes': boxes, 'pred_logits': logits},
                orig_size
            )

            labels = torch.stack([post_process_result['labels'] for post_process_result in post_process_results]) # Corresponding labels
            scores = torch.stack([post_process_result['scores'] for post_process_result in post_process_results])

            remove_mask = (labels != 0) | (scores < self._track_obj_score_threshold)

            for b in range(logits.shape[0]):  # Iterate over each batch

                # Get the number of elements for the current batch
                num_elements = logits[b].shape[0]
                # Get the indices for the current batch
                batch_remove_mask = remove_mask[b]
                batch_remove_indices = torch.where(batch_remove_mask)[0]  # Indices where label != 0 for this batch

                if num_elements > min_size:
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

        result['pred_logits'] = torch.stack(filtered_logits, dim=0)
        result['pred_boxes'] = torch.stack(filtered_boxes, dim=0)
        return result, targets_flat

    def populate_results(
            self, device, current_targets, out_baseline, result, targets_flat, timestamp, experiment
    ):
        current_timestamp_targets = [current_target.copy() for current_target in current_targets]
        self.populate_timestamp_and_experiment_information(device, current_timestamp_targets, timestamp, experiment)
        result['pred_logits'].append(out_baseline['pred_logits'])
        result['pred_boxes'].append(out_baseline['pred_boxes'])
        targets_flat.extend(current_timestamp_targets)
        return current_timestamp_targets

    def populate_timestamp_and_experiment_information(self, device, current_timestamp_targets, timestamp, experiment):
        for current_target in current_timestamp_targets:
            current_target['timestamp'] = torch.tensor(timestamp, device=device)
            current_target['experiment'] = experiment
            if experiment == 'blind':
                current_target['number_of_consecutive_zero_frame'] = torch.tensor(timestamp, device=device)
            if experiment == 'gap':
                current_target['number_of_consecutive_gap_frame_followed_by_image'] = torch.tensor(timestamp, device=device)

    @abstractmethod
    def populate_targets_with_query_hs_and_reference_boxes(self, current_targets, hs_embeds, num_track_queries_reused):
        pass

    def filter_hs_embeds(self, orig_size, out):
        post_process_results = self._obj_detector_post['bbox'](out, orig_size)
        hs_embeds = []
        num_track_queries_reused = []
        for i, post_process_result in enumerate(post_process_results):
            track_scores = post_process_result['scores']

            track_keep = torch.logical_and(
                track_scores > self._track_obj_score_threshold,
                post_process_result['labels'][:] == self._label_person
            )

            track_scored_prev = track_scores[:-self.num_queries]
            num_track_scored_prev_true = torch.sum(track_scored_prev > 0)
            num_track_queries_reused.append(num_track_scored_prev_true)
            hs_embeds.append(
                (out['hs_embed'][i][track_keep], post_process_result['boxes'][track_keep])
            )
        return hs_embeds, num_track_queries_reused


class DETRArTracking(DETRArTrackingBase, DETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRArTrackingBase.__init__(self, **tracking_kwargs)


class DeformableDETRArTracking(DETRArTrackingBase, DeformableDETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRArTrackingBase.__init__(self, **tracking_kwargs)
        self._label_person = 0

    def populate_targets_with_query_hs_and_reference_boxes(self, current_targets, hs_embeds, num_track_queries_reused):
        # Copy the current targets
        current_targets = current_targets.copy()

        # If there are embeddings present
        if len(hs_embeds) > 0:
            # Get the maximum length of track queries across all targets (i.e., across all hs_embeds)
            max_num_queries = max(hs_embed[0].shape[0] for hs_embed in hs_embeds)

            # Iterate over each target and pad `track_query_hs_embeds` with 0s and `track_query_boxes` with float('nan')
            for i, current_target in enumerate(current_targets):
                track_query_hs_embed = hs_embeds[i][0]  # Embeddings
                track_query_boxes = hs_embeds[i][1]  # Boxes

                # Pad `track_query_hs_embeds` with zeros to match max_num_queries
                if track_query_hs_embed.shape[0] < max_num_queries:
                    padding_size = max_num_queries - track_query_hs_embed.shape[0]
                    padded_hs_embed = torch.cat([track_query_hs_embed,
                                                 torch.zeros((padding_size, track_query_hs_embed.shape[1]),
                                                             dtype=track_query_hs_embed.dtype,
                                                             device=track_query_hs_embed.device)], dim=0)
                else:
                    padded_hs_embed = track_query_hs_embed

                # Pad `track_query_boxes` with float('nan') to match max_num_queries
                if track_query_boxes.shape[0] < max_num_queries:
                    padding_size = max_num_queries - track_query_boxes.shape[0]
                    nan_padding = torch.full((padding_size, track_query_boxes.shape[1]), float(0),
                                             dtype=track_query_boxes.dtype, device=track_query_boxes.device)
                    padded_track_query_boxes = torch.cat([track_query_boxes, nan_padding], dim=0)
                else:
                    padded_track_query_boxes = track_query_boxes

                # Add the padded values back to the target
                current_target['track_query_hs_embeds'] = padded_hs_embed
                current_target['track_query_boxes'] = padded_track_query_boxes

                # Handle `num_prev_track_queries_used`
                current_target['num_prev_track_queries_used'] = (
                    torch.tensor(0.0, dtype=torch.float32) if len(num_track_queries_reused) == 0
                    else num_track_queries_reused[i]
                )

        return current_targets

class PerceiverArTracking(DETRArTrackingBase, PerceiverDetection):

    def __init__(self, tracking_kwargs, perceiver_kwargs):
        PerceiverDetection.__init__(self, **perceiver_kwargs)
        DETRArTrackingBase.__init__(self, **tracking_kwargs)
        self._label_person = 1

    def populate_targets_with_query_hs_and_reference_boxes(self, current_targets, hs_embeds, num_track_queries_reused):
        # Copy the current targets
        current_targets = current_targets.copy()

        # If there are embeddings present
        if len(hs_embeds) > 0:
            for i, current_target in enumerate(current_targets):
                track_query_hs_embed = hs_embeds[i][0]  # Embeddings

                # Add the padded values back to the target
                current_target['track_query_hs_embeds'] = track_query_hs_embed
                # current_target['track_query_boxes'] = track_query_boxes

                # Handle `num_prev_track_queries_used`
                current_target['num_prev_track_queries_used'] = (
                    torch.tensor(0.0, dtype=torch.float32) if len(num_track_queries_reused) == 0
                    else num_track_queries_reused[i]
                )

        return current_targets