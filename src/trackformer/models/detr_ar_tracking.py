from abc import abstractmethod

import torch
import torch.nn as nn
from torchvision.ops import nms

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
        self.detection_nms_thresh = 0.9

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        src, mask = samples.decompose()

        if len(src.shape) < 5:
            # samples without a time dimension
            raise NotImplementedError("Not implemented yet samples without a time dimension.")

        src = src.permute(1, 0, 2, 3, 4)  # change dimension order from BT___ to TB___

        device = src.device
        result = {'pred_logits': [], 'pred_boxes': []}
        targets_flat = []
        orig_size = torch.stack([t[-1]["orig_size"] for t in targets], dim=0).to(device)

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

            if self._disable_propagate_track_query_experiment:
                # Experiment: No track query propagation
                out_no_track_query_prop, targets_resp, features, memory, hs = super().forward(
                    samples=batch, targets=current_targets_base
                )

                out_no_track_query_prop = self.filter_output_result(
                    orig_size, out_no_track_query_prop, self.get_number_of_prev_track_queries(current_targets_base))

                if has_ground_truth:
                    # If we have a ground truth for this timestamp
                    self.populate_results(
                        device, current_targets_base, out_no_track_query_prop, result,
                        targets_flat, timestamp, 'no_track_query_propagation'
                    )

            current_targets_baseline = current_targets_base
            # Experiment: Baseline
            if out_baseline:
                # Populate a previous hidden state
                current_targets_baseline = self.populate_targets_with_query_hs_and_reference_boxes(
                    current_targets_base, out_baseline)

            out_baseline, targets_resp, features, memory, hs = super().forward(
                samples=batch, targets=current_targets_baseline
            )

            out_baseline = self.filter_output_result(
                orig_size, out_baseline, self.get_number_of_prev_track_queries(current_targets_baseline))

            if has_ground_truth:
                # If we have a ground truth for this timestamp
                self.populate_results(
                    device, current_targets_baseline, out_baseline, result, targets_flat, timestamp, 'baseline'
                )

            # Evaluation experiments
            if not self.training:

                # Experiment: blind
                if timestamp > 0:
                    current_targets_blind = current_targets_base

                    # Experiment where the model is supposed to receive input at every timestamp.
                    # We assume the input is not available, so we feed a zero image as input.
                    # This allows us to evaluate how well the model can predict a new object position without actual input.
                    zero_batch = torch.zeros_like(batch)
                    zero_samples = nested_tensor_from_tensor_list(zero_batch)
                    zero_mask = torch.ones(zero_samples.mask.shape, dtype=torch.bool, device=device)
                    zero_samples = NestedTensor(zero_samples.tensors, zero_mask)

                    if out_blind:
                        current_targets_blind = self.populate_targets_with_query_hs_and_reference_boxes(
                            current_targets_blind, out_blind)

                    out_blind, *_ = super().forward(
                        samples=zero_samples, targets=current_targets_blind
                    )

                    out_blind = self.filter_output_result(
                        orig_size, out_blind, self.get_number_of_prev_track_queries(current_targets_blind))

                    if has_ground_truth:
                        self.populate_results(
                            device, current_targets_blind, out_blind, result, targets_flat, timestamp,
                            'blind'
                        )

                    # Experiment: gap
                    if timestamp > 1 and has_ground_truth:
                        # Experiment to evaluate how well the model can predict an object's position after a time gap
                        # where the input was either skipped or replaced with zero images due to missing data.
                        out_gap, *_ = super().forward(
                            samples=batch, targets=current_targets_blind
                        )

                        out_gap = self.filter_output_result(
                            orig_size, out_gap, self.get_number_of_prev_track_queries(current_targets_blind))

                        self.populate_results(
                            device, current_targets_blind, out_gap, result, targets_flat, timestamp,
                            'gap'
                        )

        result = self.pad_and_stack_results(result)
        return result, targets_flat

    def get_number_of_prev_track_queries(self, current_targets):
        return torch.tensor(
            [t['num_track_queries_used'].item() if 'num_track_queries_used' in t else 0 for t in current_targets],
            dtype=torch.int64
        )

    def pad_and_stack_results(self, result):
        max_size = max([logit.shape[0] for logit in result['pred_logits']])  # Maximum number of queries

        padded_logits = []
        padded_boxes = []

        for logits, boxes in zip(result['pred_logits'], result['pred_boxes']):
            # for b in range(logits.shape[0]):  # Iterate over each batch
            # Get the number of elements (queries) for the current batch
            num_elements = logits.shape[0]

            # Pad if necessary to match max_size
            if num_elements < max_size:
                # Pad logits and boxes to match max_size
                pad_size = max_size - num_elements
                padding_logits = torch.zeros((pad_size, logits.shape[1]), device=logits.device)
                padding_boxes = torch.zeros((pad_size, boxes.shape[1]), device=boxes.device)

                # Concatenate original logits/boxes with padding
                logits_padded_batch = torch.cat([logits, padding_logits], dim=0)
                boxes_padded_batch = torch.cat([boxes, padding_boxes], dim=0)
            else:
                # If num_elements equals or exceeds max_size, truncate if needed (though unlikely)
                logits_padded_batch = logits[:max_size]
                boxes_padded_batch = boxes[:max_size]

            # Append the padded tensors to the result lists
            padded_logits.append(logits_padded_batch)
            padded_boxes.append(boxes_padded_batch)

        # Step 4: Stack the padded logits and boxes for all batches and time steps
        result['pred_logits'] = torch.stack(padded_logits, dim=0)
        result['pred_boxes'] = torch.stack(padded_boxes, dim=0)

        return result

    def populate_results(
            self, device, current_targets, output, result, targets_flat, timestamp, experiment
    ):
        current_timestamp_targets = [current_target.copy() for current_target in current_targets]

        for current_target in current_timestamp_targets:
            current_target['timestamp'] = torch.tensor(timestamp, device=device)
            current_target['experiment'] = experiment
        assert len(current_timestamp_targets) == len(output['pred_logits']) == len(output['pred_boxes'])
        result['pred_logits'].extend(output['pred_logits'])
        result['pred_boxes'].extend(output['pred_boxes'])
        targets_flat.extend(current_timestamp_targets)
        return current_timestamp_targets

    @abstractmethod
    def populate_targets_with_query_hs_and_reference_boxes(self, current_targets, hs_embeds):
        pass

    def filter_output_result(self, orig_size, output, numbers_previous_track_queries):
        post_process_results = self._obj_detector_post['bbox'](output, orig_size)
        filtered_output = {key: [] for key in output.keys() if key != 'aux_outputs'}
        print(f'numbers_previous_track_queries: {numbers_previous_track_queries}')
        for i, post_process_result in enumerate(post_process_results):
            number_previous_track_queries = numbers_previous_track_queries[i]
            previous_track_scores = post_process_result['scores'][:number_previous_track_queries]
            previous_track_labels = post_process_result['labels'][:number_previous_track_queries]

            print(f'Previous tracks: {number_previous_track_queries}')

            previous_track_keep = torch.logical_and(
                previous_track_scores > self._track_obj_score_threshold,
                previous_track_labels == self._label_person
            )

            previous_keep_count = previous_track_keep.sum().item()
            print(f'Previous track keep: {previous_keep_count} (-{number_previous_track_queries-previous_keep_count})')

            previous_track_boxes = output['pred_boxes'][i][:number_previous_track_queries][previous_track_keep]
            previous_track_pred_logits = output['pred_logits'][i][:number_previous_track_queries][previous_track_keep]
            previous_hs_embed = output['hs_embed'][i][:number_previous_track_queries][previous_track_keep]
            previous_track_scores = torch.zeros_like(previous_track_scores[previous_track_keep])
            previous_track_scores.fill_(float('inf'))
            assert torch.all(previous_track_scores == float('inf'))

            new_track_scores = post_process_result['scores'][-self.num_queries:]
            new_track_labels = post_process_result['labels'][-self.num_queries:]

            number_new_tracks = new_track_scores.shape[0]
            print(f'New tracks: {number_new_tracks}')

            new_track_keep = torch.logical_and(
                new_track_scores > self._track_obj_score_threshold,
                new_track_labels == self._label_person
            )

            new_keep_count = new_track_keep.sum().item()
            print(f'New track keep: {new_keep_count} (-{number_new_tracks-new_keep_count})')

            new_track_boxes = output['pred_boxes'][i][-self.num_queries:][new_track_keep]
            new_track_pred_logits = output['pred_logits'][i][-self.num_queries:][new_track_keep]
            new_hs_embed = output['hs_embed'][i][-self.num_queries:][new_track_keep]
            new_track_scores = new_track_scores[new_track_keep]

            print(f'previous_track_boxes: {previous_track_boxes.shape}')
            print(f'new_track_boxes: {new_track_boxes.shape}')

            track_boxes = torch.cat([previous_track_boxes, new_track_boxes])
            logits = torch.cat([previous_track_pred_logits, new_track_pred_logits])
            hs_embeds = torch.cat([previous_hs_embed, new_hs_embed])

            track_scores = torch.cat([previous_track_scores, new_track_scores])

            print(f'track_boxes: {track_boxes.shape}')
            print(f'track_scores: {track_scores.shape}')

            keep = nms(track_boxes, track_scores, self.detection_nms_thresh)

            print(f'keep: {keep.shape}')

            after_nms_count = keep.shape[0]

            # Print the number of filtered tracks after NMS
            print(f"Tracks after NMS: {after_nms_count} (-{previous_keep_count + new_keep_count - after_nms_count})")

            filtered_output['pred_boxes'].append(track_boxes[keep])
            filtered_output['pred_logits'].append(logits[keep])
            filtered_output['hs_embed'].append(hs_embeds[keep])

        return filtered_output


class DETRArTracking(DETRArTrackingBase, DETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRArTrackingBase.__init__(self, **tracking_kwargs)


class DeformableDETRArTracking(DETRArTrackingBase, DeformableDETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRArTrackingBase.__init__(self, **tracking_kwargs)
        self._label_person = 0

    def populate_targets_with_query_hs_and_reference_boxes(self, current_targets, output):
        # Copy the current targets
        current_targets = current_targets.copy()

        hs_embeds = output['hs_embed']
        pred_boxes = output['pred_boxes']

        # If there are embeddings present
        if len(hs_embeds) > 0:
            # Get the maximum length of track queries across all targets (i.e., across all hs_embeds)
            max_num_queries = max(hs_embed.shape[0] for hs_embed in hs_embeds)

            # Iterate over each target and pad `track_query_hs_embeds` with 0s and `track_query_boxes` with float('nan')
            for i, current_target in enumerate(current_targets):
                track_query_hs_embed = hs_embeds[i]  # Embeddings
                track_query_boxes = pred_boxes[i]  # Boxes

                num_track_queries_used = track_query_hs_embed.shape[0]
                current_target['num_track_queries_used'] = torch.tensor(num_track_queries_used, dtype=torch.float32)
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
        else:
            for i, current_target in enumerate(current_targets):
                current_target['num_track_queries_used'] = torch.tensor(0.0, dtype=torch.float32)

        return current_targets

class PerceiverArTracking(DETRArTrackingBase, PerceiverDetection):

    def __init__(self, tracking_kwargs, perceiver_kwargs):
        PerceiverDetection.__init__(self, **perceiver_kwargs)
        DETRArTrackingBase.__init__(self, **tracking_kwargs)
        self._label_person = 0

    def populate_targets_with_query_hs_and_reference_boxes(self, current_targets, output):
        # Copy the current targets
        current_targets = current_targets.copy()

        hs_embeds = output['hs_embed']

        # If there are embeddings present
        if len(hs_embeds) > 0:
            for i, current_target in enumerate(current_targets):
                track_query_hs_embed = hs_embeds[i]  # Embeddings

                # Add the padded values back to the target
                current_target['track_query_hs_embeds'] = track_query_hs_embed
                num_track_queries_used = track_query_hs_embed.shape[0]

                current_target['num_track_queries_used'] = torch.tensor(num_track_queries_used, dtype=torch.float32)
        else:
            for i, current_target in enumerate(current_targets):
                current_target['num_track_queries_used'] = torch.tensor(0.0, dtype=torch.float32)

        return current_targets