import csv
import os
from typing import Dict, Any
import torch
from torch import nn
import torch.nn.functional as F

from trackformer.util import box_ops


class ResultSaver:

    def __init__(self, file_name):
        self._file_name = file_name
        assert file_name.endswith('.csv'), "File must be a .csv file"
        self._file_exists = os.path.isfile(self._file_name)
        self._buffer = []

    def save(self, result: Dict[str, Any]):
        for experiment, experiment_results in result.items():
            for frame_number, image_results in experiment_results.items():
                for image_id, results in image_results.items():
                    for box, score, label in zip(
                            results['boxes'].tolist(),
                            results['scores'].tolist(),
                            results['labels'].tolist()):
                        self._buffer.append({
                            'experiment': experiment,
                            'frame_number': frame_number,
                            'image_id': image_id,
                            'box': box,
                            'score': score,
                            'label': label
                        })
                if len(self._buffer) >= 1000:
                    self._write_to_file()
        self._flush()


    def _write_to_file(self):
        """Write buffered data to file."""
        write_header = not self._file_exists

        with open(self._file_name, mode='a', newline='') as file:
            fieldnames = ['experiment', 'frame_number', 'image_id', 'box', 'score', 'label']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()
                self._file_exists = True

            writer.writerows(self._buffer)
            self._buffer.clear()  # Clear buffer after writing

    def _flush(self):
        """Manually flush remaining buffered data to file."""
        if self._buffer:
            self._write_to_file()

class PostProcessResultSave(nn.Module):
    """ This module converts the model's output into the format expected by the ResultSaver"""

    def process_boxes(self, boxes, target_sizes):
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return boxes

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of
                          each images of the batch For evaluation, this must be the
                          original image size (before any data augmentation) For
                          visualization, this should be the image size after data
                          augment, but before padding
        """
        target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = self.process_boxes(out_bbox, target_sizes)

        results_orig = [
            {'scores': s, 'labels': l, 'boxes': b}
            for s, l, b in zip(scores, labels, boxes)]

        experiment_results = self.partition_by_experiment_and_timestamp(results_orig, targets)

        return experiment_results

    def partition_by_experiment_and_timestamp(self, results_orig, targets):
        result = {}
        for target, output in zip(targets, results_orig):
            experiment = target['experiment']

            if experiment not in result:
                result[experiment] = {}

            experiment_result_dict = result[experiment]
            timestamp = target['timestamp'].item()
            image_id = target['image_id'].item()

            if timestamp in experiment_result_dict:
                if image_id in experiment_result_dict[timestamp]:
                    print(
                        f'Warn overriding results for experiment {experiment} '
                        f'timestamp {timestamp} image id {image_id}'
                    )
                experiment_result_dict[timestamp][image_id] = output
            else:
                experiment_result_dict[timestamp] = {
                    image_id: output
                }
        return result