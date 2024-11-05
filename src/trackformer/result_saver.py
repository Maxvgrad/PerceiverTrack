import csv
import os
from typing import Dict, Any
import torch
from torch import nn
import torch.nn.functional as F

from trackformer.util import box_ops


class ResultSaver:

    def __init__(self, file_name, increment_class_label=False):
        self._file_name = file_name
        assert file_name.endswith('.csv'), "File must be a .csv file"
        self._file_exists = os.path.isfile(self._file_name)
        self._buffer = []
        self._increment_class_label = increment_class_label
        print(f'Result saver increments label {self._increment_class_label}')

    def save(self, result: Dict[str, Any]):
        for experiment, experiment_results in result.items():
            for frame_number, image_results in experiment_results.items():
                for image_id, results in image_results.items():
                    for box, score, label in zip(
                            results['boxes'].tolist(),
                            results['scores'].tolist(),
                            results['labels'].tolist()):

                        if self._increment_class_label:
                            label += 1

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

    def __init__(self, bbox_postprocessor):
        super().__init__()
        self._bbox_postprocessor = bbox_postprocessor

    @torch.no_grad()
    def forward(self, outputs, targets):
        target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_orig = self._bbox_postprocessor(outputs, target_sizes)
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