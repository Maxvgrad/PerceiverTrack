import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import path as osp

import numpy as np
from nuimages import NuImages
from nuimages.utils.utils import name_to_index_mapping
from tqdm import tqdm

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

NAME_MAPPING = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument('--data-root', type=str, default='./data/nuimages',
                        help='specify the root path of dataset')
    parser.add_argument('--version', type=str, nargs='+', default=['v1.0-mini'],
                        required=False, help='specify the dataset version')
    parser.add_argument('--out-dir', type=str, default='./data/nuimages/annotations/',
                        required=False, help='path to save the exported json')
    parser.add_argument('--nproc', type=int, default=1, required=False,
                        help='workers to process semantic masks')
    parser.add_argument('--extra-tag', type=str, default='nuimages')
    parser.add_argument('--cameras', type=str, nargs='+',
                        default=[
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT',
                            'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'
                        ])
    args = parser.parse_args()
    return args


def mkdir_or_exist(directory):
    os.makedirs(directory, exist_ok=True)


def dump_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def get_previous_samples(nuim, sample_info, sample_image_id):
    previous_samples = []
    prev_token = sample_info['prev']

    # Traverse backward and collect previous sample data
    while prev_token:
        prev_sample_info = nuim.get('sample_data', prev_token)
        previous_samples.append(dict(
            id=len(previous_samples),
            file_name=prev_sample_info['filename'],
            sample_image_id=sample_image_id,
        ))
        prev_token = prev_sample_info['prev']

    return previous_samples


def get_img_annos(nuim, img_info, cat2id, out_dir, data_root, seg_root):
    """Get semantic segmentation map for an image."""
    sd_token = img_info['token']
    image_id = img_info['id']
    name_to_index = name_to_index_mapping(nuim.category)

    # Get image data.
    width, height = img_info['width'], img_info['height']
    semseg_mask = np.zeros((height, width)).astype('uint8')

    # Load object instances.
    object_anns = [o for o in nuim.object_ann if o['sample_data_token'] == sd_token]
    object_anns = sorted(object_anns, key=lambda k: k['token'])

    annotations = []
    for i, ann in enumerate(object_anns, start=1):
        category_token = ann['category_token']
        category_name = nuim.get('category', category_token)['name']

        if category_name in NAME_MAPPING:
            cat_name = NAME_MAPPING[category_name]
            cat_id = cat2id[cat_name]

            x_min, y_min, x_max, y_max = ann['bbox']

            data_anno = dict(
                image_id=image_id,
                category_id=cat_id,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                iscrowd=0)
            annotations.append(data_anno)

    img_filename = img_info['file_name']

    return annotations, 0


def export_nuim_to_coco(nuim, data_root, out_dir, extra_tag, version, nproc, cameras):
    print('Process category information')
    categories = [dict(id=nus_categories.index(cat_name), name=cat_name)
                  for cat_name in nus_categories]
    cat2id = {k_v['name']: k_v['id'] for k_v in categories}

    images = []
    past_images = []
    print('Process image meta information...')

    for sample in tqdm(nuim.sample):
        sample_info = nuim.get('sample_data', sample['key_camera_token'])

        if sample_info['is_key_frame'] and any(c for c in cameras if c in sample_info['filename'].split('/')):
            img_id = len(images)

            # Get the previous samples
            previous_samples = get_previous_samples(nuim, sample_info, img_id)

            # Insert the previous samples before the current sample
            for prev_sample in reversed(previous_samples):  # Add previous samples in reverse order
                prev_sample['id'] = len(past_images)
                prev_sample['sample_image_id'] = img_id
                past_images.append(prev_sample)

            # Insert the current sample
            images.append(
                dict(
                    id=img_id,
                    token=sample_info['token'],
                    sample_token=sample['token'],
                    file_name=sample_info['filename'],
                    width=sample_info['width'],
                    height=sample_info['height'],
                    num_previous_frames=len(previous_samples),
                )
            )

    def process_img_anno(img_info):
        return get_img_annos(nuim, img_info, cat2id, out_dir, data_root, None)

    print('Process img annotations...')
    if nproc > 1:
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            futures = {executor.submit(process_img_anno, img): img for img in images}
            outputs = [future.result() for future in tqdm(as_completed(futures), total=len(images))]
    else:
        outputs = [process_img_anno(img_info) for img_info in tqdm(images)]

    print('Process annotation information...')
    annotations = []
    max_cls_ids = []
    for single_img_annos, max_cls_id in outputs:
        max_cls_ids.append(max_cls_id)
        for img_anno in single_img_annos:
            img_anno.update(id=len(annotations))
            annotations.append(img_anno)

    max_cls_id = max(max_cls_ids)
    print(f'Max ID of class in the semantic map: {max_cls_id}')

    coco_format_json = dict(
        images=images, annotations=annotations, categories=categories, past_images=past_images)

    mkdir_or_exist(out_dir)
    out_file = osp.join(out_dir, f'{extra_tag}_{version}.json')
    print(f'Annotation dumped to {out_file}')
    dump_json(coco_format_json, out_file)


def main():
    args = parse_args()
    print(args)
    for version in args.version:
        print(f'Converting {version} dataset to COCO format...')
        nuim = NuImages(dataroot=args.data_root, version=version, verbose=True, lazy=True)
        export_nuim_to_coco(nuim, args.data_root, args.out_dir, args.extra_tag, version, args.nproc, args.cameras)
