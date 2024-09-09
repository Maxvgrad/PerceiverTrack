import os
import random
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from PIL import Image

from trackformer.datasets.coco import make_coco_transforms, ConvertCocoPolysToMask


class NuImagesDetection(torchvision.datasets.CocoDetection):

    def __init__(
            self,
            img_folder,
            ann_file,
            transforms,
            norm_transforms,
            debug=False,
    ):
        super(NuImagesDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self._norm_transforms = norm_transforms
        self.prepare = ConvertCocoPolysToMask()
        self._debug = debug

        self.img_to_past = {img['id']: [] for img in self.coco.imgs.values()}

        if 'past_images' in self.coco.dataset:
            for past_img in self.coco.dataset['past_images']:
                self.img_to_past[past_img['sample_image_id']].append(past_img)
        num_ids_before = len(self.ids)
        self.ids = [i for i in self.ids if len(self.img_to_past[i]) == 6]
        print(f'Number of images filtered: {num_ids_before - len(self.ids)}')

    def _getitem_by_id(self, idx, random_state=None):

        if random_state is not None:
            curr_random_state = {
                'random': random.getstate(),
                'torch': torch.random.get_rng_state()}
            random.setstate(random_state['random'])
            torch.random.set_rng_state(random_state['torch'])

        targets = []

        img, target = super(NuImagesDetection, self).__getitem__(idx)
        img_id = self.ids[idx]
        imgs = []

        target = {'image_id': img_id, 'annotations': target}
        img, target = self.prepare(img, target)

        for past_img_info in self.img_to_past[img_id]:
            past_img_path = os.path.join(self.root, past_img_info['file_name'])
            past_img = Image.open(past_img_path).convert("RGB")
            past_img_target = {'keep_frame': torch.tensor([1], dtype=torch.int64)}

            if self._transforms is not None:
                past_img, past_img_target = self._transforms(past_img, past_img_target)  # No targets for past frames

            if random_state is not None:
                random.setstate(curr_random_state['random'])
                torch.random.set_rng_state(curr_random_state['torch'])

            past_img, past_img_target = self._norm_transforms(past_img, past_img_target)

            imgs.append(past_img)
            targets.append(past_img_target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        img, target = self._norm_transforms(img, target)

        imgs.append(img)
        targets.append(target)

        return imgs, targets

    def __getitem__(self, idx):
        random_state = {
            'random': random.getstate(),
            'torch': torch.random.get_rng_state()}
        imgs, targets = self._getitem_by_id(idx, random_state)

        return imgs, targets


def build(image_set, args):
    root = Path(args.nuimages_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    split = getattr(args, f"{image_set}_split")

    img_folder = root
    ann_file = root / f'annotations/nuimages_v1.0-{split}.json'

    transforms, norm_transforms = make_coco_transforms(image_set)

    dataset = NuImagesDetection(
        root,
        ann_file,
        transforms,
        norm_transforms,
        args.debug
    )

    return dataset
