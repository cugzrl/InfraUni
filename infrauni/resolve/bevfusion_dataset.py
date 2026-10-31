"""Four exact RESOLVE camera pairs for the vendored BEVFusion baseline."""
from pathlib import Path
import numpy as np
from PIL import Image
from infrauni.resolve.dataset import ResolveSequenceDataset


class ResolveBEVFusionDataset(ResolveSequenceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not (self.root_path / '_READY').is_file():
            raise RuntimeError('The canonical multimodal dataset is not ready')
        if not self.infos:
            raise RuntimeError('Empty RESOLVE split')
        tokens = [row['token'] for row in self.infos]
        if len(set(tokens)) != len(tokens):
            raise ValueError('Duplicate frame tokens')
        for row in self.infos:
            if len(row['images']) != 4 or not np.asarray(row['camera_valid']).all():
                raise ValueError('Four exact camera pairs are required: ' + row['token'])

    def load_camera_info(self, input_dict, info):
        images, transforms = [], []
        target_h, target_w = self.camera_image_config.FINAL_DIM
        for path in info['images']:
            with Image.open(path) as source:
                image = source.convert('RGB')
            width, height = image.size
            # Keep the complete roadside field of view, with a small training crop.
            scale = max(target_w / width, target_h / height)
            if self.training:
                scale *= np.random.uniform(1., 1.08)
            new_w, new_h = max(target_w, round(width * scale)), max(target_h, round(height * scale))
            left = int(np.random.randint(new_w - target_w + 1)) if self.training else (new_w-target_w)//2
            top = int(np.random.randint(new_h - target_h + 1)) if self.training else (new_h-target_h)//2
            images.append(image.resize((new_w, new_h), Image.BILINEAR).crop((left, top, left+target_w, top+target_h)))
            transform = np.eye(4, dtype=np.float32)
            transform[0, 0], transform[1, 1] = new_w/width, new_h/height
            transform[0, 3], transform[1, 3] = -left, -top
            transforms.append(transform)
        extrinsics = np.asarray(info['lidar2cam'], dtype=np.float32)
        intrinsics = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
        intrinsics[:, :3, :3] = np.asarray(info['K'], dtype=np.float32)
        projections = intrinsics @ extrinsics
        if not np.allclose(projections, info['projection_mat'], atol=.02, rtol=1e-5):
            raise ValueError('Inconsistent camera projection: ' + info['token'])
        input_dict.update(camera_imgs=images, img_aug_matrix=np.asarray(transforms),
                          camera_intrinsics=intrinsics, camera2lidar=np.linalg.inv(extrinsics).astype(np.float32),
                          lidar2camera=extrinsics, lidar2image=projections)
        return input_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if {r['metadata']['token'] for r in det_annos} != {r['token'] for r in self.infos} or len(det_annos) != len(self.infos):
            raise ValueError('Evaluation must include every canonical frame, including empty GT')
        return super().evaluation(det_annos, class_names, **kwargs)
