"""Streaming images and labels from datasets created with dataset_tool.py."""

import numpy as np
import PIL.Image
import PIL.ImageOps
from training.dataset import ImageFolderDataset


class DynamicDataset(ImageFolderDataset):

    resolution = None
    crop = "center"

    def __init__(self, path, resolution=None, **super_kwargs):
        print("DynamicDataset.__init__:", locals())

        if resolution is None:
            raise IOError('Resolution must be set (e.g. 1024, 512 or 256')

        self.resolution = resolution
        super().__init__(path=path, resolution=resolution, **super_kwargs)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            image_pil = PIL.Image.open(f).convert('RGB')

            # Center crop
            image_pil = PIL.ImageOps.fit(
                image_pil,
                size=(self.resolution, self.resolution),
                method=PIL.Image.LANCZOS,
                centering=(0.5, 0.5)
            )

            image = np.array(image_pil)
            print("Dataset", raw_idx, image_pil.width, image_pil.height, image.ndim)

        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC

        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image
