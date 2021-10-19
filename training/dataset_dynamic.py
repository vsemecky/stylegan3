"""Streaming images and labels from datasets created with dataset_tool.py."""
import random
import numpy as np
import PIL.Image
import PIL.ImageOps
from training.dataset import ImageFolderDataset


class DynamicDataset(ImageFolderDataset):

    def __init__(self, path, resolution=None, crop="center", **super_kwargs):
        self._resolution = resolution
        self._crop = crop
        self._size = (resolution, resolution)

        if resolution is None:
            raise IOError('Resolution must be explicitly set when using Dynamic Dataset, e.g. --dd-res=1024')

        super().__init__(path=path, resolution=resolution, **super_kwargs)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._crop == "center":
                centering = (0.5, 0.5)  # Center crop
            else:
                centering = (random.uniform(0, 1), random.uniform(0, 1))  # Random crop

            image_pil = PIL.Image.open(f).convert('RGB')
            image_pil = PIL.ImageOps.fit(
                image_pil,
                size=self._size,
                method=PIL.Image.LANCZOS,
                centering=centering
            )

            image_np = np.array(image_pil)

        image_np = image_np.transpose(2, 0, 1)  # HWC => CHW
        return image_np
