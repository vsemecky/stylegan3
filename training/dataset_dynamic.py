"""Streaming images and labels from datasets created with dataset_tool.py."""
import random
import numpy as np
import PIL.Image
import PIL.ImageOps
import PIL.ImageFile
from training.dataset import ImageFolderDataset


class DynamicDataset(ImageFolderDataset):

    def __init__(self, path, resolution=None, crop="center", autocontrast_probability=0, autocontrast_max_cutoff=0, use_labels=False, **super_kwargs):
        self._resolution = resolution
        self._crop = crop
        self._use_labels = use_labels
        self._size = (resolution, resolution)  # (width, height)
        self._autocontrast_probability = autocontrast_probability
        self._autocontrast_max_cutoff = autocontrast_max_cutoff

        if resolution is None:
            raise IOError('Resolution must be explicitly set when using Dynamic Dataset, e.g. --dd-res=1024')

        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

        super().__init__(path=path, resolution=resolution, **super_kwargs)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:

            image_pil = PIL.Image.open(f).convert('RGB')

            # Autocontrast
            if random.random() < self._autocontrast_probability:
                cutoff = random.uniform(0, self._autocontrast_max_cutoff)
                image_pil = PIL.ImageOps.autocontrast(image_pil, cutoff=cutoff, preserve_tone=True)

            # Calc centering
            if self._crop == "center":
                centering = (0.5, 0.5)  # Center crop
            else:
                centering = (random.uniform(0, 1), random.uniform(0, 1))  # Random crop

            image_pil = PIL.ImageOps.fit(
                image_pil,
                size=self._size,
                method=PIL.Image.LANCZOS,
                centering=centering
            )

            image_np = np.array(image_pil)

        image_np = image_np.transpose(2, 0, 1)  # HWC => CHW
        return image_np

    def _load_raw_labels(self):
        if not self._use_labels:
            return None

        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
