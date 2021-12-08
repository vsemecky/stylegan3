"""Alternative dataset"""
import pathlib
import random
import re
import numpy as np
import PIL.Image
import PIL.ImageOps
import PIL.ImageFile
from training.dataset import ImageFolderDataset


class DynamicDataset(ImageFolderDataset):

    def __init__(self, path, resolution, extend, crop="center", scale=0.8, autocontrast_probability=0, autocontrast_max_cutoff=0, use_labels=False, **super_kwargs):
        self._width, self._height = self.decode_resolution(resolution)
        self._ratio = self._width / self._height
        self._crop = crop
        self._scale = scale  # Scale factor defines, how much we can zoom in (or cut into the image). 1 = no zoooming. e.g. 0.8 = up to 20% of the image is randomly cropped
        self._use_labels = use_labels
        self._size = (self._width, self._height)
        self._autocontrast_probability = autocontrast_probability
        self._autocontrast_max_cutoff = autocontrast_max_cutoff
        self._extend = extend
        if extend:
            self._extend_width, self._extend_height = self.decode_resolution(extend)

        if resolution is None:
            raise IOError('Resolution must be explicitly set when using Dynamic Dataset, e.g. --dd-res=1024')

        super().__init__(path=path, resolution=self._width, use_labels=use_labels, **super_kwargs)

    def _load_raw_image(self, raw_idx):
        # Load image
        fname = self._image_fnames[raw_idx]
        try:
            with self._open_file(fname) as f:
                image_pil = PIL.Image.open(f).convert('RGB')
        except Exception as e:
            # Problem with loading image? Use another randomly selected image instead
            new_idx = random.randrange(0, len(self._all_fnames))
            print("Bad image:", f"#{raw_idx} {fname} - using #{new_idx}")
            return self._load_raw_image(new_idx)

        # Autocontrast
        if random.random() < self._autocontrast_probability:
            cutoff = random.uniform(0, self._autocontrast_max_cutoff)
            # image_pil = PIL.ImageOps.autocontrast(image_pil, cutoff=cutoff, preserve_tone=True)
            image_pil = self.autocontrast(image_pil, cutoff=cutoff)

        # Crops
        if self._crop == "center":
            centering = (0.5, 0.5)  # Center crop
            image_pil = PIL.ImageOps.fit(
                image_pil,
                size=self._size,
                method=PIL.Image.LANCZOS,
                centering=centering
            )
        elif self._crop == "old_random":  # todo Odstranit
            centering = (random.uniform(0, 1), random.uniform(0, 1))  # Random crop
            image_pil = PIL.ImageOps.fit(
                image_pil,
                size=self._size,
                method=PIL.Image.LANCZOS,
                centering=centering
            )
        else:
            # image_pil = self.random_zoom_crop(image=image_pil)
            if image_pil.width <= self._width or image_pil.height <= self._height:
                # If image too small to zoom-in, do simple random crop without zooming
                # @todo Místo jiného zpracování upscalovat aby i šířka i výška splňovali minimum
                image_pil = PIL.ImageOps.fit(
                    image_pil,
                    size=self._size,
                    method=PIL.Image.LANCZOS,
                    centering=(random.uniform(0, 1), random.uniform(0, 1))
                )
            else:
                image_pil = self.random_zoom_crop(image=image_pil)

        if self._extend:
            extended_pil = PIL.Image.new(mode="RGB", size=(self._extend_width, self._extend_height), color=(0, 0, 0))
            offset = ((self._extend_width - image_pil.width) // 2, (self._extend_height - image_pil.height) // 2)
            extended_pil.paste(image_pil, offset)
            image_pil = extended_pil

        image_np = np.array(image_pil).transpose(2, 0, 1)  # HWC => CHW
        return image_np

    def _load_raw_labels(self):
        if not self._use_labels:
            return None

        labels = {}
        for fname in self._all_fnames:
            p = pathlib.Path(fname)
            try:
                label = int(p.parts[0])
                labels[fname] = label
            except:
                print(f"Invalid label '{p.parts[0]}' in file path '{fname}'")
                exit()

        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def random_zoom_crop(self, image: PIL.Image):
        max_window_width, max_window_height = self.get_max_window_size(image)
        scale = random.uniform(max(self._width / max_window_width, self._scale), 1)  # Random scale
        window_width = round(scale * max_window_width)
        window_height = round(scale * max_window_height)
        left = random.randint(0, image.width - window_width)
        top = random.randint(0, image.height - window_height)
        crop_box = (left, top, left + window_width, top + window_height)
        image_cropped = image.resize(size=self._size, resample=PIL.Image.LANCZOS, box=crop_box)

        return image_cropped

    def get_max_window_size(self, image: PIL.Image):
        """ Returns maximum size (width, height) of inscribed rectangle with specific aspect ratio """
        width, height = image.size
        current_ratio = width / height
        if self._ratio == current_ratio:
            return width, height
        elif self._ratio > current_ratio:
            return width, round(width / self._ratio)
        elif self._ratio < current_ratio:
            return round(height * self._ratio), height

    @staticmethod
    def decode_resolution(resolution: str):
        """ Converts string like '1024x1024' to (width, height) """
        result = re.match("(\d+)\D*(\d+)?", resolution)
        if result:
            dd_width, dd_height = result.groups()
            if dd_height is None:
                dd_height = dd_width
            return int(dd_width), int(dd_height)
        else:
            raise ValueError(f"resolution should be in format '1024x1024' or '1024'")

    @staticmethod
    def autocontrast(image, cutoff=0, ignore=None, mask=None):
        """
        Photoshop-like autocontrast.

        This is taken from PIL 8.2.0 which is curently not supported in Colab. Once Colab upgrades to Python 3.8+,
        we will start using PIL.Imageops.autocontrast(..., preserve_tone=True) instead and this method will be removed.
        """
        histogram = image.convert("L").histogram(mask)  # Always preserve tones

        lut = []
        for layer in range(0, len(histogram), 256):
            h = histogram[layer : layer + 256]
            if ignore is not None:
                # get rid of outliers
                try:
                    h[ignore] = 0
                except TypeError:
                    # assume sequence
                    for ix in ignore:
                        h[ix] = 0
            if cutoff:
                # cut off pixels from both ends of the histogram
                if not isinstance(cutoff, tuple):
                    cutoff = (cutoff, cutoff)
                # get number of pixels
                n = 0
                for ix in range(256):
                    n = n + h[ix]
                # remove cutoff% pixels from the low end
                cut = n * cutoff[0] // 100
                for lo in range(256):
                    if cut > h[lo]:
                        cut = cut - h[lo]
                        h[lo] = 0
                    else:
                        h[lo] -= cut
                        cut = 0
                    if cut <= 0:
                        break
                # remove cutoff% samples from the high end
                cut = n * cutoff[1] // 100
                for hi in range(255, -1, -1):
                    if cut > h[hi]:
                        cut = cut - h[hi]
                        h[hi] = 0
                    else:
                        h[hi] -= cut
                        cut = 0
                    if cut <= 0:
                        break
            # find lowest/highest samples after preprocessing
            for lo in range(256):
                if h[lo]:
                    break
            for hi in range(255, -1, -1):
                if h[hi]:
                    break
            if hi <= lo:
                # don't bother
                lut.extend(list(range(256)))
            else:
                scale = 255.0 / (hi - lo)
                offset = -lo * scale
                for ix in range(256):
                    ix = int(ix * scale + offset)
                    if ix < 0:
                        ix = 0
                    elif ix > 255:
                        ix = 255
                    lut.append(ix)

        lut = lut + lut + lut
        return image.point(lut)
