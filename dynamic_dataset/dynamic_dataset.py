"""
Dynamic Dataset for StyleGan3

An alternative dataset that uses raw images and crops/resize them on the fly.
You don't need to use `dataset_tool.py` before training.
"""
import os
import pathlib
import random
import re
import zipfile
import numpy as np
import PIL.Image
import PIL.ImageOps
import PIL.ImageFile
from training.dataset import Dataset


class DynamicDataset(Dataset):

    def __init__(
        self,
        path,
        resolution,
        anamorphic,
        focus="random",
        max_bleed=0.1,
        autocontrast_probability=0,
        autocontrast_max_cutoff=0,
        use_labels=False,
        xflip=False,
        yflip=False,
        **super_kwargs
    ):
        self._path = path
        self._zipfile = None
        self._resolution = self.decode_resolution(resolution)  # Final resolution of the network - tuple (width, height)
        self._focus = focus
        self._max_bleed = max_bleed
        self._use_labels = use_labels
        self._autocontrast_probability = autocontrast_probability
        self._autocontrast_max_cutoff = autocontrast_max_cutoff
        self._anamorphic = anamorphic

        if anamorphic:
            self.anamorphic_resolution = self.decode_resolution(anamorphic)  # tuple (width, height)
            self._crop_size = self.decode_resolution(anamorphic)  # tuple (width, height)
            print("Anamorphic: ", self._crop_size)
        else:
            self._crop_size = self._resolution  # tuple (width, height)
            print("Normal square: ", self._crop_size)

        # ImageFolderInit
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()

        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)

        super().__init__(name=name, raw_shape=raw_shape, use_labels=use_labels, xflip=xflip, yflip=yflip, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

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
        if self._focus == "center":
            centering = (0.5, 0.5)
        else:
            centering = (random.uniform(0, 1), random.uniform(0, 1))  # random centering

        image_pil = self.dynamic_fit(
            image_pil,
            size=self._crop_size,
            final_size=self._resolution,
            max_bleed=self._max_bleed,
            centering=centering
        )

        image_np = np.array(image_pil).transpose(2, 0, 1)  # HWC => CHW
        return image_np

    def _load_raw_labels(self):
        if not self._use_labels:
            return None

        labels = {}
        for fname in self._all_fnames:
            p = pathlib.Path(fname)
            try:
                # label = int(p.parts[0])
                label = int(re.search(r'\d+', p.parts[0]).group())
                labels[fname] = label
            except:
                print(f"Invalid label '{p.parts[0]}' in file path '{fname}'")
                exit()

        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    @staticmethod
    def get_max_window_size(image: PIL.Image, ratio: float):
        """ Returns maximum size (width, height) of inscribed rectangle with specific aspect ratio """
        width, height = image.size
        current_ratio = width / height
        if ratio == current_ratio:
            return width, height
        elif ratio > current_ratio:
            return width, round(width / ratio)
        elif ratio < current_ratio:
            return round(height * ratio), height

    @staticmethod
    def decode_resolution(resolution: str):
        """ Converts string like '1024x1024' or just '1024' to (width, height) """
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
        I will start using PIL.Imageops.autocontrast(..., preserve_tone=True) instead and this method will be removed.
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

    @staticmethod
    def dynamic_fit(image, size=(1024, 1024), final_size=None, max_bleed=0.0, centering=(0.5, 0.5)):
        """
        Returns a resized and cropped version of the image, cropped to the
        requested aspect ratio and size.

        This function is derived from the fit() function by Kevin Cazabon from the PIL library:
        https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageOps.html#fit

        :param final_size: Finally resize image to different size than has been calculated. For anamoprhic purpose.
        :param image: The image to resize and focus.
        :param size: The requested output size in pixels, given as a
                     (width, height) tuple.
        :param max_bleed: How much can we crop into the image (use 0.01 for one percent).
        :param centering: Control the cropping position.  Use (0.5, 0.5) for
                          center cropping (e.g. if cropping the width, take 50% off
                          of the left side, and therefore 50% off the right side).
                          (0.0, 0.0) will focus from the top left corner (i.e. if
                          cropping the width, take all of the focus off of the right
                          side, and if cropping the height, take all of it off the
                          bottom).  (1.0, 0.0) will focus from the bottom left
                          corner, etc. (i.e. if cropping the width, take all of the
                          focus off the left side, and if cropping the height take
                          none from the top, and therefore all off the bottom).
        :return: An image.
        """

        # bleed_horizontal = random.uniform(0, max_bleed)
        # bleed_vertical = random.uniform(0, max_bleed)
        # bleed_pixels = (bleed_vertical * image.size[0], bleed_horizontal * image.size[1])

        # Random number of pixels to trim off on Top and Bottom, Left and Right
        bleed_pixels = (
            random.uniform(0, max_bleed) * image.size[0],
            random.uniform(0, max_bleed) * image.size[1]
        )

        live_size = (
            image.size[0] - bleed_pixels[0] * 2,
            image.size[1] - bleed_pixels[1] * 2,
        )

        # calculate the aspect ratio of the live_size
        live_size_ratio = live_size[0] / live_size[1]

        # calculate the aspect ratio of the output image
        output_ratio = size[0] / size[1]

        # figure out if the sides or top/bottom will be cropped off
        if live_size_ratio == output_ratio:
            # live_size is already the needed ratio
            crop_width = live_size[0]
            crop_height = live_size[1]
        elif live_size_ratio >= output_ratio:
            # live_size is wider than what's needed, focus the sides
            crop_width = output_ratio * live_size[1]
            crop_height = live_size[1]
        else:
            # live_size is taller than what's needed, focus the top and bottom
            crop_width = live_size[0]
            crop_height = live_size[0] / output_ratio

        # make the focus
        crop_left = bleed_pixels[0] + (live_size[0] - crop_width) * centering[0]
        crop_top = bleed_pixels[1] + (live_size[1] - crop_height) * centering[1]
        crop = (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)

        if not final_size:
            final_size = size

        # Crop and resize image to final size
        return image.resize(final_size, resample=PIL.Image.LANCZOS, box=crop)
