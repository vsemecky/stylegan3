# Dynamic Dataset

Alternative dataset loader for StyleGan3 that uses raw images with different resolution and preprocess them on the fly.

## Features
- **Uses raw images**
  - Images don't have to be the same size.
  - You don't need to use `dataset_tool.py` before training.
  - You don't need metadata file `dataset.json` for labels
- **Preprocessing images on the fly**
  - Images are converted to 24-bit RGB
  - Images are cropped and resized to the requested resolution according to parameters `--dd-res`, `--dd-crop` and `--dd-scale`
  - Images are randomly autocontrasted according to parameters `--dd-ac-prob` and `--dd-ac-cutoff`
  - Note: Since the images are cropped, resized and autocontrasted differently each time they are used, this leads to **additional augmentation**
- **Simple directory structure**
  - Both **conditional** and **non-conditional** datasets supported
  - No naming convention - image files can be named any way you prefer.
  - **Wide variety of image formats** `JPEG`, `PNG`, `GIF` and all image formats supported by the
  <a href="https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html">PIL library</a>.

## Dataset structure

* **Un-Conditional**: Folder or ZIP file full of images
* **Conditional**: Folder or ZIP file with subfolder for each `class id` (should be `integer` but it's OK to use leading zeros).

<table>
<td>

**Un-Conditional**

```
📂 dataset-folder
 ┣━🧾 image001.jpg
 ┣━🧾 image001.png
 ┣━🧾 image001.gif
 ┣━🧾 image002.png
 ┣━🧾 image003.gif
 ┣━🧾 image004.webp
 ┣━🧾 myimage.jpg
 ┣━🧾 image12345.webp
 ┣━🧾 another_image.jpg
 ┣━🧾 yet_another_image.jpg
 ┣━🧾 file-name-does-not-matter.jpg
 ┣━🧾 image-abcd.jpg
 ┣━🧾 ...
 ┣━🧾 ...
 ┣━🧾 ...
```


</td>
<td>

**Conditional**

```
📂 dataset-folder
 ┣━📂 01
 ┃  ┣━🧾 image001.jpg
 ┃  ┣━🧾 image001.png
 ┃  ┣━🧾 image001.gif
 ┃  ┣━...
 ┣━📂 02
 ┃ ┣━🧾 image001.jpg
 ┃ ┣━🧾 image001.png
 ┃ ┣━🧾 image004.webp
 ┃ ┣━...
 ┣━📂 03
   ┣━🧾 myimage.jpg
   ┣━🧾 another_image.jpg
   ┣━🧾 yet_another_image.jpg
   ┣━🧾 ...
```

</td>
</table>

## Command line arguments

Dynamic Dataset adds the following command line options to StyleGan3:
* `--dd` Tells Stylegan to use DynamicDataset instead of the original [ImageFolderDataset](https://github.com/vsemecky/stylegan3/blob/a5d04260b4037c0d2e3c3cb5ab43ce5b84de65d7/training/dataset.py#L164)
* `--dd-res`       Requested resolution (default `--dd-res=1024x1024`)
* `--dd-crop`      Cropping type: `center` or `random` (default `--dd-crop=random`)
* `--dd-scale`     Scale/zoom factor. 1 = no zoom, 0.8 = crop up to 20% (default: `--dd-scale=0.8`)
* `--dd-ac-prob`   Autocontrast probability (default: `--dd-ac-prob=0.8`)
* `--dd-ac-cutoff` Maximum percent to cut off from the histogram (default: `--dd-ac-cutoff=2`)


## Usage

- TODO
