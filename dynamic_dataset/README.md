# Dynamic Dataset


## Features
- **Uses raw images**
  - Images don't have to be the same size.
  - You don't need to use `dataset_tool.py` before training.
  Just take the images as they are and put them into single directory, or a zip file.
- **Preprocessing images on the fly**
  - Images are converted to 24-bit RGB
  - Images are cropped and resized to the requested resolution according to parameters `--dd-res`, `--dd-crop` and `--dd-scale`
  - Images are randomly autocontrasted according to parameters `--dd-ac-prob` and `--dd-ac-cutoff`
  - Since the images are cropped, resized and autocontrasted differently each time they are used, this leads to additional augmentation
- **Simple directory structure**
  - Both **conditional** and **non-conditional** datasets supported
  - No naming convention - image files can be named any way you prefer.
  - **Wide variety of image formats** `JPEG`, `PNG`, `GIF` and all image formats supported by the
  <a href="https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html">PIL library</a>.

## Command line arguments

Dynamic Dataset adds the following command line options to StyleGan3:
* `--dd` Tells Stylegan to use DynamicDataset instead of the original [ImageFolderDataset](https://github.com/vsemecky/stylegan3/blob/a5d04260b4037c0d2e3c3cb5ab43ce5b84de65d7/training/dataset.py#L164)
* `--dd-res`       Requested resolution (default `--dd-res=1024x1024`)
* `--dd-crop`      Cropping type: 'center' or 'random'
* `--dd-scale`     Scale/zoom factor. 1 = no zoom, 0.8 = crop up to 20%',    type=click.FloatRange(min=0.5, max=1), default=0.8, show_default=True)
* `--dd-ac-prob`   Autocontrast probability
* `--dd-ac-cutoff` Maximum percent to cut off from the histogram

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
 ┣━🧾 image12345.webp
 ┣━🧾 another_image.jpg
 ┣━🧾 file-name-does-not-matter.jpg
 ┣━🧾 ...
 ┣━🧾 ...
```


</td>
<td>

**Conditional**

```
📂 dataset-folder
 ┣━📂 class1
 ┃  ┣━🧾 image001.jpg
 ┃  ┣━🧾 image001.png
 ┃  ┣━🧾 image001.gif
 ┃  ┣━🧾 image004.webp
 ┃  ┣━🧾 image12345.webp
 ┃  ┣━...
 ┃  ┣━...
 ┣━📂 second_class
 ┃ ┣━🧾 image001.jpg
 ┃ ┣━🧾 image001.png
 ┃ ┣━🧾 image001.gif
 ┃ ┣━🧾 image004.webp
 ┃ ┣━...
 ┃ ┣━...
 ┣━📂 another_class
   ┣━🧾 image001.gif
   ┣━🧾 image002.png
   ┣━🧾 image003.gif
   ┣━🧾 image004.webp
   ┣━🧾 image12345.webp
```

</td>
</table>




### @todo
- Support for conditional training
- smart crop `--dd-zoom-out=90` Max zoom out factor 50-100%

Maximálně zjednodušuje přípravu datasetu. Odbourává zbytečné převádění obrázků pomocí nástrojů jako je dataset_tool.py - jejich práci dělá za běhu. 


# Datasets

## Un-Conditional Dynamic Dataset

**Folder or ZIP file full of images**. Although I recommend a flat layout (all images in a single folder), Dynamic Dataset process the image files recursively, including all subfolders.
```
/dataset-folder
    - cat_image_1.jpg
    - image2-cat.jpg
    - my_cat_photo_123.jpg
    - another-image.jpg
    - image_1.jpg
    - image-next.jpg
    - another-image.jpg
```


## Conditional
**Folder or ZIP file with subfolder for each class**.
Although I recommend just class subfolders (all class-images in a single class-subfolder), Dynamic Dataset process the class-subfolders recursively, so they can contain other subfolders whose structure does not matter.

```
/dataset-folder
    /class-cat
        - cat_image_1.jpg
        - image2-cat.jpg
        - my_cat_photo_123.jpg
        - another-image.jpg
    /second_class
        - image_1.jpg
        - image-next.jpg
        - another-image.jpg
    /1
    /2
    /another-class
```

# FAQ

## Do you support grayscale images, transparent GIFs or semitransparent PNGs with alpha channel?
No, sorry. Dynamic Dataset works in `24-bit RGB` mode. Images in other color
modes (`32-bit RGB`, `8-bit`, `4-bit`, `grayscale`, `RGBA`) are converted into `24-bit RGB`.
I do not recommend using transparent PNG/GIF - colour tones may be affected by the conversion.
