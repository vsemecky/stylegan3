# Dynamic Dataset

Before, for offline augmentation I used my simpler command-line tool <a href="https://github.com/vsemecky/augmentor">Augmentor</a>,
which I created for the <a href="https://thisbeachdoesnotexist.com/">This Beach Does Not Exist</a>. Now I've decided to integrate Augmentor directly into the <a href="https://github.com/vsemecky/stylegan3">StyleGan3</a>
as a new dataset type that will do the augmentation on the fly, so you won't need the tools like `dataset_tool.py`.

It is based on my simpler Augmentor command-line tool, which I used for offline augmentation for <a href="https://thisbeachdoesnotexist.com">This Beach Does Not Exist</a>.

I have now decided to integrate Augmentor directly into StyleGan3 as a new dataset type that will do augmentation on the fly.

## Features
- **Uses raw images** - images don't have to be the same size. You don't need to use `dataset_tool.py` before training.
  Just take the images as they are and put them into single directory, or a zip file.
- **Cropping on the fly** - Images are cropped to the requested size on the fly.
  The image is cropped differently each time it is used, which leads to the additional augmentation. 
- **simple directory structure**
- conditional and non-conditional datasets/networks supported
- **No naming convention** - image files can be named any way you prefer.
- **Wide variety of image formats** - not only `JPEG`, `PNG`, `GIF` but all image formats supported by the
  <a href="https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html">PIL library</a>.
- **24-bit RGB** mode. Images in other color modes (`32-bit RGB`, `8-bit`, `grayscale`, `RGBA`) are converted to 24-bit RGB.
  I do not recommend using transparent PNG or transparent GIF, because transparent color can break the overall color tones.

## Není killer fíčura, možná pro zjednodušení nezmiňovat
- **ignores non-image files** automatically (e.i. It doesn't matter if there are some text files in the directory).

## Command line arguments

`--dd RESOLUTION`  Init DynamicDataset with specific resolution (e.g. `--dd=1024` or `--dd=1024x1024`)

`--dd-crop`      Cropping type: 'center' or 'random' (default='random')
`--dd-scale`     Scale/zoom factor. 1 = no zoom, 0.8 = crop up to 20%',    type=click.FloatRange(min=0.5, max=1), default=0.8, show_default=True)
`--dd-ac-prob`,  Autocontrast probability (default: %(default)s)",         type=click.FloatRange(min=0, max=1), default=0.8, show_default=True)
`--dd-ac-cutoff` Maximum percent to cut off from the histogram (default: %(default)s)", type=float, default=2, show_default=True)

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
