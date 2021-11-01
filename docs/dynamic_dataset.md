# Dynamic Dataset

### Features
- **Uses raw images** - they don't have to be the same size. You don't need to use `dataset_tool.py` before training. Just take the images as they are and put them into single directory or zip file.
- Images are cropped to the requested size (e.g. `--dd-res=1024`) on the fly.
- Either **center crop** or **random crop** is available (`--dd-crop=center`, `--dd-crop=random`)
- Using **random crop** the image is cropped differently each time it is used, which leads to to additional augmentation. 

### @todo
- Support for conditional training
- smart crop `--dd-zoom-out=90` Max zoom out factor 50-100%
