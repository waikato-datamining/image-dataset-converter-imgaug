Changelog
=========

0.0.11 (????-??-??)
-------------------

- switched to `kasperl` library for base API and generic pipeline plugins


0.0.10 (2025-07-11)
-------------------

- switched from `imgaug3` to `imaug` for numpy 2 support
- added `roi-images` filter for extracting sub-images based on object-detection bounding boxes
- the `find-contours` filter now works on any type of image data
- the `transfer_region` method (from `_sub_image_utils`) now initializes the labels of the
  full image segmentation annotations with the labels of the sub-image(s) to ensure correct output


0.0.9 (2025-04-03)
------------------

- fixed `idc.imgaug.filter.transfer_region` method: image segmentation layers now use values (0,255)
- using underscores now instead of dashes in dependencies (`setup.py`)
- renamed `idc.imgaug.filter._sub_images_utils.process_image` to `extract_regions`
- added `crop-to-label` filter that crops the image to the bbox of the annotation with the specified label
- the `sub-images` and `meta-sub-images` filter now have options to generate a regions via predefined number
  of rows/cols or row height/col width on the fly, useful when dealing with images of differing dimensions
  (and the regions parameter is now optional)


0.0.8 (2025-03-14)
------------------

- added support for placeholders: `idc-combine-sub-images`


0.0.7 (2025-02-26)
------------------

- `idc-generate-regions` tool can take margins and overlaps (right/bottom) into account for its region calculations now
- `idc-combine-sub-images` tool now has more details in the exceptions when extraction of groups fail
  and prunes the annotation after the merge as well
- `meta-sub-images` now outputs a logging message if there are no annotations after transferring
  regions/pruning annotations
- method `transfer_region` now adds the sub-images rather than replacing the tile in the overall layer,
  to allow for overlaps (for reducing edge effects)


0.0.6 (2025-01-13)
------------------

- requiring image_dataset_converter>=0.0.5 now


0.0.5 (2025-01-13)
------------------

- switched to underscores in project name
- switched to new methods from idc.api: `empty_image(...)` and `array_to_image(...)`


0.0.4 (2024-07-16)
------------------

- switched from unmaintained imgaug to imgaug3 library (https://github.com/nsetzer/imgaug)


0.0.3 (2024-07-02)
------------------

- added `find-contours` filter for turning blobs in image segmentation annotations into object detection polygons.


0.0.2 (2024-06-13)
------------------

- added `change-grayscale` filter for change pixel values of grayscale images by a factor of fixed value
- added `clip-grayscale` filter for replacing pixel values that go below or above thresholds
- added `enhance` filter for enhancing brightness, contrast, color, sharpness


0.0.1 (2024-05-06)
------------------

- initial release

