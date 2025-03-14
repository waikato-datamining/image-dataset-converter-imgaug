Changelog
=========

0.0.8 (2025-03-14)
------------------

- added support for placeholers: `idc-combine-sub-images`


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

