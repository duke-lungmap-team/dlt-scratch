# Image Data

* Most (all?) of the lungmap images are in TIF format
* Some of the images are 3-channel RGB, 16-bit per channel which is not supported in many image libraries
  * OpenCV will read these but will downsample to 8-bit per channel by default
  * PIL will not read these, so they need to be converted to 8-bit per channel first
