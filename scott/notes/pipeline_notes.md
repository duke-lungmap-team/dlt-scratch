## Magic numbers & values
- color_blur_kernel = (27, 27)
- lrg_struct_blur_kernel = (31, 31)
- small_struct_blur_kernel = (91, 91)
- color list for color candidates = [
        'cyan',
        'gray',
        'green',
        'red',
        'violet',
        'white',
        'yellow'
    ]
- color contour dilation kernel = (3, 3)
- color contour dilation iterations = 3
- structuring element for growing large structure saturation candidates
  - cross_strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
  - cross_strel dilation iterations = 47
- saturation bins for small structure saturation candidates
  - mode_s_small1 = cv2.inRange(img_blur_s_small, 240, 256)
  - mode_s_small2 = cv2.inRange(img_blur_s_small, 232, 248)
  - mode_s_small3 = cv2.inRange(img_blur_s_small, 224, 240)
- size filters for small structure saturation candidates
  - min_size = 32 * 32 * 2
  - max_size = 200 * 200
- structuring element for growing large structure saturation candidates
  - cross_strel dilation iterations = 23
- Percentage overlap criteria for merging regions = 70%
- Minimum probability for accepting classified candidates = ???
