# Progress Log

## Week 1

- Created [image-subregion-extractor](https://github.com/whitews/image-subregion-extractor) for quickly saving sub-regions from images to create training and test sets [Scott]
- Start TensorFlow image detection pipeline [Ben]
- Blob detection using differential operators [Lina]
- Kingshuk came as guest - made following suggestions
    - Use full use of biological knowledge including 3D if available
    - Watershed is difficult to make work well in his experience

### Group meeting actions

- Use systematic approach to pipeline stage evaluation [Lina]
- To prepare tutorial on TensorFlow (Ben)
- Evaluate classification pipeline (Scott)

## Week 2

- Update on marker spreadsheet by Susan
- Revisions to blob detection pipeline [Lina]
- Cascade classifier performs very poorly (Scott)
- Created [image-subregion-detector](https://github.com/whitews/image-subregion-detector) for real-time image segmentation (Scott)
![Detected](images/detected.png)
- Created mapping of symbolic color names to HSV space (Scott)
- Presented TensorFlow demo (Ben)

### Group meeting actions

- Develop list of segmentation targets and characteristics [Lina]
- Manual feature selection/reduction for classification of acinar tubules [Lina]
- Implement [digit recognition from Google Street View](https://www.udacity.com/course/deep-learning--ud730) pipeline for TensorFlow (Ben)
- Add features to image-subregion-detector (Scott)
    - API for region detection pipeline plugins
    - Accept/Reject detected image regions
    - Consider how to integrate image metadata to improve region detection
    - Consider selection of positive and negative regions to Start
    - Consider iterative improvement of detected regions (reinforcement learning)

## Week 3

- Updates to [image-subregion-detector](https://github.com/whitews/image-subregion-detector) (Scott)
    - Expanded definition for 'black' HSV color range
    - Added functionality to specify (multiple) background color ranges
    - Remove detected sub-regions by right-clicking on them
    - BUGFIXES: properly clear user-drawn rects, support 16-bit RGB TIFs (by downsampling to 8-bit)
- Explored segmentation targets (cells and anatomical structures as below) and their characteristics (Lina)
  (brief descriptions are in this [pdf](https://github.com/duke-lungmap-team/lungmap-scratch/tree/master/Lina/Segmentation_targets.pdf) )
    - Pericytes
    - Club cells
    - Ciliated cells
    - Bronchioles
    - Acinar tubules (some thoughts on acinar tubule criteria)
- Continuing research into the use of machine learning algorithms implemented via TensorFlow (Ben)
    - If we were to build a training set (Scott's tool), we'd have about one-tenth the amount of usual machine learning training data
    - Currnetly, implementing the digit recognition algorithm with low sample counts to simulate our situation to get a feel for "deterioration" of accurarcy with low training sets
    - Spiking on the following machine learning techniques in TensorFlow:
        - Covnets
        - Regularization
        - Dropout
        - ReLUs (Rectified Linear Units)

## Week 4

- More updates to [image-subregion-detector](https://github.com/whitews/image-subregion-detector) (Scott)
    - [Sub-region detector presentation](./scott/Sub-region Detector Algo Summary.pdf)
    - Added interactive preview image for easier navigation around the main canvas
    - Added some stats about the detected regions (count, min/max/avg size)
    - Display color range % for user drawn rectangle (helps choose appropriate bg colors)
    - Better region detection for single cells (via new pre-erosion option)
    - BUGFIXES:
        - Clear sub-regions when selecting a new image
        - Fix typo in maximum area label
        - Fix error thrown if zero regions are detected
    - New screenshot:
    ![Detected_week4](scott/ImageSubregionDetector_week4.png)
- Built two neural network algorithms (TensorFlow) to examine impact of sample size on accuracy (Ben)
    - Files for replication can be found in [./queries](./queries/README.md)
    - **Results**
        - Multinomial Logistic Classifier (Gradient Descent) - [visually seems to indicate > 4,000 training images](./queries/viable_sample_size/logistic_classifier2.png)
        - Multinomial Logistic Classifier (Stochastic Gradient Descent) - [visually seems to indicate 4,000 training images](./queries/viable_sample_size/mlc_sgd.png)
    - Where to go from here?
        - Continue for at least one more week to see if there are techniques for dealing with small sample sizes, such as:
        1. Try to make the network "deeper" - add several more nodes including Rectified Linear Units (ReLUs) and Regularization
        2. Stocastic Gradient Descent, can be thought of as a type of bootstrap, would like to play around with this idea a bit more
        3. Explore the idea of building a (supervised probabilistic classifier via Gaussian Copulas)[http://www.cimat.mx/~mrivera/bookchapt/salinas_copulas_lncs10.pdf]
- Literature reading, summarized potential useful features that may facilitate structure classification, and identified blood vessels(Lina)
    - [features](https://github.com/duke-lungmap-team/lungmap-scratch/tree/master/Lina/Commonly used features for analyzing Histology Images.pdf)
        - Color
        - Texture
        - Morphology
        - Architecture
    - Identified [Blood vessels in 6 images](https://github.com/duke-lungmap-team/lungmap-scratch/tree/master/Lina/blood_vessels.pdf)

## Week of Oct 3rd

- Created script to parse sub-region detector output (Scott)
    - Output is a JSON file compatible with TensorBox
    - Also created a Jupyter notebook to verify the location of the JSON bounding boxes

## Week of Oct 10th

- Extracted sub-regions for acinar tubles from experiment 73 - all 20x images at E16.5 (Scott)
    - exp 73 has labelled proteins: Acta2 (white), Sftpc (red), and Sox9 (green)
    - extracted distal acinar tubules as contours using the detector
    - extracted proximal acinar tubules as sub-regions using the extractor
    - extracted acinar tubule negative set of sub-regions using the extractor
- Began looking at other experiments with 20x E16.5 images (Scott)
    - Focused on experiment 41 - has labelled proteins: Sox2 (green), Nkx2-1 (red), Acta2 (white)
    - Noticed the red probe was different than the red used in exp 73
    - For red, exp 73's Sftpc used Alexa Fluor 568 and exp 41's Nkx2-1 used Alexa Fluor 555
    - Is there any way to get the fluorophore from the API?
- Aligned first pipeline from lungmap images to Tensorbox. (Ben)
  - The results seem promising, so will continue to examine this tool for creating one algorithm capable of segmenting multiple anatomical objects within one image. However, to make this goal real, many iterations of models will need to be examined and experimented with. To help keep track of all algorithms, a separate repository (just for machine learning algorithms) is now up and available for review [here](https://github.com/duke-lungmap-team/lungmap_algorithms). This next week, we will plan to:
  - Re-run the first model with 10,000,000 iterations (the default setting for this pipeline)
  - This means we will need to set-up a VM to host this compute, so will work on getting that infrastructure set up.
  - Train a new model that only considers the grayscale versions of the images. Allowing us to experiment with both possibilites that computer vision doesn't need the stains and/or building more complicated algorithms that first identify general anatomical structures (i.e. acinar tubule) that then feed into other algorithms that distinguish between distal and proximal.
- Began extracting sub-regions for blood vessels at all developmental stages (Lina)
- Added functions for image features (Lina)
  - perimeter_area ratio (may be useful for distinguishing bronchioles from blood vessels)
  - entropy (needed to be improved(return a single value from a sub-region instead of returning a value for each pixel))

## Summary for August 2016

### Software
- Set up an open source [Github repository](https://github.com/duke-lungmap-team/lungmap-scratch) to share code and documents among group
- Built [image-subregion-extrractor](https://github.com/whitews/image-subregion-extractor) tool for rapid construction of training sets
- Built [image-subregion-detector](https://github.com/whitews/image-subregion-detector) as GUI for evaluating image segmentation pipelines.

### Algorithms
- Investigation of Google's TensorFlow library for image classification using deep learning. Initial impression is that TensorFlow applications may require a larger training set than we have in order to give accurate classification, but this is still being [looked into](./queries/README.md).
- [Rapid two-stage image segmentation pipeline](./scott/Sub-region Detector Algo Summary.pdf) based on blob detection by immunofluorescent color foreground/background partitioning and blob classification by size constraints developed as prototype engine for image-subregion-detector.
- [Review of literature](https://github.com/duke-lungmap-team/lungmap-scratch/tree/master/Lina/Commonly used features for analyzing Histology Images.pdf) for statistical features (color, texture, morphology and architecture) useful for classifying histology images

### Knowledge Base
- Review of [characteristic features](https://github.com/duke-lungmap-team/lungmap-scratch/tree/master/Lina/Segmentation_targets.pdf)) of anatomical structures and cells in the developing lung

## Summary for September 2016

### Software
- Added ability of  [image-subregion-extractor](https://github.com/whitews/image-subregion-extractor) to capture bit masks as numpy arrays
- Built [plug-play-algorithm-app](https://github.com/duke-lungmap-team/plug-play-algorithm-app) to evaluate object recognition algorithms as plug-ins (successful demo using Haar cascade classifier plug-in for real-time face recognition)
- Script to run TensorBox training and evaluation completed - worked out issues with specifying training sets and evaluated on test data
- Script to extract bounding boxes for extracted images in training set in format required for TensorBox

### Algorithms
- Use of [TensorBox](https://github.com/Russell91/TensorBox) for object detection and segmentation

## Summary for October 2016

- Complete code for extracting image segments and bounding boxes
- Extract training sets for acinar tubules and bronchioles

### Image classification Progress

- Move to label-free system for robust classification
- Initially used manual feature extraction
- Now use `wndcharm` for extraction
- Over 4000 features per image
- Initial evaluation on 3 classes extracted by Scott (background, proxmial and distal acinar tubules)
- Use of `sklearn` to construct pipelines for classification
    - over 95% cross-validation accuracy seen with classifiers out-of-box (almost no tuning done)
    - See Classification.ipynb notebook in cliburn folder for code

## Running objectives

1. Heuristic algorithm to extract sub-images for training set (stage 1: create blobs from feature colors, stage 2: filter for blobs that are similar to exemplar)
  - [x] Graphical user interface
  - [x] Object recognition using single exemplar
  - [x] Export images to numpy arrays as training sets
  - [x] Create JSON file with bounding boxes for target locations
  - [ ] Object recognition using multiple exemplars
2. Build positive and negative training sets for anatomical objects
  - [x] [Tubules](https://duke.box.com/s/mz6a57k14b2chohhykuw4p4u9as6jr89)
  - [x] [Terminal bronchioles](https://duke.box.com/s/k64zivfozlryddqzkloatnbi70c9ndcr)
  - [ ] Blood vessels
  - [ ] Type II epithelial cells
3. Evaluate summary image features for classification
  - [ ] Define list of features for evaluation on training sets
  - [ ] Evaluate performance of individual features for classification accuracy
  - [ ] Evaluate performance of combined features for classification accuracy
  - [ ] Integrate features found into stage 2 of image extractor
4. Evaluate deep learning for in-image object recognition and segmentation
  - [x] Evaluate how TensorFlow library works
  - [x] Graphical user interface to plug-in algorithms
  - [x] Train and test on standard data sets
  - [ ] Train and test on LungMAP IHC positive and negative training sets
5. Construct a formal knowledge base of interesting anatomical structures and cells and their statistical features in the developing mouse lung (with Anna Maria)
  - [ ] Create a table with rows containing (name, stage, feature, measurement, statistic, value) e.g. (proximal tubule, E16.5, SOCS-9, area, min, 20 $\mu$m)
  - [ ] Use of knowledge base to provide sensible default parameters (e.g. # erosions) for known targets so as to increase sensitivity of blob detection (stage 1)
  - [ ] Use of knowledge base to create filters based on feature statistics so as to increase specificity of blob classification (stage 2)
6. Explore patterns with statistical analysis of discovered image segments (with Kingshuk)
  - [ ] Cross-sectional analysis of counts and distributions
  - [ ] Longitudinal analysis of counts and distributions

## Summary for December 2016
- Improved candidate search by using both the hue and saturation channels of the HSV image (Scott)

## Week of Jan 16th
- Began investigating replacing wnd-charm with custom features with good results & is much faster (Scott)

## Week of Jan 23rd
- Augmented image training set of bronchioles and blood vessels (Lina)
  - Method 1: Rotated image in 90\degree, 180\degree, 270\degree.
  - Method 2: Random transformed (shift, rescale, shear, zoom, flip, rotation) by making use of features in Keras.
- Trained classifier with the augmented training set.
- Replaced wnd-charm in the Tkinter identifier application, much faster now & seems just as accurate (Scott)
- Began work on web version of the image sub-region identifier app (Scott)

## Week of Jan 30th
- Finished working prototype of web identifier to be feature complete with the Tkinter version (Scott)

## Week of Feb 6th
- Began investigating custom features to better differentiate bronchioles from the "open" blood vessels in the E16.5 mouse images (Scott)
  - One possibility is to incorporate categorical features: Cliburn recommended using the [OneHotEncoder from sklearn](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features), which looks promising

- Dockerized a version of the [wa-isit app](https://github.com/whitews/wa-isit). Previously this app was deployed using flask's default web server. As we saw during a live test, and as documented in Flask, this was not a good idea. Took time to update the docker container to serve the application using uwsgi, so the [live demo](http://rapid-235.vm.duke.edu:5000/) is much more robust. (Ben)
  - working on porting (to Angular) a prototype web application ([goldmill](https://github.com/benneely/goldmill)) that will allow us to obtain better training data and make better use of the LungMap ontology. (Ben)

## Week of Feb 13th
- Drafted a [roadmap](https://github.com/duke-lungmap-team/ihc-image-analysis/blob/master/roadmap.md) for delivery of final product to be build over the next year (Scott)

## Week of Feb 20th
- Began investigating improving the training set using polygon segmentations of target regions (Scott)
  - In theory, this should improve the SNR for the custom feature metrics by ignoring the majority of peripheral pixels
  - Began [version 2 of the sub-region extractor](https://github.com/whitews/image-subregion-poly-extractor)
- Began investigating the potential criteria to exclude "outliers" (Lina)
  - "Outliers" are sub-regions that do not belong to any anatomical structural classes in the training set but forced to be one of them by our classifier.
  - Methods that I am investigating include distance based methods and kernel based novelty detection.

## Update prior to Mar 1st meeting
- Investigating polygon training set (Scott)
  - Finished a working version of the [poly extractor](https://github.com/whitews/image-subregion-poly-extractor)
    - Nice feature of the tool is that it extracts and saves both the bounding rectangle and the grayscale polygon mask, so it is easy to compare methods using the same training set on either the masked or original rectangular regions.
  - Used the poly extractor to re-create training data for experiment 73
    - Segmented 426 regions from the 4 images...yeah, this is tedious work and takes a while!
  - Updated [lung-map-utils](https://github.com/duke-lungmap-team/lung-map-utils) to take optional user-specified mask and optional suffix for sig file names.
  - Updated notebook for [generating signature files](https://github.com/duke-lungmap-team/lungmap-scratch/blob/master/scott/custom_feature_sig_file_generation.ipynb)
  - Created notebook comparing accuracy of [custom features vs custom masked features vs wndcharm](https://github.com/duke-lungmap-team/lungmap-scratch/blob/master/scott/custom_features_vs_custom_feature_masked_vs_wndcharm.ipynb)
  - **TODO:** If we want continue with the masked region strategy, the next step is to solve the contour filling issue with the candidates.
