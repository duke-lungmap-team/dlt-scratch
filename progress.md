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
- Literature reading, summarized potential useful features that may facilitate structure classification(Lina)
    - [features](https://github.com/duke-lungmap-team/lungmap-scratch/tree/master/Lina/Commonly used features for analyzing Histology Images.pdf)
        - Color
        - Texture
        - Morphology
        - Architecture
    - Identified [Blood vessels in 5 images](https://github.com/duke-lungmap-team/lungmap-scratch/tree/master/Lina/blood_vessels.pdf)