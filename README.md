# AACNN
Tensorflow Inplementation for AACNN : Attribute Augmented Convolutional Neural Network for Face Hallucination (NTIRE2018)
# Paper
[Attribute Augmented Convolutional Neural Network for Face Hallucination](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Lee_Attribute_Augmented_Convolutional_CVPR_2018_paper.pdf) <br/>
[Cheng-Han Lee](https://github.com/steven413d)<sup> 1</sup>, [Kaipeng Zhang](http://kpzhang93.github.io/)<sup> 1</sup>, Hu-Cheng Lee<sup> 1</sup>, Chia-Wen Cheng<sup> 2</sup>, and [Winston H. Hsu](https://winstonhsu.info/)<sup> 1</sup>    <br/>
<sup>1 </sup>National Taiwan University, <sup>2 </sup>The University of Texas at Austin <br/>
IEEE Conference on Computer Vision and Pattern Recognition Workshop, ([NTIRE 2018](http://www.vision.ee.ethz.ch/ntire18/))
<br/>
## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [TensorFlow 1.8](https://www.tensorflow.org/)

## Train_Model
* The Installation completely the same as our dependencies. Make sure you have correctly installed before using our code.
* Download the align & cropped version of [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset for training and testing
* Preprocess the training face images, including detection, alignment, etc. Here we strongly recommend [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment), which is an effective and efficient open-source tool for face detection and alignment.
* Put aligned images under "./data/CelebA"
* For L2 version : bash train.sh #NUM_GPU, For L2 + GAN version :  bash train_gan.sh #NUM_GPU

## Inference_Model
* bash test.sh #NUM_GPU
* For evaluation : run test_psnr.m & test_ssim.m on Matlab

## To Do
* Add auxilary classifier loss
* Replace BN in discriminator with SN
