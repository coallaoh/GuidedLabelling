# GuidedLabelling, CVPR'17

#### Seong Joon Oh, Rodrigo Benenson, Anna Khoreva, Zeynep Akata, Mario Fritz, Bernt Schiele.

#### Max-Planck Institute for Informatics.

[Exploiting Saliency for Object Segmentation from Image Level Labels](https://arxiv.org/abs/1701.08261), CVPR'17

![TEASER](http://datasets.d2.mpi-inf.mpg.de/joon17cvpr/teaser.jpg)

There have been remarkable improvements in the semantic labelling task in the recent years. However, the state of the art methods rely on large-scale pixel-level annotations. This paper studies the problem of training a pixel-wise semantic labeller network from image-level annotations of the present object classes. Recently, it has been shown that high quality seeds indicating discriminative object regions can be obtained from image-level labels. Without additional information, obtaining the full extent of the object is an inherently ill-posed problem due to co-occurrences. We propose using a saliency model as additional information and hereby exploit prior knowledge on the object extent and image statistics. We show how to combine both information sources in order to recover 80% of the fully supervised performance - which is the new state of the art in weakly supervised training for pixel-wise semantic labelling.

## Installation

Clone this repository recursively.

```bash
$ git clone https://github.com/coallaoh/GuidedLabelling.git --recursive
```

#### Install Caffe

```bash
$ cd caffe
```

Follow [caffe installation](http://caffe.berkeleyvision.org/installation.html) to configure Makefile.config, and run

```bash
$ make -j50 && make pycaffe
```

#### Downloads

Download precomputed saliency maps, network initialisations, etc.

```bash
$ ./downloads.sh
```

#### Python requirements

Install Python requirements.

```bash
$ pip install numpy && pip install scipy && pip install -r ./pip-requirements
```

Install OpenCV for python, following the instructions in: http://opencv.org.

Install PyDenseCRF (https://github.com/lucasb-eyer/pydensecrf).

```bash
$ pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Running

For every image, you compute (1) a seed heatmap, (2) a saliency map, 
and (3) a guide labelling as the training ground truth for the segmentation network
(DeepLab in this case).

The following script does

* Seed network training.
* Computation of seed heatmaps for each image.
* Generate guide labels by combining seed and saliency.
* Train semantic segmentation network using guide labels.
* Test and evaluate the segmentation network.

```bash
$ ./script.py
```

Before running, please change the variable `PASCALROOT` to indicate the root directory for your [Pascal VOC](http://host.robots.ox.ac.uk:8080/pascal/VOC/) database, and set the variable `GPU` to the gpu device number of your choice. Please read the script for greater details.

The final segmentation performance on the Pascal *val* set is computed automatically: it should be **51.419** and **56.153** mIoU, before and after the CRF postprocessing, respectively. They are slightly better than what is reported in our paper (51.2 and 55.7 respectively). 

#### For keen people

```bash
$ src/
```

contains additional scripts for e.g. evaluating seed performance. 
Please read the source scripts and/or run for example

```bash
$ ./src/seed/train.py -h
```

to see options for playing with experimental parameters.

## Contact

For any problem with implementation or bug, please contact [Seong Joon Oh](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/seong-joon-oh/) (joon at mpi-inf dot mpg dot de).

## Citation

```
    @inproceedings{joon17cvpr,
        title = {Exploiting Saliency for Object Segmentation from Image Level Labels},
        author = {Oh, Seong Joon and Benenson, Rodrigo and Khoreva, Anna and Akata, Zeynep and Fritz, Mario and Schiele, Bernt},
        year = {2017},
        booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
        note = {to appear},
        pubstate = {published},
        tppubtype = {inproceedings}
    }
```
