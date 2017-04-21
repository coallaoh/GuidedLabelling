# GuidedLabelling
Exploiting Saliency for Object Segmentation from Image Level Labels, CVPR'17

![TEASER](http://datasets.d2.mpi-inf.mpg.de/joon17cvpr/teaser.jpg)

There have been remarkable improvements in the semantic labelling task in the recent years. However, the state of the art methods rely on large-scale pixel-level annotations. This paper studies the problem of training a pixel-wise semantic labeller network from image-level annotations of the present object classes. Recently, it has been shown that high quality seeds indicating discriminative object regions can be obtained from image-level labels. Without additional information, obtaining the full extent of the object is an inherently ill-posed problem due to co-occurrences. We propose using a saliency model as additional information and hereby exploit prior knowledge on the object extent and image statistics. We show how to combine both information sources in order to recover 80% of the fully supervised performance - which is the new state of the art in weakly supervised training for pixel-wise semantic labelling.

[Paper](https://arxiv.org/abs/1701.08261)

## Setup

```bash
$ git clone https://github.com/coallaoh/GuidedLabelling.git --recursive
```

```bash
$ cd caffe
```

Follow caffe installation to configure Makefile.config

```bash
$ make -j50 && make pycaffe
```

## Downloads

```bash
$ ./downloads.sh
```

## Python requirements

```bash
$ pip install -r ./pip-requirements
```

## Running

```bash
$ ./script.py
```

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