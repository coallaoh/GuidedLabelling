#!/usr/bin/env bash

PASCALROOT="/BS/joon_projects/work/"

# Download data
wget https://transfer.d2.mpi-inf.mpg.de/joon/joon17cvpr/data.tar.gz
tar xvf data.tar.gz
rm data.tar.gz

wget https://transfer.d2.mpi-inf.mpg.de/joon/joon17cvpr/list.tar.gz
tar xvf list.tar.gz
rm list.tar.gz

mv list {$PASCALROOT}VOC2012/ImageSets/Segmentation/
