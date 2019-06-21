#! /bin/bash
#
# download-data.sh
# Copyright (C) 2019 Jiayuan Mao <maojiayuan@gmail.com>
#
# Distributed under terms of the MIT license.
#

set +e +x

mkdir -p data

for fname in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
do
    curl -o data/$fname http://yann.lecun.com/exdb/mnist/$fname
    cd data; gunzip $fname; cd ..;
done

