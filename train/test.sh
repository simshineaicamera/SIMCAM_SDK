#!/bin/sh

latest=$(ls -t snapshot/*.caffemodel | head -n 1)
if test -z $latest; then
	exit 1
fi
/opt/movidius/ssd-caffe/build/tools/caffe train -solver="solver_test.prototxt" \
--weights=$latest \
#-gpu 0
