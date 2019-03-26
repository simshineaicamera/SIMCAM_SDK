#!/bin/sh
if ! test -f prototxts/train.prototxt ;then
	echo "error: train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot
/opt/movidius/ssd-caffe/build/tools/caffe train -solver="solver_train.prototxt" \
#-snapshot="simcam_iter_10000.solvrate" 
#-weights="simcam_iter_10000.caffemodel" \
#-gpu 0


