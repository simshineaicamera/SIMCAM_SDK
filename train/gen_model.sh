#!/bin/sh
if test -z $1 ;then
	echo usage: $0 CLASSNUM
        echo "  Please enter number of classes in your dataset"
	exit 1
fi
echo $1 |grep '^[0-9]*$' >/dev/null 2>&1
if [ $? != 0 ];then
	echo usage: $0 CLASSNUM
        echo "Please enter number, not text "
	exit 1
fi
cls_num=$1

mkdir -p prototxts
python3 gen.py -s train -l data/lmdb_files/lmdb/lmdb_files_trainval_lmdb/ -c $cls_num > prototxts/train.prototxt
python3 gen.py -s test -l data/lmdb_files/lmdb/lmdb_files_test_lmdb/ -c $cls_num > prototxts/test.prototxt
python3 gen.py -s deploy -n -c $cls_num > prototxts/deploy.prototxt

