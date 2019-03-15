
cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
cd ..
data_root_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )

root_dir=/opt/movidius/ssd-caffe
cd $root_dir

redo=1

dataset_name="lmdb_files"
mapfile=$cur_dir/labelmap.prototxt
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
  python3 $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $data_root_dir/$dataset_name/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
