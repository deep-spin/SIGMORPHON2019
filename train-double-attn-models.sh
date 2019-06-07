config=$1

for langpair in /mnt/data/bpop/sigmorphon/task-1/double-attn-sparse/* ; do
    name=$langpair/$( basename $config .yml )
    modeldir=$name-models
    python train.py -config $config -lang_location src tgt inflection -data $langpair/inf-data -save_model $modeldir/model -log_file $name.log
done
