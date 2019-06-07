config=$1

for pair in dutch--west-frisian dutch--yiddish english--murrinhpatha english--north-frisian ; do
    langpair=/mnt/data/bpop/sigmorphon/task-1/gate-sparse/$pair
    name=$langpair/$( basename $config .yml )
    modeldir=$name-models
    mkdir -p $modeldir
    python train.py -config $config -lang_location src tgt inflection -data $langpair/inf-data -save_model $modeldir/model -log_file $name.log
done
