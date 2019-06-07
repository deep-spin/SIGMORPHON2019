for config in $@ ; do
    expdir=~/sigmorphon/task-3/
    name=$(basename $config .yml )
    savepath=$expdir/$name-models
    mkdir -p $savepath
    python train.py -config $config -data $expdir/medium-inf-data -save_model $savepath/model -log_file $expdir/$name.log -lang_location src tgt inflection
done
