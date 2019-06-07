modeldir=$1
evalset=$2 #dev or test

for langdir in /mnt/data/bpop/sigmorphon/task-3/mono/high/* ; do
    datapath=/mnt/data/bpop/inflection/conll2018/task1/everything/$(basename $langdir)-$evalset
    modelrun1=$( python best_model.py $langdir/$modeldir/run1*.pt )
    modelrun2=$( python best_model.py $langdir/$modeldir/run2*.pt )
    modelrun3=$( python best_model.py $langdir/$modeldir/run3*.pt )
    echo $modelrun1 $modelrun2 $modelrun3
    predpath=$langdir/ensemble-$modeldir-$evalset.pred
    finalout=$predpath.out
    python translate.py -model $modelrun1 $modelrun2 $modelrun3 -avg_raw_probs -corpora $datapath -output $predpath -beam_size 5 -gpu 0
    python pred2sigmorphon.py $datapath $predpath > $finalout
    python /mnt/data/bpop/inflection/conll2018/task1/evaluation/evalm.py --guess $finalout --gold $datapath --task 1
done
