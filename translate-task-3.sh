modeldir=$1
run=$2
evalset=$3 #dev or test

for langdir in /mnt/data/bpop/sigmorphon/task-3/mono/high/* ; do
    datapath=/mnt/data/bpop/inflection/conll2018/task1/everything/$(basename $langdir)-$evalset
    model=$( python best_model.py $langdir/$modeldir/$run*.pt )
    echo $model
    predpath=$langdir/$run-$modeldir-$evalset.pred
    finalout=$predpath.out
    python translate.py -model $model -corpora $datapath -output $predpath -beam_size 5 -gpu 0
    python pred2sigmorphon.py $datapath $predpath > $finalout
    python /mnt/data/bpop/inflection/conll2018/task1/evaluation/evalm.py --guess $finalout --gold $datapath --task 1
done
