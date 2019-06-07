modeldir=$1 # actual path to the models, unlike in translate-task-3.sh
evalset=$2 #dev or test
preddir=$3

for langdir in /mnt/data/bpop/sigmorphon/task-3/mono/high/* ; do
    language=$(basename $langdir )
    echo $language
    datapath=/mnt/data/bpop/inflection/conll2018/task1/everything/$(basename $langdir)-$evalset
    modelrun1=$( python best_model.py $modeldir/run1*.pt )
    modelrun2=$( python best_model.py $modeldir/run2*.pt )
    modelrun3=$( python best_model.py $modeldir/run3*.pt )
    predpath=$preddir/$language.pred
    finalout=$predpath.out
    python translate.py -model $modelrun1 $modelrun2 $modelrun3 -avg_raw_probs -corpora $datapath -output $predpath -beam_size 5 -gpu 0
    python pred2sigmorphon.py $datapath $predpath > $finalout
    python /mnt/data/bpop/inflection/conll2018/task1/evaluation/evalm.py --guess $finalout --gold $datapath --task 1
done
