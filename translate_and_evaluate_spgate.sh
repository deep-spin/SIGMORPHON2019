for expdir in $@ ; do
    pair=$( basename $expdir )
    dev=/mnt/data/bpop/sigmorphon/2019/task1/$pair/*dev
    for config in spgate-sp-sp ; do
        echo $pair
        echo $config
        model=$( python best_model.py $expdir/$config/*.pt )
        echo $model
        rawout=$expdir/$config.pred
        final=$rawout.out
        python translate.py -model $model -corpora $dev -beam_size 5 -output $rawout
        python pred2sigmorphon.py $dev $rawout > $final
        python /mnt/data/bpop/sigmorphon/2019/evaluation/evaluate_2019_task1.py --reference $dev --output $final
    done
done
