for expdir in $@ ; do
    pair=$( basename $expdir )
    dev=/mnt/data/bpop/sigmorphon/2019/task1/$pair/*dev
    echo $pair
    model=$( python best_config.py $expdir )
    echo $model
    rawout=$expdir/dev.pred
    final=$rawout.out
    python translate.py -model $model -corpora $dev -beam_size 5 -output $rawout -gpu 0 -n_best 2 -attn_path $expdir/attn.pt -probs_path $expdir/probs.pt
    python pred2sigmorphon.py $dev $rawout > $final
    python /mnt/data/bpop/sigmorphon/2019/evaluation/evaluate_2019_task1.py --reference $dev --output $final
done
