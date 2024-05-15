#!/bin/bash

ratio=${1:-0.5}
vocab_size=50000
base_path=examples/roberta
data_path=wiki-data


echo install dependencies
bash $base_path/s0_install_deps.sh

echo "1. split by" $rato
for split in train valid
do
    time python3 $base_path/split_by_ratio.py wiki-data/all_dedup_shuf.raw.$split wiki-data/all_dedup_shuf.split$ratio.$split $ratio &
done

echo "Wait for step 1"
wait


# STEP 2: GPT encode
echo "2.1 tokenization"
for seg in foo bar
do
    for split in train valid
    do  
        bash $base_path/s1_clean_and_tok.sh en \
           $data_path/all_dedup_shuf.split$ratio.$split.$seg   $data_path/all_dedup_shuf.split$ratio.$split.$seg.tok-1 &
    done
done

echo "Wait for step 2.1"
wait

echo "2.2 learn SPM"
cat $data_path/all_dedup_shuf.split$ratio.train.foo.tok-1 $data_path/all_dedup_shuf.split$ratio.train.bar.tok-1 \
        > $data_path/all_dedup_shuf.split$ratio.train.all.tok-1 
bash $base_path/s2_learn_spm.sh     en     $data_path/all_dedup_shuf.split$ratio.train.all.tok-1 $vocab_size $data_path
echo "learn SPM Done"
cut -f1 $data_path/en.sp.vocab | tail -n +4 | sed "s/$/ 100/g" > $data_path/wiki.foobar.dict.txt
tail -n +2 $data_path/wiki.foobar.dict.txt > $data_path/wiki.foobar.dict.1.txt


echo "3. Apply SPM"
for seg in foo bar
do
    for split in train valid
    do  
        mkdir -p tmp_${split}_${seg}_${ratio}
        rm tmp_${split}_${seg}_${ratio}/*
        time bash $base_path/s3_apply_spm.sh $data_path/en.sp.model $data_path/all_dedup_shuf.split$ratio.$split.$seg.tok-1 \
            $data_path/all_dedup_shuf.split$ratio.spm.$split.$seg  tmp_${split}_${seg}_${ratio}  &
    done
done

wait


echo "4. Fairseq Preprocess"
rm -rf data-bin/wiki_foobar_ratio$ratio 
python3 fairseq_cli/preprocess.py \
    --source-lang foo --target-lang bar \
    --trainpref $data_path/all_dedup_shuf.split$ratio.spm.train \
    --validpref $data_path/all_dedup_shuf.split$ratio.spm.valid \
    --testpref $data_path/all_dedup_shuf.split$ratio.spm.valid \
    --destdir data-bin/wiki_foobar_ratio$ratio \
    --workers 128 --joined-dictionary \
    --srcdict $data_path/wiki.foobar.dict.1.txt


echo "all DONE!"
# wait
# echo "Wait for step 3"

