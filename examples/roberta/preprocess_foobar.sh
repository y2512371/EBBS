#!/bin/bash

# STEP 1: split 
echo "1. split"
if [ $# -eq 0]
then
    ratio=0.5
else
    ratio=$1
if

base_path=examples/roberta
for split in train valid
do
    time python3 $base_path/split_by_ratio.py wiki-data/all_dedup_shuf.raw.$split wiki-data/all_dedup_shuf.split$ratio.$split $ratio && echo split $split "done" &
done

echo "Wait for step 1"
wait

# STEP 2: GPT encode
echo "2. GPT encode"
for seg in foo bar
do
    for split in train valid
    do
        time python3 $base_path/multiprocessing_bpe_encoder.py \
            --encoder-json gpt2_bpe/encoder.json \
            --vocab-bpe gpt2_bpe/vocab.bpe \
            --inputs wiki-data/all_dedup_shuf.split$ratio.$split.$seg \
            --outputs wiki-data/all_dedup_shuf.split$ratio.bpe0.$split.$seg \
            --keep-empty \
            --workers 128  && echo encode $split "done" &
    done
done

echo "Wait for step 2"
wait

echo "3. GPT convert BPE"
for seg in foo bar
do
    for split in train valid
    do
        time python3 $base_path/gptbpe2fairseqbpe_mp.py wiki-data/all_dedup_shuf.split$ratio.bpe0.$split.$seg  \
            wiki-data/all_dedup_shuf.split$ratio.bpe.$split.$seg && echo convert split $split "done" &
    done
done

wait
echo "Wait for step 3"