dataset=$1  # iwslt_zero or europarl_zero
CKPT=$2

if [ $dataset == "iwslt_zero" ]; then
    dataset_bin="data-bin/iwslt/"
    langs="en,ro,it,nl"
    lang_pairs="en-ro,ro-en,en-it,it-en,en-nl,nl-en"
    lang_pair_list=("ro-it" "ro-nl" "it-ro" "it-nl" "nl-ro" "nl-it")
    langs_list=("en" "ro" "it" "nl")
    name=iwslt
elif [ $dataset == "europarl_zero" ]; then
    dataset_bin="data-bin/europarl_no_overlap/"
    langs="en,da,de,es,fi,fr,it,nl,pt"
    lang_pairs="en-da,en-de,en-es,en-fi,en-fr,en-it,en-nl,en-pt,da-en,de-en,es-en,fi-en,fr-en,it-en,nl-en,pt-en"
    lang_pair_list=("da-de" "da-es" "da-fi" "da-fr" "da-it" "da-nl" "da-pt" "de-da" "de-es" "de-fi" "de-fr" "de-it" \
                    "de-nl" "de-pt" "es-da" "es-de" "es-fi" "es-fr" "es-it" "es-nl" "es-pt" "fi-da" "fi-de" "fi-es" \
                    "fi-fr" "fi-it" "fi-nl" "fi-pt" "fr-da" "fr-de" "fr-es" "fr-fi" "fr-it" "fr-nl" "fr-pt" "it-da" \
                    "it-de" "it-es" "it-fi" "it-fr" "it-nl" "it-pt" "nl-da" "nl-de" "nl-es" "nl-fi" "nl-fr" "nl-it" \
                    "nl-pt" "pt-da" "pt-de" "pt-es" "pt-fi" "pt-fr" "pt-it" "pt-nl")
    langs_list=("en" "da" "de" "es" "fi" "fr" "it" "nl" "pt")
    name=europarl_no_overlap
else
    echo Not implemented
fi

OUTPUT_PATH=`dirname $CKPT`/pivot_datasets
rm -rf $OUTPUT_PATH
mkdir -p $OUTPUT_PATH

for src in "${langs_list[@]}"; do
    for trg in "${langs_list[@]}"; do
        for pvt in "${langs_list[@]}"; do
            if [[ "$src" != "en" &&  "$trg" != "en" && "$src" != "$trg" && "$trg" != "$pvt" && "$src" != "$pvt" ]]; then

                echo "$src $pvt $trg"

                pivot_tmp_dir=$OUTPUT_PATH/$src-$pvt-$trg
                mkdir -p $pivot_tmp_dir

                SRC_2_PVT_DIR=$pivot_tmp_dir/pivot_${src}2${pvt}
                PVT_2_TRG_DIR=$pivot_tmp_dir/pivot_${pvt}2${trg}

                mkdir ${SRC_2_PVT_DIR}
                mkdir ${PVT_2_TRG_DIR}

                # dummy src2trg translation
                # only used to obtain src-trg pairs
                echo Source to Pivot translation
                python3 fairseq_cli/generate.py  $dataset_bin  --path $CKPT \
                    --task translation_multi_simple_epoch   --gen-subset test  --source-lang $src \
                    --target-lang $trg --batch-size 300  --langs $langs   --lang-pairs $lang_pairs   \
                    --encoder-langtok "tgt"  --max-len-a 1 --max-len-b 100 --sacrebleu --user-dir fs_plugins \
                    --beam 5 \
                    --decoder-langtok \
                    > $pivot_tmp_dir/${src}2${trg}.out

                grep ^T- $pivot_tmp_dir/${src}2${trg}.out | sort -V | cut -f 2 > ${SRC_2_PVT_DIR}/${src}2${pvt}.${pvt}  # Not pvt, but we don't evaluate anyways
                grep ^S- $pivot_tmp_dir/${src}2${trg}.out | sort -V | cut -f 2 > ${SRC_2_PVT_DIR}/${src}2${pvt}.${src}

                grep ^T- $pivot_tmp_dir/${src}2${trg}.out | sort -V | cut -f 2 > ${PVT_2_TRG_DIR}/${pvt}2${trg}.${trg}  # Save as reference for en2trg translation

                # preprocess for src2pvt dataset
                bin_path=${SRC_2_PVT_DIR}/data-bin
                rm -rf $bin_path
                python3 fairseq_cli/preprocess.py \
                    --source-lang $src --target-lang $pvt \
                    --testpref ${SRC_2_PVT_DIR}/${src}2${pvt} \
                    --destdir $bin_path \
                    --workers 128 --joined-dictionary \
                    --srcdict data-bin/${name}/dict.${src}.txt

                # src2pvt translation
                echo Source to English translation
                python3 fairseq_cli/generate.py  $bin_path  --path $CKPT \
                    --task translation_multi_simple_epoch   --gen-subset test  --source-lang $src \
                    --target-lang $pvt --batch-size 300  --langs $langs   --lang-pairs $lang_pairs   \
                    --encoder-langtok "tgt"  --max-len-a 1 --max-len-b 100 --sacrebleu --user-dir fs_plugins \
                    --beam 5  \
                    --decoder-langtok \
                    > $pivot_tmp_dir/${src}2${pvt}.out

                # preprocess for pvt2trg dataset
                grep ^H- $pivot_tmp_dir/${src}2${pvt}.out | sort -V | cut -f 3 > ${PVT_2_TRG_DIR}/${pvt}2${trg}.${pvt}

                bin_path=${PVT_2_TRG_DIR}/data-bin
                rm -rf $bin_path
                python3 fairseq_cli/preprocess.py \
                    --source-lang ${pvt} --target-lang ${trg} \
                    --testpref ${PVT_2_TRG_DIR}/${pvt}2${trg} \
                    --destdir $bin_path \
                    --workers 128 --joined-dictionary \
                    --srcdict data-bin/${name}/dict.${pvt}.txt
            fi
        done
    done
done
