# Setup
Dependencies
```
pip install --editable ./
pip install -r requirements.txt
```
Datasets: [iwslt](https://drive.google.com/file/d/17AuW_aXG4bARmH7nUglV2nf2snryoDan/view?usp=drive_link), [europarl](https://drive.google.com/file/d/1LEhUc81iMZVyRBBl6l2hz2BOQV7nuSNJ/view?usp=drive_link)

Checkpoints: [iswlt](https://drive.google.com/file/d/1xJn8CpXiF9ecCH7yj9C_cZeVVroVq-_S/view?usp=drive_link), [europarl](https://drive.google.com/file/d/1j7ymMJxw4eH8X-50o0KTzWE-jpcInQ1z/view?usp=drive_link)

# Generation
1. Generate a "pivot dataset", which creates pivot translations to build ensembles.
```
bash build_pivot_datasets.sh iwslt_zero [ckpt_path]
```
3. Run the ensemble decoding script for IWSLT
```
python3 fairseq_cli/multipivot_generate.py [dataset_bin] --path [ckpt_path] \
    --task translation_multi_pivot_epoch --gen-subset test  \
    --source-lang it --target-lang nl --pivot-langs en ro \
    --remove-bpe --batch-size 300  --langs en,ro,it,nl   --lang-pairs en-ro,ro-en,en-it,it-en,en-nl,nl-en   \
    --pivot-datadir=[pivot_dir] \
    --ensemble_mode=[ensemble_mode] \
    --encoder-langtok "tgt"  --max-len-a 1 --max-len-b 100 --sacrebleu --user-dir fs_plugins \
    --beam 1  \
    --max-sentences=300 \
    --decoder-langtok \
```
Here, `ensemble_mode` can be one of the following values:
1. `averaging` for word-level averaging ensembles
2. `word-voting` for word-level voting ensembles
3. `voting` for EBBS
   
# Evaluation
To evaluate, we apply detokenization and use sacrebleu.
```
detokenizer=mosesdecoder/scripts/tokenizer/detokenizer.perl

grep ^H- generate.out | sort -V | cut -f 3 > H.tmp_
grep ^T- generate.out | sort -V | cut -f 2 > T.tmp_

sed -e "s/@@ //g" T.tmp_   | sed -e "s/@@$//g" | sed -e "s/&apos;/'/g" -e 's/&#124;/|/g' -e "s/&amp;/&/g" -e 's/&lt;/>/g' -e \
    's/&gt;/>/g' -e 's/&quot;/"/g' -e 's/&#91;/[/g' -e 's/&#93;/]/g' -e 's/ - /-/g' | sed -e "s/ '/'/g" | sed -e "s/ '/'/g" | \
    sed -e "s/%- / -/g" | sed -e "s/ -%/- /g" | perl -nle 'print ucfirst' > T.tmp_.1

sed -e "s/@@ //g" H.tmp_   | sed -e "s/@@$//g" | sed -e "s/&apos;/'/g" -e 's/&#124;/|/g' -e "s/&amp;/&/g" -e 's/&lt;/>/g' -e \
    's/&gt;/>/g' -e 's/&quot;/"/g' -e 's/&#91;/[/g' -e 's/&#93;/]/g' -e 's/ - /-/g' | sed -e "s/ '/'/g" | sed -e "s/ '/'/g" | \
    sed -e "s/%- / -/g" | sed -e "s/ -%/- /g" | perl -nle 'print ucfirst' > H.tmp_.1

cat T.tmp_.1 | $detokenizer -l [language] > T.detok.tmp_
cat H.tmp_.1 | $detokenizer -l [language] > H.detok.tmp_

cat T.detok.tmp_ | mosesdecoder/scripts/recaser/detruecase.perl > T.detok.tmp_.1
cat H.detok.tmp_ | mosesdecoder/scripts/recaser/detruecase.perl > H.detok.tmp_.1

sacrebleu T.detok.tmp_.1  -w 2 -i H.detok.tmp_.1  2>&1 | tee bleu.txt
```
