:<<!
[script description]: build a datastore for vanilla-knn-mt
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

CONFIDENCE_THRESHOLDS=(0.99 0.98 0.95 0.90 0.80 0.70 0.60 0.50)
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/test_medical

for threshold in "${CONFIDENCE_THRESHOLDS[@]}"
do
    echo "Building for confidence threshold $threshold"
    DATASTORE_SAVE_PATH="$PROJECT_PATH/datastore/chunk/test_medical/$threshold"
    mkdir -p $DATASTORE_SAVE_PATH

    CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/validate.py $DATA_PATH \
    --task translation \
    --path $BASE_MODEL \
    --model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
    --dataset-impl mmap \
    --valid-subset train \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 4096 \
    --bpe fastbpe \
    --user-dir $PROJECT_PATH/knnbox/models \
    --arch chunk_knn_mt@transformer_wmt19_de_en \
    --knn-mode build_datastore \
    --knn-datastore-path $DATASTORE_SAVE_PATH \
    --max-chunk-size 100 \
    --confidence-threshold $threshold
done


