#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="eval1"
#test_sets="eval1 eval2 eval3 train_dev train_nodup_sp"
#test_sets="eval1"

asr_config=conf/train_asr_rnn.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

# NOTE: The default settings require 4 GPUs with 32 GB memory
./asr.sh \
    --ngpu 1 \
    --lang jp \
    --token_type word \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --expdir "exp_20221125_ctc_none" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/train_nodev/text" "$@"
    # --inference_asr_model "5epoch.pth" \ 
    # --expdir "exp_20221006" \
    # --inference_nj 64 \
    # --gpu_inference true \
    # --use_lm false \
    # --stop_stage 1 \