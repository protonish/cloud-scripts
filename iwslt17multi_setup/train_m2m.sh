
ROOT="/home/ubuntu/cloud-wordemb/nishant/wordemb"

FAIRSEQ="${ROOT}/fairseq"
FAIRSCRIPTS="${FAIRSEQ}/scripts"
SPM_TRAIN="${FAIRSCRIPTS}/spm_train.py"
SPM_ENCODE="${FAIRSCRIPTS}/spm_encode.py"

USR_DIR="${ROOT}/embex"

# MMCR4NLP="${ROOT}/multilingual_data/mmcr4nlp"
# RAW_DATA="${MMCR4NLP}/iwslt2017/all"

EXP="${ROOT}/experiments"
####################################################
#### this is where the usable data will exist #####
####################################################
SETUP_NAME="iwslt17multi_setup"
SETUP="${EXP}/${SETUP_NAME}"
DEST="${SETUP}/data"
DEST_BIN="${DEST}/bin"
mkdir -p "${DEST}/bpe" "${DEST_BIN}"
####################################################


#########################
LANGS=(
    "de"
    "it"
    "nl"
    "ro"
)

CENTRE="en"

#########################

watermark(){
    date_time=$(date '+%d/%m/%Y %H:%M:%S')
    host_name=$(hostname)
    dns_name=$(dnsdomainname)

    echo "---------------------------------------"
    echo "============== new run ================"
    echo $date_time
    echo $host_name
    echo $dns_name
    echo ""
}

clear_checkpoint_dir(){
    ckpt=$1
    echo "removing checkpoints.. can't be undone"
    rm -rf $ckpt
    echo "done!"
}

train(){
    ############################
    watermark
    ############################

    DATA=$1
    SAVE_CKPT=$2
    LANG_PAIRS=$3
    EVAL_LANG_PAIRS=$4
    BSZ=$5
    USR="--user-dir $6"
    
    if [ -z "$7" ]
        then
            echo "No extra options"
            EXTRAS=""
        else
            EXTRAS="$7"
            echo "custom options set"
    fi

    # set ARCH argument -- important
    python3 ${FAIRSEQ}/fairseq_cli/train.py "${DATA}" \
    --fp16 --memory-efficient-fp16 \
    --log-format simple \
    --log-interval 200 \
    --save-dir "${SAVE_CKPT}" ${USR} \
    ${EXTRAS} \
    --task translation_multi_simple_epoch_eval \
    --langs "de,it,nl,ro,en" \
    --lang-pairs "${LANG_PAIRS}" \
    --eval-lang-pairs "${EVAL_LANG_PAIRS}" \
    --ignore-unused-valid-subsets \
    --sampling-method temperature \
    --sampling-temperature 1 \
    --max-tokens "${BSZ}" \
    --encoder-langtok tgt \
    --criterion label_smoothed_cross_entropy_agreement \
    --label-smoothing 0.1 \
    --optimizer adam \
    --lr-scheduler inverse_sqrt \
    --lr 5e-04 \
    --warmup-updates 6000 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --max-update 200000 \
    --train-subset "train" \
    --valid-subset "valid" \
    --update-freq 1 \
    --empty-cache-freq 50 \
    --save-interval-updates 5000 \
    --keep-interval-updates 20 \
    --keep-last-epochs 20 \
    --patience 10 \
    --arch emb_transformer_wmt_en_de \
    --encoder-layers 6 \
    --decoder-layers 6 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --ddp-backend no_c10d \
    --num-workers 6 \
    --best-checkpoint-metric ppl \
    --wandb-project "embex_iwslt17multi_m2m"

    # --share-all-embeddings \
    # --memory-efficient-fp16 \
    # --maximize-best-checkpoint-metric --> this saves the largest ppl
    # --fp16 --memory-efficient-fp16 
}

run_expt_m2m(){
    name=$1
    ckpt="${SETUP}/${name}/checkpoints"
    log_dir="${SETUP}/${name}/logs"
    mkdir -p $log_dir $ckpt "${SETUP}/${name}/results"

    data=$DEST_BIN
    
    lang_pairs="de-en,en-de,nl-en,en-nl,it-en,en-it,ro-en,en-ro"
    eval_lang_pairs="de-it,de-nl,de-ro,nl-de,nl-it,nl-ro,it-de,it-nl,it-ro,ro-de,ro-nl,ro-it"
    
    bsz=$2

    DEVICES=$3

    EXTRAS=$4
    echo "--> $EXTRAS"

    # setting wandb run name
    export WANDB_NAME=$name

    export CUDA_VISIBLE_DEVICES=$DEVICES
    train $data $ckpt $lang_pairs $eval_lang_pairs $bsz $USR_DIR "$EXTRAS" | tee -a "${log_dir}/train.log"
}



# call run expt

# 0. clear checkpoints
# clear_checkpoint_dir "${SETUP}/m2o_baseline/checkpoints"

## 1. baseline many 2 many
# run_expt_m2m "m2m_baseline" 4000 "0,1"

## enc latent emb
# run_expt_m2m "m2m_latent_emb" 5000 "0,1" "--encoder-latent-embeds "

## ann agreement
# run_expt_m2m "m2m_ann_emb_kl" 5000 "0,1" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 "

## rdrop agreement
# run_expt_m2m "m2m_rdrop_kl" 5000 "0,1" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 --no-knn-loss "

## ann equal weights
# run_expt_m2m "m2m_ann_emb_kl_eq_k" 5000 "0,1" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 --equal-weights-k "

## ann + no kl
# run_expt_m2m "m2m_ann_no_kl_eq_k" 5000 "0,1" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 --equal-weights-k --agreement-alpha -1 "

## ann + no kl + no latemt
# run_expt_m2m "m2m_ann_no_kl_no_latent_eq_k" 5000 "0,1" "--encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 --equal-weights-k --agreement-alpha -1 "

## ann + kl + eq
run_expt_m2m "m2m_cloud_ann_emb_kl_eq_k" 2048 "0,1,2,3,4,5,6,7" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 --equal-weights-k --no-kl-till-steps 200 --agreement-alpha 5 "


### ----------------------- for reference only -------------------------
## enc latent embeds al shared
# run_expt_m2o "m2o_enc_latent_emb_shared" 5000 "0,1" "--encoder-latent-embeds --share-all-embeddings" 

## enc simplical embeds all shared
# run_expt_m2o "m2o_enc_simp_emb_shared" 5000 "0,1" "--encoder-simplical-embeds --share-all-embeddings" 

## enc knn embeds
# run_expt_m2o "m2o_enc_knn_emb_shared_256_train_only" 5000 "0,1" "--encoder-knn-embeds --share-all-embeddings --encoder-embed-dim 256 --decoder-embed-dim 256 " 

## enc knn embeds
# run_expt_m2o "m2o_enc_knn_emb_shared_latent_train_only" 5000 "0,1" "--encoder-knn-embeds --share-all-embeddings --encoder-latent-embeds "

## enc knn embeds
# run_expt_m2o "m2o_enc_knn_emb_shared_latent_train_only_1" 5000 "0,1" "--encoder-knn-embeds --share-all-embeddings --encoder-latent-embeds "

## enc knn embeds
# run_expt_m2o "m2o_enc_knn_emb_shared_latent" 5000 "0,1" "--encoder-knn-embeds --share-all-embeddings --encoder-latent-embeds --encoder-knn-ratio 0.3 --knn-eval-also "
# run_expt_m2o "m2o_enc_knn_emb_shared_latent_0.5" 5000 "0,1" "--encoder-knn-embeds --share-all-embeddings --encoder-latent-embeds --encoder-knn-ratio 0.5 --knn-eval-also "


# ann debug
# run_expt_m2o "m2o_enc_ann_emb_shared_latent_debug" 5000 "0,1" "--encoder-knn-embeds --share-all-embeddings --encoder-latent-embeds --encoder-knn-ratio 0.9 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 "
# run_expt_m2o "m2o_enc_ann_emb_debug" 5000 "0,1" "--encoder-knn-embeds --share-all-embeddings --encoder-latent-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 "
# run_expt_m2o "m2o_enc_ann_emb_kl_debug" 5000 "0,1" "--encoder-knn-embeds --share-all-embeddings --encoder-latent-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 "

# rdrop test
# run_expt_m2o "m2o_enc_rdrop_kl_debug" 5000 "0,1" "--encoder-knn-embeds --share-all-embeddings --encoder-latent-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 --no-knn-loss "

# ann debug
# run_expt_m2o "m2o_enc_ann_emb_kl_stronger_k" 5000 "0,1" "--encoder-knn-embeds --share-all-embeddings --encoder-latent-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 "