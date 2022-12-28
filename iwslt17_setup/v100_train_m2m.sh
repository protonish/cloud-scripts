
ROOT="/local-scratch/nishant/wordemb"

FAIRSEQ="${ROOT}/fairseq"
FAIRSCRIPTS="${FAIRSEQ}/scripts"
SPM_TRAIN="${FAIRSCRIPTS}/spm_train.py"
SPM_ENCODE="${FAIRSCRIPTS}/spm_encode.py"

USR_DIR="${ROOT}/embex"

MMCR4NLP="${ROOT}/multilingual_data/mmcr4nlp"
RAW_DATA="${MMCR4NLP}/iwslt2017/all"

EXP="${ROOT}/experiments"
####################################################
#### this is where the usable data will exist #####
####################################################
SETUP_NAME="iwslt17_setup"
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
    --log-interval 100 \
    --save-dir "${SAVE_CKPT}" ${USR} \
    ${EXTRAS} \
    --task translation_multi_simple_epoch_eval \
    --langs "de,it,nl,ro,en" \
    --lang-pairs "${LANG_PAIRS}" \
    --eval-lang-pairs "${EVAL_LANG_PAIRS}" \
    --sampling-method temperature \
    --sampling-temperature 1 \
    --max-tokens "${BSZ}" \
    --encoder-langtok tgt \
    --criterion label_smoothed_cross_entropy_agreement \
    --label-smoothing 0.1 \
    --optimizer adam \
    --lr-scheduler inverse_sqrt \
    --lr 5e-04 \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --max-update 150000 \
    --train-subset "train" \
    --update-freq 2 \
    --empty-cache-freq 100 \
    --save-interval-updates 5000 \
    --keep-interval-updates 2 \
    --keep-last-epochs 5 \
    --patience 10 \
    --arch emb_transformer \
    --encoder-layers 5 \
    --decoder-layers 5 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --ddp-backend no_c10d \
    --num-workers 2 \
    --best-checkpoint-metric ppl \
    --wandb-project "embex_iwslt17_m2m"

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

############ bsz ##############
#   4096 x 8 --> 32000
#   5000 x 2x2 --> 20000 -- won't fit on v100
#   3000 x 8x2 --> 48000
###############################


# call run expt

## ann agreement
# run_expt_m2m "m2m_ann_emb_kl" 5000 "0,1" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 "

## no knn agreement
# run_expt_m2m "m2m_noknn_kl" 5000 "0,1" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 --no-knn-loss "

## equal weight knn kl
run_expt_m2m "m2m_ann_emb_kl_eq_k" 4096 "0,1,2,3,4,5,6,7" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 --equal-weights-k "


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