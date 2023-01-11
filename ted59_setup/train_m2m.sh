
ROOT="/home/ubuntu/nishant/wordemb"

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
SETUP_NAME="ted59_setup"
SETUP="${EXP}/${SETUP_NAME}"
DEST="${SETUP}/data"
DEST_BIN="${DEST}/bin"
mkdir -p "${DEST}/bpe" "${DEST_BIN}"
####################################################

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
    --langs "en,es,ptbr,fr,ru,he,ar,ko,zhcn,it,ja,zhtw,nl,ro,tr,de,vi,pl,pt,bg,el,fa,sr,hu,hr,uk,cs,id,th,sv,sk,sq,lt,da,my,sl,mk,frca,fi,hy,hi,nb,ka,mn,et,ku,gl,mr,zh,ur,eo,ms,az,ta,bn,kk,be,eu,bs" \
    --lang-pairs "${LANG_PAIRS}" \
    --eval-lang-pairs "${EVAL_LANG_PAIRS}" \
    --sampling-method temperature \
    --sampling-temperature 5 \
    --max-tokens "${BSZ}" \
    --criterion label_smoothed_cross_entropy_agreement \
    --label-smoothing 0.1 \
    --optimizer adam \
    --lr-scheduler inverse_sqrt \
    --lr 1.5 \
    --warmup-init-lr 0.001 \
    --warmup-updates 20000 \
    --clip-norm 2 \
    --dropout 0.2 \
    --max-update 200000 \
    --train-subset "train" \
    --update-freq 2 \
    --empty-cache-freq 100 \
    --save-interval-updates 5000 \
    --keep-interval-updates 10 \
    --keep-last-epochs 10 \
    --patience 10 \
    --arch emb_transformer_wmt_en_de \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --attention-dropout 0.2 \
    --activation-dropout 0.2 \
    --ddp-backend no_c10d \
    --num-workers 4 \
    --best-checkpoint-metric ppl \
    --wandb-project "embex_ted59_m2m"

    # --share-all-embeddings \
    # --memory-efficient-fp16 \
    # --maximize-best-checkpoint-metric --> this saves the largest ppl
    # --fp16 --memory-efficient-fp16 

    # --weight-decay 0.0001 \
}

run_expt_m2m(){
    name=$1
    ckpt="${SETUP}/${name}/checkpoints"
    log_dir="${SETUP}/${name}/logs"
    mkdir -p $log_dir $ckpt "${SETUP}/${name}/results"

    data=$DEST_BIN
    
    lang_pairs="es-en,en-es,ptbr-en,en-ptbr,fr-en,en-fr,ru-en,en-ru,he-en,en-he,ar-en,en-ar,ko-en,en-ko,zhcn-en,en-zhcn,it-en,en-it,ja-en,en-ja,zhtw-en,en-zhtw,nl-en,en-nl,ro-en,en-ro,tr-en,en-tr,de-en,en-de,vi-en,en-vi,pl-en,en-pl,pt-en,en-pt,bg-en,en-bg,el-en,en-el,fa-en,en-fa,sr-en,en-sr,hu-en,en-hu,hr-en,en-hr,uk-en,en-uk,cs-en,en-cs,id-en,en-id,th-en,en-th,sv-en,en-sv,sk-en,en-sk,sq-en,en-sq,lt-en,en-lt,da-en,en-da,my-en,en-my,sl-en,en-sl,mk-en,en-mk,frca-en,en-frca,fi-en,en-fi,hy-en,en-hy,hi-en,en-hi,nb-en,en-nb,ka-en,en-ka,mn-en,en-mn,et-en,en-et,ku-en,en-ku,gl-en,en-gl,mr-en,en-mr,zh-en,en-zh,ur-en,en-ur,eo-en,en-eo,ms-en,en-ms,az-en,en-az,ta-en,en-ta,bn-en,en-bn,kk-en,en-kk,be-en,en-be,eu-en,en-eu,bs-en,en-bs"
    underscore_lang_pairs="es_en,en_es,pt-br_en,en_pt-br,fr_en,en_fr,ru_en,en_ru,he_en,en_he,ar_en,en_ar,ko_en,en_ko,zh-cn_en,en_zh-cn,it_en,en_it,ja_en,en_ja,zh-tw_en,en_zh-tw,nl_en,en_nl,ro_en,en_ro,tr_en,en_tr,de_en,en_de,vi_en,en_vi,pl_en,en_pl,pt_en,en_pt,bg_en,en_bg,el_en,en_el,fa_en,en_fa,sr_en,en_sr,hu_en,en_hu,hr_en,en_hr,uk_en,en_uk,cs_en,en_cs,id_en,en_id,th_en,en_th,sv_en,en_sv,sk_en,en_sk,sq_en,en_sq,lt_en,en_lt,da_en,en_da,my_en,en_my,sl_en,en_sl,mk_en,en_mk,fr-ca_en,en_fr-ca,fi_en,en_fi,hy_en,en_hy,hi_en,en_hi,nb_en,en_nb,ka_en,en_ka,mn_en,en_mn,et_en,en_et,ku_en,en_ku,gl_en,en_gl,mr_en,en_mr,zh_en,en_zh,ur_en,en_ur,eo_en,en_eo,ms_en,en_ms,az_en,en_az,ta_en,en_ta,bn_en,en_bn,kk_en,en_kk,be_en,en_be,eu_en,en_eu,bs_en,en_bs"
    eval_lang_pairs="ar-fr,fr-ar,ru-uk,uk-ru"
    
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
# run_expt_m2m "m2m_baseline" 5000 "0,1"

## enc latent emb
# run_expt_m2m "m2m_latent_emb" 5000 "0,1" "--encoder-latent-embeds "

## ann agreement
# run_expt_m2m "m2m_ann_emb_kl" 5000 "0,1" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 "

## rdrop agreement
# run_expt_m2m "m2m_rdrop_kl" 5000 "0,1" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 --no-knn-loss "

## ann equal weights
run_expt_m2m "m2m_aharoni_ann_emb_kl_eq_k" 5000 "0,1,2,3,4,5,6,7" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 400 --cache-scann --knn-value 3 --agreement-warmup 100 --equal-weights-k --no-kl-till-steps 25000 "


# local test -- running
# run_expt_m2m "m2m_ann_emb_kl_eq_k" 5000 "0,1" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 400 --cache-scann --knn-value 3 --agreement-warmup 100 --equal-weights-k --no-kl-till-steps 20000 "

## ann + no kl
# run_expt_m2m "m2m_ann_no_kl_eq_k" 5000 "0,1" "--encoder-latent-embeds --encoder-knn-embeds --encoder-knn-ratio 0.7 --knn-type approx --use-scann --index-trigger 300 --cache-scann --knn-value 3 --agreement-warmup 100 --equal-weights-k --agreement-alpha -1 "



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