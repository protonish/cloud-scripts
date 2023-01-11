
ROOT="/local-scratch/nishant/wordemb"

FAIRSEQ="${ROOT}/fairseq"
FAIRSCRIPTS="${FAIRSEQ}/scripts"
SPM_TRAIN="${FAIRSCRIPTS}/spm_train.py"
SPM_ENCODE="${FAIRSCRIPTS}/spm_encode.py"

# MMCR4NLP="${ROOT}/experiments/ted59_setup/raw"
RAW_DATA="${ROOT}/experiments/ted59_setup/raw"

EXP="${ROOT}/experiments"
####################################################
#### this is where the usable data will exists #####
####################################################
SETUP_NAME="ted59_setup"
SETUP="${EXP}/${SETUP_NAME}"
DEST="${SETUP}/data"
DEST_BIN="${DEST}/bin"
mkdir -p "${DEST}/bpe" "${DEST_BIN}"
####################################################


#########################
LANGS=(
    "es"
    ko
    zh-cn
    it
    ja
    zh-tw
    nl
    ro
    tr
    de
    vi
    pl
    pt
    bg
    el
    fa
    sr
    hu
    hr
    uk
    cs
    id
    th
    sv
    sk
    sq
    lt
    da
    my
    sl
    mk
    fr-ca
    fi
    hy
    hi
    nb
    ka
    mn
    et
    ku
    gl
    mr
    zh
    ur
    eo
    ms
    az
    ta
    bn
    kk
    be
    eu
    bs
    pt-br
    fr
    ru
    he
    ar
    )

CENTRE="en"

TRAIN_PREF="train"
DEV_PREF="dev"
TEST_PREF="test"
#########################



learn_spm_joint(){
    sw_type=$1
    num_bpe=$2
    train_max_len=${TRAIN_MAXLEN}

    RAW_TRAIN_FILES=$(
        echo ${RAW_DATA}/ru_en/${TRAIN_PREF}.en | tr "\n" ",";
        for LANG in "${LANGS[@]}"; do 
            echo ${RAW_DATA}/${LANG}_${CENTRE}/${TRAIN_PREF}.${LANG};
            # echo ${RAW_DATA}/${CENTRE}_${LANG}/${TRAIN_PREF}.${CENTRE};
            if [[ $LANG == ${LANGS[-1]} ]]; then
                echo "${RAW_DATA}/${LANG}_${CENTRE}/${TRAIN_PREF}.${CENTRE}";
                # echo ${RAW_DATA}/${CENTRE}_${LANG}/${TRAIN_PREF}.${LANG};
            fi
        done | tr "\n" ",")

    USER_SYMBOLS=$(
        echo "__en__" | tr "\n" ",";
        for LANG in "${LANGS[@]}"; do 
            echo "__${LANG}__";
        done | tr "\n" ",")


    echo "learning joint BPE over ${RAW_TRAIN_FILES} .."

    # echo "${USER_SYMBOLS}"

    python "${SPM_TRAIN}" \
        --input=${RAW_TRAIN_FILES} \
        --model_prefix="${DEST}/bpe/spm.${num_bpe}" \
        --vocab_size=${num_bpe} \
        --character_coverage=1.0 \
        --model_type="${sw_type}" \
        --input_sentence_size=2000000 \
        --shuffle_input_sentence=true \
        --user_defined_symbols="${USER_SYMBOLS}"

}

spm_encode_pair_parallel(){
    # prefix=$1
    src=$1
    tgt=$2
    subset=$3

    echo "-- spm encoding ${src} and ${tgt} parallel: ${subset} subset .."
    python3 ${SPM_ENCODE} \
        --model "${DEST}/bpe/spm.${BPESIZE}.model" \
        --output_format=piece \
        --inputs "${RAW_DATA}/${src}_${tgt}/${subset}.${src}" "${RAW_DATA}/${src}_${tgt}/${subset}.${tgt}" \
        --outputs "${DEST}/bpe/spm.${subset}.${src}-${tgt}.${src}" "${DEST}/bpe/spm.${subset}.${src}-${tgt}.${tgt}"

    echo "finished encoding !"

    DICT="jointdict.txt"

    echo "Generating joined dictionary for all languages based on SPM.."
    # strip the first three special tokens and append fake counts for each vocabulary
    tail -n +4 "${DEST}/bpe/spm.${BPESIZE}.vocab" | cut -f1 | sed 's/$/ 100/g' > "${DEST}/bpe/${DICT}"
}


spm_encode_pair(){
    src=$1
    tgt=$2
    subset=$3

    echo "-- spm encoding ${src} and ${tgt} ${subset} subset .."
    python3 ${SPM_ENCODE} \
        --model "${DEST}/bpe/spm.${BPESIZE}.model" \
        --output_format=piece \
        --inputs="${RAW_DATA}/${subset}.${src}-${tgt}.${src}" \
        --outputs="${DEST}/bpe/spm.${subset}.${src}-${tgt}.${src}"

    python3 ${SPM_ENCODE} \
        --model "${DEST}/bpe/spm.${BPESIZE}.model" \
        --output_format=piece \
        --inputs="${RAW_DATA}/${subset}.${src}-${tgt}.${tgt}" \
        --outputs="${DEST}/bpe/spm.${subset}.${src}-${tgt}.${tgt}"

    echo "finished encoding !"

    DICT="jointdict.txt"

    echo "Generating joined dictionary for all languages based on SPM.."
    # strip the first three special tokens and append fake counts for each vocabulary
    tail -n +4 "${DEST}/bpe/spm.${BPESIZE}.vocab" | cut -f1 | sed 's/$/ 100/g' > "${DEST}/bpe/${DICT}"
}

spm_encode_all(){
    SUBSETS=(
        $TEST_PREF
    )

    for LANG in "${LANGS[@]}"; do
        for SUBSET in "${SUBSETS[@]}"; do
            spm_encode_pair_parallel $LANG $CENTRE $SUBSET
            spm_encode_pair_parallel $CENTRE $LANG $SUBSET
        done
    done

    echo "finsihed encoding all language pairs!"
}

spm_encode_zero_shot(){
    SUBSETS=(
        $DEV_PREF
        $TEST_PREF
    )
    for SUBSET in "${SUBSETS[@]}"; do
        spm_encode_pair_parallel "ar" "fr" $SUBSET
        spm_encode_pair_parallel "fr" "ar" $SUBSET
        spm_encode_pair_parallel "uk" "ru" $SUBSET
        spm_encode_pair_parallel "ru" "uk" $SUBSET
    done

    echo "finsihed encoding zero language pairs!"
}

build_zero_shot(){
    echo "preparing zero shot data .."
    mkdir -p "${DEST}/bpe/zero"
    
    SUBSETS=(
        $DEV_PREF
        $TEST_PREF1
    )

    for SRC in "${LANGS[@]}"; do
        for TGT in "${LANGS[@]}"; do
            if [ ! ${SRC} = ${TGT} ]; then
                for SUBSET in "${SUBSETS[@]}"; do
                    echo "copying ${SRC} --> ${TGT} ${SUBSET}"
                    cp "${DEST}/bpe/spm.${SUBSET}.${SRC}-${CENTRE}.${SRC}" "${DEST}/bpe/zero/spm.${SUBSET}.${SRC}-${TGT}.${SRC}"
                    cp "${DEST}/bpe/spm.${SUBSET}.${TGT}-${CENTRE}.${TGT}" "${DEST}/bpe/zero/spm.${SUBSET}.${SRC}-${TGT}.${TGT}"
                done
            fi
        done
    done
}

binarize(){
    src=$1
    tgt=$2
    srcdict=$3
    tgtdict=$4
    train=$5
    valid=$6
    test=$7

    if [ -z "$8" ]
        then
            echo ""
            echo "supervised direction -- binarizing $src and $tgt"
            fairseq-preprocess \
            --source-lang $src --target-lang $tgt \
            --trainpref $train \
            --validpref $valid \
            --testpref $test \
            --thresholdsrc 0 --thresholdtgt 0 \
            --destdir $DEST_BIN \
            --srcdict ${srcdict} --tgtdict ${tgtdict}
        else
            echo ""
            echo "zero shot direction -- binarizing $src and $tgt"
            fairseq-preprocess \
            --source-lang $src --target-lang $tgt \
            --validpref $valid \
            --testpref $test \
            --thresholdsrc 0 --thresholdtgt 0 \
            --destdir $DEST_BIN \
            --srcdict ${srcdict} --tgtdict ${tgtdict}
    fi

    # echo ""
    # echo "binarizing $src and $tgt"

    
}

binarize_supervised_to_en(){
    echo "binarizing spm data.."
    for LANG in "${LANGS[@]}"; do
        binarize $LANG $CENTRE "${DEST}/bpe/jointdict.txt" "${DEST}/bpe/jointdict.txt" \
        "${DEST}/bpe/spm.${TRAIN_PREF}.${LANG}-${CENTRE}" \
        "${DEST}/bpe/spm.${DEV_PREF}.${LANG}-${CENTRE}" \
        "${DEST}/bpe/spm.${TEST_PREF}.${LANG}-${CENTRE}"
    done

    echo "done!"
    
}

binarize_supervised_from_en(){
    echo "binarizing spm data.."
    for LANG in "${LANGS[@]}"; do
        binarize $CENTRE $LANG "${DEST}/bpe/jointdict.txt" "${DEST}/bpe/jointdict.txt" \
        "${DEST}/bpe/spm.${TRAIN_PREF}.${CENTRE}-${LANG}" \
        "${DEST}/bpe/spm.${DEV_PREF}.${CENTRE}-${LANG}" \
        "${DEST}/bpe/spm.${TEST_PREF}.${CENTRE}-${LANG}"
    done

    echo "done!"
    
}

binarize_zero(){

    echo "binarizing spm data.."

    SUBSETS=(
        $DEV_PREF
        $TEST_PREF
    )

    LANGS=(
        "uk"
        "ru"
    )

    for SRC in "${LANGS[@]}"; do
        for TGT in "${LANGS[@]}"; do
            if [ ! ${SRC} = ${TGT} ]; then

                binarize "${SRC}" "${TGT}" "${DEST}/bpe/jointdict.txt" "${DEST}/bpe/jointdict.txt" \
                "no_train_data" \
                "${DEST}/bpe/spm.${DEV_PREF}.${SRC}-${TGT}" \
                "${DEST}/bpe/spm.${TEST_PREF}.${SRC}-${TGT}" \
                "zero_shot"

            fi
        done
    done

    echo "done!"
    
}



#########################
SUBWORD_TYPE="unigram"
BPESIZE=32000
TRAIN_MAXLEN=500
#########################


# 1. learn bpe
# learn_spm_joint ${SUBWORD_TYPE} ${BPESIZE}

# 2. spm encode all lang pairs
# spm_encode_all

# 3. build_zero_shot
# spm_encode_zero_shot

# 4. binarize tokenized data
# binarize_supervised_to_en
# binarize_supervised_from_en
binarize_zero

