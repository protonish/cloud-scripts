
ROOT="/local-scratch/nishant/wordemb"
SETUP="ted59_setup"
TED59_ROOT="${ROOT}/experiments/${SETUP}/raw"

EMBEX="${ROOT}/embex"

mkdir -p ${TED59_ROOT}

CSV_DATA="/cs/natlang-expts/nishant/ted59_data/word-embeddings-for-nmt/ted_talks"

# python3 "${EMBEX}/ted_reader_options.py" --data "${CSV_DATA}" --output "${TED59_ROOT}" --langs-list "${CSV_DATA}/langs.txt" --to-en

LANGS=(
    "es"
    pt-br
    fr
    ru
    he
    ar
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
    )


USER_SYMBOLS=$(
        for LANG in "${LANGS[@]}"; do 
            echo "${LANG}_en";
            echo "en_${LANG}";
        done | tr "\n" ",")

echo $USER_SYMBOLS

# for lang in "${LANGS[@]}";do
#     echo "--> $lang-en data .."
#     python3 "${EMBEX}/ted_reader_options.py" --data "${CSV_DATA}" --output "${TED59_ROOT}" --src-langs $lang --tgt-langs en
    
#     echo "--> en-$lang data .."
#     python3 "${EMBEX}/ted_reader_options.py" --data "${CSV_DATA}" --output "${TED59_ROOT}" --src-langs en --tgt-langs $lang

# done

# python3 "${EMBEX}/ted_reader_options.py" --data "${CSV_DATA}" --output "${TED59_ROOT}" --src-langs ar --tgt-langs fr --zero
# python3 "${EMBEX}/ted_reader_options.py" --data "${CSV_DATA}" --output "${TED59_ROOT}" --src-langs fr --tgt-langs ar --zero
# python3 "${EMBEX}/ted_reader_options.py" --data "${CSV_DATA}" --output "${TED59_ROOT}" --src-langs uk --tgt-langs ru --zero
# python3 "${EMBEX}/ted_reader_options.py" --data "${CSV_DATA}" --output "${TED59_ROOT}" --src-langs ru --tgt-langs uk --zero