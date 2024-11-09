# cd src/
# bash save_decompx.sh

args=(
    --dataset "re-docred"  # nq, re-docred
    --model "facebook/contriever-msmarco"
    --use_gold_docs False
)
python save_decompx.py "${args[@]}"
