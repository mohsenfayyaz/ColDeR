# cd src/

args=(
    --dataset "nq"
    --model "facebook/contriever-msmarco"
    --use_gold_docs False
)
python beir_save_results.py "${args[@]}"
