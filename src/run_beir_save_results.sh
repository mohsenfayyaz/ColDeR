# cd src/

args=(
    --dataset "nq"
    --model "facebook/contriever-msmarco"
    --use_gold_docs True
)
python beir_save_results.py "${args[@]}"