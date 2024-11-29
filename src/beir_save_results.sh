# cd src/
# bash beir_save_results.sh

args=(
    --dataset "re-docred"  # nq, re-docred
    --query_model "OpenMatch/cocodr-base-msmarco"  # facebook/contriever-msmarco 
    --context_model "OpenMatch/cocodr-base-msmarco"
    --pooling "cls"
    --use_gold_docs False
)
python beir_save_results.py "${args[@]}"

args=(
    --dataset "re-docred"  # nq, re-docred
    --query_model "facebook/contriever-msmarco"  # facebook/contriever-msmarco 
    --context_model "facebook/contriever-msmarco"
    --pooling "avg"
    --use_gold_docs False
)
python beir_save_results.py "${args[@]}"

args=(
    --dataset "re-docred"  # nq, re-docred
    --query_model "facebook/dragon-plus-query-encoder"  # facebook/contriever-msmarco 
    --context_model "facebook/dragon-plus-context-encoder"
    --pooling "cls"
    --use_gold_docs False
)
python beir_save_results.py "${args[@]}"
