# cd src/
# bash beir_save_results.sh


### --- NQ --- ###
args=(
    --dataset "nq"  # nq, re-docred
    --query_model "OpenMatch/cocodr-base-msmarco"  # facebook/contriever-msmarco 
    --context_model "OpenMatch/cocodr-base-msmarco"
    --pooling "cls"
    --use_gold_docs False
)
python beir_save_results.py "${args[@]}"


### --- Re Docred --- ###

# args=(
#     --dataset "re-docred"  # nq, re-docred
#     --query_model "OpenMatch/cocodr-base-msmarco"  # facebook/contriever-msmarco 
#     --context_model "OpenMatch/cocodr-base-msmarco"
#     --pooling "cls"
#     --use_gold_docs False
# )
# python beir_save_results.py "${args[@]}"

# args=(
#     --dataset "re-docred"  # nq, re-docred
#     --query_model "OpenMatch/cocodr-large-msmarco"  # facebook/contriever-msmarco 
#     --context_model "OpenMatch/cocodr-large-msmarco"
#     --pooling "cls"
#     --use_gold_docs False
# )
# python beir_save_results.py "${args[@]}"

# args=(
#     --dataset "re-docred"  # nq, re-docred
#     --query_model "facebook/contriever-msmarco"  # facebook/contriever-msmarco 
#     --context_model "facebook/contriever-msmarco"
#     --pooling "avg"
#     --use_gold_docs False
# )
# python beir_save_results.py "${args[@]}"

# args=(
#     --dataset "re-docred"  # nq, re-docred
#     --query_model "facebook/dragon-plus-query-encoder"  # facebook/contriever-msmarco 
#     --context_model "facebook/dragon-plus-context-encoder"
#     --pooling "cls"
#     --use_gold_docs False
# )
# python beir_save_results.py "${args[@]}"

# args=(
#     --dataset "re-docred"  # nq, re-docred
#     --query_model "facebook/dragon-roberta-query-encoder"  # facebook/contriever-msmarco 
#     --context_model "facebook/dragon-roberta-context-encoder"
#     --pooling "cls"
#     --use_gold_docs False
# )
# python beir_save_results.py "${args[@]}"

# args=(
#     --dataset "re-docred"
#     --query_model "Shitao/RetroMAE"
#     --context_model "Shitao/RetroMAE"
#     --pooling "cls"
#     --use_gold_docs False
# )
# python beir_save_results.py "${args[@]}"

# args=(
#     --dataset "re-docred"
#     --query_model "Shitao/RetroMAE_MSMARCO"
#     --context_model "Shitao/RetroMAE_MSMARCO"
#     --pooling "cls"
#     --use_gold_docs False
# )
# python beir_save_results.py "${args[@]}"

# args=(
#     --dataset "re-docred"
#     --query_model "Shitao/RetroMAE_MSMARCO_finetune"
#     --context_model "Shitao/RetroMAE_MSMARCO_finetune"
#     --pooling "cls"
#     --use_gold_docs False
# )
# python beir_save_results.py "${args[@]}"

# args=(
#     --dataset "re-docred"
#     --query_model "facebook/contriever"
#     --context_model "facebook/contriever"
#     --pooling "avg"
#     --use_gold_docs False
# )
# python beir_save_results.py "${args[@]}"
