# cd src/
# bash beir_save_results.sh

export CUDA_VISIBLE_DEVICES=7
DATASET="nq"  # nq, re-docred

args=(
    --dataset $DATASET
    --query_model "facebook/contriever"
    --context_model "facebook/contriever"
    --pooling "avg"
    --use_gold_docs False
)
python beir_save_results.py "${args[@]}"

args=(
    --dataset $DATASET
    --query_model "facebook/contriever-msmarco"  # facebook/contriever-msmarco 
    --context_model "facebook/contriever-msmarco"
    --pooling "avg"
    --use_gold_docs False
)
python beir_save_results.py "${args[@]}"

args=(
    --dataset $DATASET
    --query_model "facebook/dragon-plus-query-encoder"  # facebook/contriever-msmarco 
    --context_model "facebook/dragon-plus-context-encoder"
    --pooling "cls"
    --use_gold_docs False
)
python beir_save_results.py "${args[@]}"

args=(
    --dataset $DATASET
    --query_model "facebook/dragon-roberta-query-encoder"  # facebook/contriever-msmarco 
    --context_model "facebook/dragon-roberta-context-encoder"
    --pooling "cls"
    --use_gold_docs False
)
python beir_save_results.py "${args[@]}"

# args=(
#     --dataset $DATASET
#     --query_model "Shitao/RetroMAE"
#     --context_model "Shitao/RetroMAE"
#     --pooling "cls"
#     --use_gold_docs False
# )
# python beir_save_results.py "${args[@]}"

# args=(
#     --dataset $DATASET
#     --query_model "Shitao/RetroMAE_MSMARCO"
#     --context_model "Shitao/RetroMAE_MSMARCO"
#     --pooling "cls"
#     --use_gold_docs False
# )
# python beir_save_results.py "${args[@]}"

args=(
    --dataset $DATASET
    --query_model "Shitao/RetroMAE_MSMARCO_finetune"
    --context_model "Shitao/RetroMAE_MSMARCO_finetune"
    --pooling "cls"
    --use_gold_docs False
)
python beir_save_results.py "${args[@]}"

args=(
    --dataset $DATASET  # nq, re-docred
    --query_model "OpenMatch/cocodr-base-msmarco"  # facebook/contriever-msmarco 
    --context_model "OpenMatch/cocodr-base-msmarco"
    --pooling "cls"
    --use_gold_docs False
)
python beir_save_results.py "${args[@]}"
