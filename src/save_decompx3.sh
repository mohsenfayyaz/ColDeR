# cd src/
# bash save_decompx.sh

export HF_HOME="/local1/mohsenfayyaz/.hfcache/"
export CUDA_VISIBLE_DEVICES=3

# args=(
#     --beir_file "hf://datasets/Retriever-Contextualization/datasets/re-docred/facebook--contriever-msmarco_corpus105925.pkl"
# )
# python save_decompx.py "${args[@]}"

# args=(
#     --beir_file "hf://datasets/Retriever-Contextualization/datasets/re-docred/facebook--dragon-plus-query-encoder_corpus105925.pkl"
# )
# python save_decompx.py "${args[@]}"

args=(
    --beir_file "hf://datasets/Retriever-Contextualization/datasets/re-docred/OpenMatch--cocodr-base-msmarco_corpus105925.pkl"
)
python save_decompx.py "${args[@]}"
