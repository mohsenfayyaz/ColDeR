import os
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from huggingface_hub import login

from beir.retrieval import models
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

load_dotenv()
login(os.environ["HF_TOKEN"])
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"], "HF_HOME:", os.environ["HF_HOME"])

def main(args):
    DATASET = args.dataset
    MODEL = args.model  # "facebook/contriever-msmarco"  # "msmarco-distilbert-base-v3"

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout[]

    
    import pathlib, os
    from beir import util
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(DATASET)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    logging.info("Dataset downloaded here: {}".format(data_path))

    data_path = f"datasets/{DATASET}"
    corpus_raw, queries, qrels = GenericDataLoader(data_path).load(split="test") # or split = "train" or "dev"

    gold_docs = set()
    for test_k, test_v in tqdm(qrels.items()):
        for doc_k, doc_v in test_v.items():
            gold_docs.add(doc_k)
    logging.info({
        "#Corpus:": len(corpus_raw), 
        "#Gold_Corpus:": len(gold_docs),
        "#Queries&qrels:": len(queries)
    })
    
    
    if args.use_gold_docs:
        corpus = {d: corpus_raw[d] for d in gold_docs}
    else:
        corpus = corpus_raw
    # print(gold_docs)
    # print(corpus_raw["doc101116"])
    # print(qrels["test2955"])
    # print("doc101116" in gold_docs)
    # print(corpus["doc101116"])

    model = DRES(models.SentenceBERT(MODEL), batch_size=128)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    # print(len(results["test0"]))
    # print(results["test2955"]["doc101116"])
    
    #### UPLOAD ON HF
    df_dict = []
    sorted_results = {k: dict(sorted(v.items(), key=lambda item: item[1], reverse=True)) for k, v in results.items()}
    for query_id in tqdm(sorted_results.keys()):
        score_values = list(sorted_results[query_id].values())
        df_dict.append({
            "query_id": query_id,
            "query": queries[query_id],
            "gold_docs": [doc_id for doc_id, v in qrels[query_id].items()],
            "gold_docs_text": {doc_id: corpus[doc_id] for doc_id, v in qrels[query_id].items()},
            "scores_stats": {"len": len(score_values), "max": max(score_values), "min": min(score_values), "std": np.std(score_values), "mean": np.mean(score_values), "median": np.median(score_values)},
            "scores_gold": {doc_id: sorted_results[query_id].get(doc_id, None) for doc_id, v in qrels[query_id].items()},
            "scores_1000": dict(list(sorted_results[query_id].items())[:1000]),
            # "scores_1000": sorted_results[query_id],
            "predicted_docs_text_10": {doc_id: corpus[doc_id] for doc_id, v in dict(list(sorted_results[query_id].items())[:10]).items()},
        })
    df = pd.DataFrame(df_dict)
    df.attrs['model'] = MODEL
    df.attrs['dataset'] = DATASET
    df.attrs['corpus_size'] = len(corpus)
    df.attrs['eval'] = {"ndcg": ndcg, "map": _map, "recall": recall, "precision": precision}
    logging.info(df.attrs)
    hf_path = f"hf://datasets/Retriever-Contextualization/datasets/{DATASET}/{MODEL.replace('/', '--')}_corpus{len(corpus)}"
    df.to_pickle(f"hf://datasets/Retriever-Contextualization/datasets/{DATASET}/{MODEL.replace('/', '--')}_corpus{len(corpus)}.pkl")
    df.to_pickle(f"hf://datasets/Retriever-Contextualization/datasets/{DATASET}/{MODEL.replace('/', '--')}_corpus{len(corpus)}.pkl.gz")
    df.to_parquet(f"./datasets/{DATASET}/{MODEL.replace('/', '--')}_corpus{len(corpus)}.parquet")
    df.to_parquet(f"hf://datasets/Retriever-Contextualization/datasets/{DATASET}/{MODEL.replace('/', '--')}_corpus{len(corpus)}.parquet")
    logging.info(f"UPLOADED: {hf_path}")
    df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nq")
    parser.add_argument("--model", type=str, default="facebook/contriever-msmarco")
    parser.add_argument("--use_gold_docs", type=lambda x: x.lower() == 'true', default=False, help="Only use gold docs and discard others")
    args = parser.parse_args()
    print(args)
    main(args)
