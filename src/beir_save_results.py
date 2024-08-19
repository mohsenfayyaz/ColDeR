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
login(os.environ["HF_API_TOKEN"])
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

    model = DRES(models.SentenceBERT(MODEL), batch_size=128)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)


    #### UPLOAD ON HF
    df_dict = []
    sorted_results = {k: dict(sorted(v.items(), key=lambda item: item[1], reverse=True)) for k, v in results.items()}
    for key in tqdm(sorted_results.keys()):
        df_dict.append({
            "key": key,
            "query": queries[key],
            "gold_docs": [k for k, v in qrels[key].items()],
            "gold_docs_text": [corpus[k] for k, v in qrels[key].items()],
            "results": sorted_results[key],
            "predicted_docs_text_5": [corpus[k] for k, v in dict(list(sorted_results[key].items())[:5]).items()],
        })
    df = pd.DataFrame(df_dict)
    df.attrs['model'] = MODEL
    df.attrs['dataset'] = DATASET
    df.attrs['corpus_size'] = len(corpus)
    df.attrs['eval'] = {"ndcg": ndcg, "map": _map, "recall": recall, "precision": precision}
    logging.info(df.attrs)
    hf_path = f"hf://datasets/Retriever-Contextualization/datasets/{DATASET}/{MODEL.replace('/', '--')}_corpus{len(corpus)}.parquet"
    df.to_parquet(hf_path.replace("hf://datasets/Retriever-Contextualization/", "./"))
    df.to_parquet(hf_path)
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
