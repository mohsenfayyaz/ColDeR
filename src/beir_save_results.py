import os
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
from typing import List, Dict, Union, Tuple
from transformers import AutoTokenizer, AutoModel


from beir.retrieval import models
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

load_dotenv()
login(os.environ["HF_TOKEN"])
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"], "HF_HOME:", os.environ["HF_HOME"])

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout[]

class DatasetLoader:
    def __init__(self):
        pass

    def load_dataset(self, dataset_name, use_gold_docs=False) -> dict:
        """
        write docstirng here

        Args:
            dataset_name (str): name of the dataset [re-docred, nq, ...]
            use_gold_docs (bool, optional): To use only gold docs as corpus or not. Defaults to False.
        
        Returns:
            dict: {
                queries: {'test0': 'what is non controlling interest on balance sheet', ...}
                qrels: {'test0': {'doc0': 1, 'doc1': 1}, ...)
                corpus: {'doc0': {'text': "In accou...", 'title': 'Minority interest'}, ...})
            }
        """
        logging.info(f"Loading dataset: {dataset_name}")
        if dataset_name == "re-docred":
            dataset = self.load_redocred_dataset()
        else:
            dataset = self.load_beir_datasets(dataset_name, use_gold_docs)
        logging.info({
            "#Corpus:": len(dataset['corpus']), 
            "#Queries&qrels:": len(dataset['queries']),
        })
        return dataset
    
    def load_redocred_dataset(self):
        df = pd.read_pickle("hf://datasets/Retriever-Contextualization/datasets/Re-DocRED/queries_test_validation_clean.pkl")
        queries = {row["id"]: row["query_question"] for i, row in df.iterrows()}
        qrels = {row["id"]: {row["title"]: 1} for i, row in df.iterrows()}
        df_cropus = pd.read_pickle("hf://datasets/Retriever-Contextualization/datasets/Re-DocRED/corpus_all.pkl.gz")
        corpus = {row["title"]: {"text": " ".join([" ".join(sent) for sent in row["sents"]]), "title": row["title"]} for i, row in df_cropus.iterrows()}
        return {
            "corpus": corpus,
            "queries": queries,
            "qrels": qrels
        }
    
    def load_beir_datasets(self, dataset_name, use_gold_docs):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
        out_dir = os.path.join(os.getcwd(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        logging.info("Dataset downloaded here: {}".format(data_path))

        data_path = f"datasets/{dataset_name}"
        corpus_raw, queries, qrels = GenericDataLoader(data_path).load(split="test") # or split = "train" or "dev"

        gold_docs = set()
        for test_k, test_v in tqdm(qrels.items()):
            for doc_k, doc_v in test_v.items():
                gold_docs.add(doc_k)
        if use_gold_docs:
            corpus = {d: corpus_raw[d] for d in gold_docs}
        else:
            corpus = corpus_raw
        return {
            "corpus": corpus,
            "queries": queries,
            "qrels": qrels
        }
        
class YourCustomDEModel:
    def __init__(self, q_model, doc_model, pooling, sep: str = " ", **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(q_model)
        self.query_encoder = AutoModel.from_pretrained(q_model)
        self.context_encoder = AutoModel.from_pretrained(doc_model)
        self.pooling = pooling
        self.sep = sep
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size=128, **kwargs) -> np.ndarray:
        print("Q")
        print(len(queries))
        return self.encode_in_batch(self.query_encoder, queries, batch_size)
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size=128, **kwargs) -> np.ndarray:
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.encode_in_batch(self.context_encoder, sentences, batch_size)

    def encode_in_batch(self, model, sentences: List[str], batch_size=128, **kwargs) -> np.ndarray:
        model.to(self.device)
        all_embeddings = []
        for batch in tqdm(torch.utils.data.DataLoader(sentences, batch_size=batch_size, shuffle=False)):
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            outputs = model(**inputs)
            ### POOLING
            if self.pooling == "avg":
                embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
            elif self.pooling == "cls":
                embeddings = outputs.last_hidden_state[:, 0, :]  # [128, 768] = [batch, emb_dim]
            else:
                raise ValueError("Pooling method not supported")
            all_embeddings.extend(embeddings.detach().cpu().numpy())
        all_embeddings = np.array(all_embeddings)
        print(all_embeddings.shape)
        return all_embeddings

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings



def main(args):
    DATASET = args.dataset
    MODEL = args.query_model  # "facebook/contriever-msmarco"  # "msmarco-distilbert-base-v3"
    POOLING = args.pooling

    dataset = DatasetLoader().load_dataset(DATASET, use_gold_docs=args.use_gold_docs)
    corpus, queries, qrels = dataset["corpus"], dataset["queries"], dataset["qrels"]

    # model = DRES(models.SentenceBERT(MODEL), batch_size=128)
    model = DRES(YourCustomDEModel(args.query_model, args.context_model, POOLING), batch_size=16)
    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    
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
    df.attrs['query_model'] = args.query_model
    df.attrs['context_model'] = args.context_model
    df.attrs['pooling'] = POOLING
    df.attrs['dataset'] = DATASET
    df.attrs['corpus_size'] = len(corpus)
    df.attrs['eval'] = {"ndcg": ndcg, "map": _map, "recall": recall, "precision": precision}
    logging.info(df.attrs)
    hf_path = f"hf://datasets/Retriever-Contextualization/datasets/{DATASET}/{MODEL.replace('/', '--')}_corpus{len(corpus)}"
    df.to_pickle(f"hf://datasets/Retriever-Contextualization/datasets/{DATASET}/{MODEL.replace('/', '--')}_corpus{len(corpus)}.pkl")
    df.to_pickle(f"hf://datasets/Retriever-Contextualization/datasets/{DATASET}/{MODEL.replace('/', '--')}_corpus{len(corpus)}.pkl.gz")
    logging.info(f"UPLOADED: {hf_path}")
    df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nq")
    parser.add_argument("--query_model", type=str, default="facebook/contriever-msmarco")
    parser.add_argument("--context_model", type=str, default="facebook/contriever-msmarco")
    parser.add_argument("--pooling", type=str, default="avg")
    parser.add_argument("--use_gold_docs", type=lambda x: x.lower() == 'true', default=False, help="Only use gold docs and discard others")
    args = parser.parse_args()
    print(args)
    main(args)
