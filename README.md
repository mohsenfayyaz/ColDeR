<center>
<h1 align="center">❄️ Collapse of Dense Retrievers [ ACL 2025 ] ❄️</h1>

<!-- Provide a quick summary of the dataset. -->
<p align="center">A Framework for Identifying Biases in Retrievers</p>

<p align="center">
  <a style="display: inline; max-width: none" href="https://arxiv.org/abs/2503.05037"><img style="display: inline; max-width: none" alt="arXiv" src="https://img.shields.io/badge/arXiv-2503.05037-b31b1b.svg"></a>
  <a style="display: inline; max-width: none" href="https://huggingface.co/datasets/mohsenfayyaz/ColDeR"><img style="display: inline; max-width: none" alt="HuggingFace Dataset" src="https://img.shields.io/badge/🤗-Hugging%20Face%20Dataset-FFD21E?style=flat"></a>
  <a style="display: inline; max-width: none" href="https://colab.research.google.com/github/mohsenfayyaz/ColDeR/blob/main/Benchmark_Eval.ipynb"><img style="display: inline; max-width: none" alt="Benchmark Eval Colab Demo" src="https://img.shields.io/badge/​-Evaluate%20in%20Colab-blue?logo=googlecolab&logoColor=F9AB00&style=flat"></a>
  <a style="display: inline; max-width: none" href="https://github.com/mohsenfayyaz/ColDeR"><img style="display: inline; max-width: none" alt="Github Code" src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white&style=flat"></a>
</p>

<p align="center">
<code align="center">⚠️ The best accuracy of Dense Retrievers on the foil (default) set is lower than 🔴10%🔴. </code>
</p>
<!-- Provide a longer summary of what this dataset is. -->
<blockquote align="center">
Retrievers consistently score <b>document_1</b> higher than <b>document_2</b> in all subsets. <br>
<!-- It shows their preference for the more biased document in each bias scenario. <br> -->
⇒ Retrieval biases often outweigh the impact of answer presence.
</blockquote>


<h2 align="center">🏆 Leaderboard 🏆</h2>

<div align="center">
  
| Model               | Accuracy | Paired t-Test Statistic | p-value |
|----------------------|:-------------:|:---------------:|:-----------------------:|
|🥇[ReasonIR-8B](https://huggingface.co/reasonir/ReasonIR-8B) 🆕 | 8.0\% | -36.92 | < 0.01 |
|🥈[ColBERT (v2)](https://huggingface.co/colbert-ir/colbertv2.0) 🆕 | 7.6\% | -20.96 | < 0.01 |
|🥉[COCO-DR Base MSMARCO](https://huggingface.co/OpenMatch/cocodr-base-msmarco) | 2.4\% | -32.92 | < 0.01 |
|[Dragon+](https://huggingface.co/facebook/dragon-plus-query-encoder)  | 1.2\% | -40.94 | < 0.01 |
|[Dragon RoBERTa](https://huggingface.co/facebook/dragon-roberta-query-encoder)  | 0.8\% | -36.53 | < 0.01 |
|[Contriever MSMARCO](https://huggingface.co/facebook/contriever-msmarco) | 0.8\% | -42.25 | < 0.01 |
|[RetroMAE MSMARCO FT](https://huggingface.co/Shitao/RetroMAE_MSMARCO_finetune) | 0.4\% | -41.49 | < 0.01 |
|[Contriever](https://huggingface.co/facebook/contriever)  | 0.4\% | -34.58 | < 0.01 |

Evaluate any model using this code: [https://colab.research.google.com/github/mohsenfayyaz/ColDeR/blob/main/Benchmark_Eval.ipynb](https://colab.research.google.com/github/mohsenfayyaz/ColDeR/blob/main/Benchmark_Eval.ipynb)


<h2 align="center">🔍 Dataset Examples 🔍</h2>
<img src="https://huggingface.co/datasets/mohsenfayyaz/ColDeR/resolve/main/figs/examples.png" width="90%" title="" style="border-radius: 5px; max-width: 800px">
<!-- <img src="https://huggingface.co/datasets/mohsenfayyaz/ColDeR/resolve/main/figs/fig1.png" width="300" title="" style="border-radius: 15px;"> -->
</center>

</div>

---

### Dataset Subsets

* **foil (default):**
  * **document_1:** Foil Document with Multiple Biases but No Evidence: This document contains multiple biases, such as repetition and position biases. It includes two repeated mentions of the head entity in the opening sentence, followed by a sentence that mentions the head but not the tail (answer). So it does not include the evidence.
  * **document_2:** Evidence Document with Unrelated Content: This document includes four unrelated sentences from another document, followed by the evidence sentence with both the head and tail entities. The document ends with the same four unrelated sentences.
  <!-- <img src="https://huggingface.co/datasets/mohsenfayyaz/ColDeR/resolve/main/figs/fig2.png" width="200" title="" style="border-radius: 5px;"> -->
* **answer_importance:**
  * **document_1:** Document with Evidence:  Contains a leading evidence sentence with both the head entity and the tail entity (answer).
  * **document_2:** Document without Evidence: Contains a leading sentence with only the head entity but no tail.
* **brevity_bias:**
  * **document_1:** Single Evidence, consisting of only the evidence sentence.
  * **document_2:** Evidence+Document, consisting of the evidence sentence followed by the rest of the document.
* **literal_bias:**
  * **document_1:** Both query and document use the shortest name variant (short-short).
  * **document_2:** The query uses the short name but the document contains the long name variant (short-long).
* **position_bias:**
  * **document_1:** Beginning-Evidence Document: The evidence sentence is positioned at the start of the document.
  * **document_2:** End-Evidence Document: The same evidence sentence is positioned at the end of the document.
* **repetition_bias:**
  * **document_1:** More Heads, comprising an evidence sentence and two more sentences containing head mentions but no tails
  * **document_2:** Fewer Heads, comprising an evidence sentence and two more sentences without head or tail mentions from the document
* **poison:**
  * **document_1:** Poisoned Biased Evidence: We add the evidence sentence to foil document 1 and replace the tail entity in it with a contextually plausible but entirely incorrect entity using GPT-4o.
  * **document_2:** Correct Evidence Document with Unrelated Content: This document includes four unrelated sentences from another document, followed by the evidence sentence with both the head and tail entities. The document ends with the same four unrelated sentences.

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Paper:** [https://arxiv.org/abs/2503.05037](https://arxiv.org/abs/2503.05037)
- **Dataset:** [https://huggingface.co/datasets/mohsenfayyaz/ColDeR](https://huggingface.co/datasets/mohsenfayyaz/ColDeR)
- **Repository:** [https://github.com/mohsenfayyaz/ColDeR](https://github.com/mohsenfayyaz/ColDeR)


## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**
If you found this work useful, please consider citing our paper:
```bibtex
@inproceedings{fayyaz-etal-2025-collapse,
    title = "Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence",
    author = "Fayyaz, Mohsen  and
      Modarressi, Ali  and
      Schuetze, Hinrich  and
      Peng, Nanyun",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.447/",
    pages = "9136--9152",
    ISBN = "979-8-89176-251-0",
    abstract = "Dense retrieval models are commonly used in Information Retrieval (IR) applications, such as Retrieval-Augmented Generation (RAG). Since they often serve as the first step in these systems, their robustness is critical to avoid downstream failures. In this work, we repurpose a relation extraction dataset (e.g., Re-DocRED) to design controlled experiments that quantify the impact of heuristic biases, such as a preference for shorter documents, on retrievers like Dragon+ and Contriever. We uncover major vulnerabilities, showing retrievers favor shorter documents, early positions, repeated entities, and literal matches, all while ignoring the answer{'}s presence! Notably, when multiple biases combine, models exhibit catastrophic performance degradation, selecting the answer-containing document in less than 10{\%} of cases over a synthetic biased document without the answer. Furthermore, we show that these biases have direct consequences for downstream applications like RAG, where retrieval-preferred documents can mislead LLMs, resulting in a 34{\%} performance drop than providing no documents at all.https://huggingface.co/datasets/mohsenfayyaz/ColDeR"
}
```
