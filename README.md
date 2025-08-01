#SCRAD
# SCRAD: Sequence Coherence-based Retrieval-Augmented Anomaly Detection

This repository contains the implementation of **SCRAD**, a graph anomaly detection framework that aligns Large Language Models (LLMs) with graph structures through retrieval-augmented sequence coherence discrimination.

## Framework Overview

SCRAD transforms graph anomaly detection into a sequence coherence discrimination task, leveraging two key innovations:
1. **Dual-Layer Semantic Edge Encoder**: Captures both micro-level (local) and macro-level (global) contextual information to enhance coherence in normal sequences and amplify distortions in anomalous ones.
2. **Entropy-Constrained Retrieval**: Retrieves semantically similar sequences from a sequence-derived knowledge graph, using structural correlations to improve coherence assessment.

The framework consists of four core modules: Sequence Construction, Dual-Layer Semantic Edge Encoding, Entropy-Constrained Sequence Retrieval, and Coherence-Driven Anomaly Detection.

## Dataset Sources
* The sources and usage of the [Reddit](https://snap.stanford.edu/data/soc-redditHyperlinks.html) dataset can be found on [snap.stanford.edu](https://snap.stanford.edu/data/soc-redditHyperlinks.html), as introduced in [Kumar *et al*., 2019](https://arxiv.org/abs/1902.07243). It contains a user interaction graph with edges denoting hyperlink relationships. The data can be downloaded from: [https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv.gz](https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv.gz)

* The sources and usage of the [Question](https://github.com/serranoqm/gadbench/tree/main/data/Question) dataset can be found in the [GADBench GitHub repository](https://github.com/serranoqm/gadbench), as described in [Platonov *et al*., 2024](https://arxiv.org/abs/2402.12847). It is a Q\&A forum graph with textual and structural features. The dataset is available at: [https://github.com/serranoqm/gadbench/tree/main/data/Question](https://github.com/serranoqm/gadbench/tree/main/data/Question)

* The sources and usage of the [Heal-Fraud](https://github.com/Graph-COM/HEAL) dataset can be found on the [HEAL project GitHub](https://github.com/Graph-COM/HEAL), as introduced in [Ma *et al*., 2023](https://arxiv.org/abs/2308.07873). It consists of healthcare insurance claims connected by shared attributes. The data can be accessed at: [https://github.com/Graph-COM/HEAL/tree/main/Heal-Data](https://github.com/Graph-COM/HEAL/tree/main/Heal-Data)

* The sources and usage of the [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) dataset can be found on [kaggle.com](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), as introduced in [Weber *et al*., 2019](https://arxiv.org/abs/1908.02591). It is a Bitcoin transaction graph used for anti-money laundering research. The dataset is available at: [https://www.kaggle.com/datasets/ellipticco/elliptic-data-set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

* The sources and usage of the [Amazon (Musical Instruments)](http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Musical_Instruments.json.gz) dataset can be found on the [Amazon Review Dataset website](https://nijianmo.github.io/amazon/index.html), as described in [McAuley *et al*., 2013](https://cseweb.ucsd.edu/~jmcauley/pdfs/sna2013.pdf). It includes user reviews and TF-IDF features. The dataset can be downloaded from: [http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Musical\_Instruments.json.gz](http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Musical_Instruments.json.gz)

* The sources and usage of the [Epinions](https://www.cse.msu.edu/~tangjili/trust.html) dataset can be found on [Tang’s trust network project page](https://www.cse.msu.edu/~tangjili/trust.html), and are also included in [GADBench](https://arxiv.org/abs/2306.12251) by \[Tang *et al*., 2023]. It is a trust-oriented user review graph. The dataset can be downloaded from: [https://www.cse.msu.edu/\~tangjili/data/trust.tar.gz](https://www.cse.msu.edu/~tangjili/data/trust.tar.gz)

Place datasets in the `./datasets` folder for processing.

## Installation

Ensure you have `Python=3.10` and `PyTorch=1.13.1` installed. Install dependencies via:
```bash
pip install -r requirements.txt
This repository contains the code implementation ofSCRAD for graph anomlay detection. 
```


## Preparing Data
Preparing Data
Preprocess raw data to generate sequences, embeddings, and knowledge graphs:
```bash
python preprocess.py --dataname [DATASET_NAME]

#Replace [DATASET_NAME] with one of: Reddit, Question, Elliptic, Heal-Fraud, Amazon, Epinions.
```



This script will process the raw data and prepare it for training and testing.

## Model Training
To train the SCRAD model, set the mode argument to train and run the `main.py` script:
```bash
python main.py --mode train --dataname [DATASET_NAME]
```
Ensure you have configured the necessary parameters and data paths in the script or through a configuration file.
Training uses a frozen LLM (DeepSeek) for coherence estimation and optimizes only the dual-layer encoder. Hyperparameters are automatically set based on the dataset.

## Model Testing
After training the model, you can test its performance by setting the mode argument to test and running the `main.py` script:
```python
python main.py --mode test --dataname [DATASET_NAME]
```
This outputs key metrics: AUC-ROC, AUC-PR, and Macro-F1.
This will evaluate the trained model on your test dataset and provide performance metrics.

## Result
SCRAD outperforms state-of-the-art baselines across six datasets. Key results (AUC-ROC) include:
Reddit: 97.72% (vs. 95.69% for best baseline)
Amazon: 98.92% (vs. 98.12% for best baseline)
Elliptic: 95.18% (vs. 93.32% for best baseline)
For detailed comparisons across all metrics and datasets, refer to the paper.

The following shows the test results of the Amazon dataset:
![](./fig/result.png)

