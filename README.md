# SCRAD: Aligning LLMs with Graph Anomaly Detection via Retrieval-Augmented Sequence Coherence Discrimination

Project repository: [TY-Tongyao/SCRAD](https://github.com/TY-Tongyao/SCRAD).

SCRAD formulates **graph anomaly detection** as **sequence coherence discrimination**. For a target node, the method constructs neighborhood sequences, learns dual-level semantic edge representations, retrieves informative reference sequences through entropy-constrained matching and a sequence-derived knowledge graph (KG), and uses a frozen large language model (LLM) to assess sequence coherence. Sequence-level decisions are aggregated to identify anomalous nodes.

The framework consists of four main components:

1. **Sequence construction:** DFS-based multi-hop neighborhood sequences.
2. **Dual-level semantic edge encoding:** complementary local and global edge representations with adaptive gating.
3. **Entropy-constrained retrieval:** sequence matching, KG path evidence, and a retrieval memory bank.
4. **Coherence-driven anomaly detection:** quantized and projected embeddings, a fixed two-shot prompt, LLM coherence predictions, and node-level aggregation.

> **Code availability:** The public code bundle accompanying this README currently includes configured training and inference entry points for **Reddit (Reddit2)** and **Questions**. The revised manuscript evaluates **six** datasets. The four other datasets are documented below for provenance and manuscript correspondence; this code bundle does not provide ready-to-run configurations for all six. Do not assume that a raw upstream download is directly compatible with the SCRAD loaders.

## 1. Datasets used in the manuscript

The statistics below are reported in **Table 1 of the revised manuscript**, describing the graphs used for the paper's evaluation. These are experimental graph statistics and should not automatically be treated as the statistics of the unprocessed upstream releases.

| Dataset | Nodes | Edges | Features | Anomaly ratio | Synthetic anomaly groups |
| --- | ---: | ---: | ---: | ---: | --- |
| Reddit | 232,965 | 11,606,919 | 602 | 4.8% | 15 × 373 |
| Question | 48,921 | 153,540 | 301 | 7.3% | 15 × 119 |
| Elliptic | 204,816 | 239,472 | 167 | 11.5% | — |
| Heal-Fraud | 79,654 | 5,023,112 | 19 | 3.8% | — |
| Amazon | 12,078 | 4,465,102 | 23 | 7.9% | — |
| Epinions | 74,512 | 507,884 | 52 | 9.8% | — |

The manuscript organizes these benchmarks into social (Reddit, Question), financial/fraud-related (Elliptic, Heal-Fraud), and review/trust (Amazon, Epinions) domains. Its Section 4.1 describes the two social datasets as using **injected structural and contextual anomalies**, with seed **42** and the same generated anomaly sets for compared methods. The exact experiment-ready versions of the other four graphs are not distributed in this code ZIP.

### 1.1. Original sources and scholarly references

The links in this table point to **upstream datasets, project pages, or academic references**, **not necessarily to the exact processed SCRAD graphs**. To obtain the ready-to-use Reddit and Questions data, use Section 2 instead.

| Dataset | Original/upstream source | Manuscript citation and version notes |
| --- | --- | --- |
| **Reddit (Reddit2)** | [PyTorch Geometric Reddit2](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Reddit2.html); [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT) | The manuscript cites **Hamilton et al. (2017), [23]**. The **Reddit2/GraphSAINT** resource matches 232,965 nodes and 602 features. It is **not** the SNAP Reddit Hyperlink Network linked in an older README. PyG reports 23,213,838 directed edge entries, corresponding to 11,606,919 undirected pairs when symmetrically stored. The original community-class labels are **not** SCRAD binary anomaly labels. |
| **Question / Questions** | [Yandex Research: heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs), `data/questions` | **Platonov et al., ICLR 2023, [24]**. The original Questions release reports 48,921 nodes, 153,540 undirected edges, and 301 features. Original classification labels should not be confused with injected anomaly labels. The manuscript calls it *Question*; the code uses `questions`. |
| **Heal-Fraud** | [HEAL project](https://github.com/Graph-COM/HEAL) | **Ma et al., IJCAI 2023, [25]**. This is an upstream research/project source; the ZIP does not provide a verified conversion to the specific Table 1 graph. |
| **Elliptic** | [Elliptic data on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) | **Weber et al., KDD 2019, [13]**. Access and preprocessing follow upstream terms; the exact SCRAD graph transformation is not included here. |
| **Amazon** | [UCSD Amazon Review Data](https://nijianmo.github.io/amazon/index.html) ([Musical Instruments reviews](https://nijianmo.github.io/amazon/index.html)) | **McAuley and Leskovec, WWW 2013, [14]**. This points to upstream review records, **not** a ready-to-run 12,078-node, 23-feature anomaly graph. In particular, Amazon Musical Instruments here should not be silently substituted with a different benchmark's Amazon graph. |
| **Epinions** | [GADBench project](https://github.com/squareRoot3/GADBench); [original trust-network resource](https://www.cse.msu.edu/~tangjili/trust.html) | The manuscript cites **Tang et al., NeurIPS 2023 (GADBench), [26]**. The original trust-network resource and benchmark preprocessing are separate from a confirmed match to the manuscript's Table 1 statistics. |

**Important provenance note.** The scholarly citation identifies relevant research, while a source link identifies a place to obtain upstream data. Neither by itself establishes that the downloaded file has the **same adjacency representation, feature extraction, anomaly labels, split, or number of edges** as the SCRAD experiments. Please use the released experiment-ready bundle for the currently supported datasets.

## 2. Obtaining the processed SCRAD data and experimental resources

**The source-code ZIP does not contain the large processed datasets, pretrained checkpoints, retrieval artifacts, or full node-level predictions. These resources are distributed separately.** The `README.md` supplied with the current code bundle provides an external resource share:

- **Baidu Netdisk:** [SCRAD resource share](https://pan.baidu.com/s/11LeQ_Q6Wwk5A7C9Ylrv4ew?pwd=y6mm)
- **Extraction code:** `y6mm`

This is a **project-provided distribution location for processed resources**, not the academic origin of Reddit or Questions. The filenames and resource descriptions below follow the README bundled with the released code. Check the actual downloaded filenames and archive contents before extraction.

The supplied code README lists:

| Archive name shown in the supplied README | Intended contents | Expected destination |
| --- | --- | --- |
| `data.zip` (also described as `scrad-datasets.tar.gz`) | Original and anomaly-injected Reddit/Questions graphs plus fixed splits | Project `data/` directory |
| `scrad-weight.tar.gz` | Trained checkpoints | Project `weights/` directory |
| `scrad-contexts.tar.gz` | Prepared retrieval bank, KG inputs, sequences/embeddings, evaluation labels | Project `artifacts/` directory |
| `scrad-deepseek14b-awq.tar.gz` | DeepSeek 14B AWQ model and tokenizer | Project `models/` directory |
| `scrad-predictions.tar.gz` | Saved per-node test predictions and evaluation records | Project `reference/` directory |

**Archive names:** The released code README calls the dataset archive `data.zip` and also describes a possible `scrad-datasets.tar.gz` packaging; it lists `scrad-weight.tar.gz` but one extraction example uses `scrad-weights.tar.gz`. Use the **actual downloaded filename** (not an assumed alternative).

For `.tar.gz` archives, use `tar -xzf ARCHIVE_NAME.tar.gz`; for `data.zip`, use a ZIP extractor. Extract data and other resources so that the expected directories appear directly under the **repository root**; avoid accidental double nesting such as `SCRAD/data/data/reddit/`.

### 2.1. Expected processed-data layout

```text
SCRAD/
├── data/
│   ├── original/
│   │   ├── reddit/
│   │   └── questions/
│   ├── reddit/
│   │   ├── reddit.npz
│   │   ├── reddit_adj.npz
│   │   └── dataset.json
│   └── questions/
│       ├── questions.npz
│       ├── questions_adj.npz
│       └── dataset.json
├── weights/
├── artifacts/
├── models/
└── reference/
```

The project-supplied README describes each processed dataset directory as containing:

- `<dataset>.npz`: processed features, labels and split information;
- `<dataset>_adj.npz`: sparse adjacency matrix;
- `dataset.json`: dataset statistics/metadata.

For the released Reddit and Questions processed bundles, the supplied README records:

| Dataset | Injected anomaly nodes | Train nodes | Validation nodes | Test nodes |
| --- | ---: | ---: | ---: | ---: |
| Reddit | 11,182 | 93,186 | 46,593 | 93,186 |
| Questions | 3,571 | 19,568 | 9,784 | 19,569 |

The splits implement approximately **40% / 20% / 40%** at the target-node level, as stated in the manuscript. The prepared datasets, rather than raw upstream node-classification labels, should be used for SCRAD anomaly training/evaluation.

### 2.2. Preparing raw upstream data yourself

The upstream resources in Section 1.1 are useful for provenance and independent study. **The current ZIP does not include a verified end-to-end script that reproduces the exact published preprocessing and synthetic-anomaly generation from all six upstream sources.** In particular, downloading Reddit2 or Questions directly does **not** substitute for obtaining the preprocessed binary-anomaly labels, matching sparse graph layout, and fixed splits. For reproduction of the currently released experiments, download the processed dataset archive in Section 2.

## 3. Installation and requirements

The new code bundle specifies **Python 3.11** and the following versions in `requirements.txt`:

```text
numpy==2.2.6
scipy==1.17.1
scikit-learn==1.7.2
torch==2.8.0
transformers==4.57.0
accelerate==1.10.1
safetensors==0.8.0
vllm==0.10.2
```

The supplied README cites an **NVIDIA A100 80 GB** reference environment; GPU memory requirements depend on training/inference stage and model placement. These dependencies supersede the older README's Python 3.10 / PyTorch 1.13.1 instructions.

```bash
python -m pip install -r requirements.txt
```

The code expects a local model directory at:

```text
models/DeepSeek-R1-Distill-Qwen-14B-AWQ/
```

The provided metadata references this upstream [AWQ checkpoint revision](https://huggingface.co/casperhansen/deepseek-r1-distill-qwen-14b-awq/tree/bc43ec1bbf08de53452630806d5989208b4186db). Ensure the model and tokenizer files are placed in the path expected by the scripts. The local prepared model archive, if available, is listed in Section 2. The paper identifies the underlying model as **DeepSeek-R1-Distill-Qwen-14B**; the code bundle specifically expects an **AWQ-quantized distribution**.

## 4. Training (current release: Reddit and Questions)

Train with the top-level `train.py` launcher, which supports `--dataset reddit` and `--dataset questions`. The launcher reads `configs/<dataset>.json` and runs the staged pipeline:

- `pretrain`: pretraining / initialization;
- `prepare`: build prepared inference/training resources;
- `single300`: single-view LLM alignment;
- `joint25`: joint alignment.

The supplied README describes PMP pretraining for 200 epochs, the dual-level encoder for 50 epochs, single-view alignment for 300 steps, and joint alignment for 25 steps, with LLM weights frozen. For authoritative stage parameters, inspect `configs/reddit.json` and `configs/questions.json`.

```bash
python train.py --dataset questions --gpu 0
python train.py --dataset reddit --gpu 0
```

To preview commands without executing them:

```bash
python train.py --dataset questions --gpu 0 --dry-run
```

To execute a single stage (after ensuring its prerequisites are available):

```bash
python train.py --dataset questions --gpu 0 --stage pretrain
```

By default, generated outputs/logs are placed under `runs/<dataset>/`. The script also supports `--data`, `--model`, `--output`, and `--gpu-memory-fraction`. The new code uses **`train.py`**, not the original repository's `main.py --mode train`.

## 5. Inference and evaluation

Before inference, unpack compatible checkpoint, prepared-artifact and model archives into `weights/`, `artifacts/` and `models/`. By default, `infer.py` requires:

```text
weights/<dataset>/checkpoint.pt
artifacts/<dataset>/prepared.json
models/DeepSeek-R1-Distill-Qwen-14B-AWQ/config.json
```

Run inference with the supplied resources:

```bash
python infer.py --dataset questions --gpus 0
python infer.py --dataset reddit --gpus 0,1
```

For newly trained checkpoints, the original code README provides:

```bash
python infer.py --dataset questions --run runs/questions/prepared --checkpoint runs/questions/joint25/last.pt --gpus 0
python infer.py --dataset reddit --run runs/reddit/prepared --checkpoint runs/reddit/joint25/last.pt --gpus 0,1
```

The script prints Macro-F1 and writes an inference log and report under `outputs/<dataset>/` by default. The new code uses **`infer.py`**, not `main.py --mode test`.

### 5.1. Recalculate saved-test-prediction Macro-F1 on CPU

After obtaining and extracting the prediction and context archives from Section 2, verify that these files exist:

```text
reference/reddit/predictions.jsonl
reference/questions/predictions.jsonl
artifacts/reddit/evaluation.npz
artifacts/questions/evaluation.npz
```

Then run:

```bash
python scripts/evaluate_saved_f1.py --dataset reddit
python scripts/evaluate_saved_f1.py --dataset questions
```

The code-package README documents single-run Macro-F1 values of **96.7702% (Reddit)** and **93.8160% (Questions)** for its saved predictions. **These are not the manuscript's five-run mean results**. The revised manuscript reports **93.02 ± 0.27% (Reddit)** and **91.10 ± 0.33% (Question)**. Do not compare or interchange these figures without confirming identical data versions, checkpoints, split/sampling configurations, and evaluation implementations. These figures have different reporting contexts; users should verify the exact data split, model checkpoint, and metric implementation before making direct comparisons.

![Saved test-result illustration from code package](docs/results.png)

## 6. Experimental protocol in the revised manuscript

Section 4.1 of the revised manuscript states the following experiment design:

- **Target-node splits:** 40% train / 20% validation / 40% test.
- **Transductive sequence construction:** DFS over the full graph; query and training nodes may share neighbors, but validation/test labels are excluded from training, retrieval references and memory updates.
- **Sequence length:** 7; approximately 9 sequences per node for inference.
- **Retrieval:** top-`K = 5`; reference KG and memory bank restricted to **training-target sequences** at validation/test time.
- **Model:** DeepSeek-R1-Distill-Qwen-14B, frozen during alignment; fixed vector-quantization codebook initialized on training-partition data.
- **Quantization:** codebook size `Kc = 512`, embedding dimension `dc = 256`.
- **Memory bank:** maximum 10,000 sequences; retrieval weighting coefficients `lambda1 = 0.1`, `lambda2 = 0.5`, `lambda3 = 0.01`.
- **Decision threshold:** `tau = 0.5`.
- **Reported metrics:** AUC-ROC, AUC-PR and Macro-F1; the manuscript reports the mean and standard deviation of five runs.

This section summarizes the **paper's experimental specification**. The available scripts/configurations may correspond to a narrower or different release; consult the source files and validate with the supplied artifacts before claiming exact paper-wide replication.

## 7. Reproducibility scope and interpretation

SCRAD's revised manuscript describes a fixed **two-shot coherence-classification prompt** with one coherent and one incoherent demonstration. Appendix A shows an Amazon experiment example with sequence length `r = 7` and retrieval size `K = 5`, including a query, five retrieved references, KG structural paths and an observed LLM coherence label.

The appendix displays **codebook indices** to make quantized embeddings readable. These indices identify fixed codebook vectors and **are not LLM vocabulary token IDs**. Selected vectors are projected to the LLM input-embedding space. The supplied code bundle includes quantization, KG, memory and LLM-related modules under `src/scrad_complete/`, but **does not establish that the exact Amazon Appendix A record can be reproduced from this two-dataset release**.

The full manuscript reports six datasets. The provided ready-to-run scripts and processed-resource documentation in this repository are currently scoped to **Reddit** and **Questions**. Training on the other four datasets requires additional processed graphs and validated configurations/implementation support.

## 8. Repository layout

```text
SCRAD/
├── README.md
├── requirements.txt
├── pyproject.toml
├── downloads.json
├── train.py
├── infer.py
├── evaluate.py
├── configs/
│   ├── reddit.json
│   └── questions.json
├── src/
│   ├── scrad/
│   └── scrad_complete/
├── scripts/
│   └── evaluate_saved_f1.py
└── docs/
    └── results.png
```

`data/`, `weights/`, `artifacts/`, `models/`, and `reference/` are **external-resource destinations**; they are not bundled inside the source ZIP.

## 9. References and attribution

- **SCRAD manuscript:** *Aligning LLMs with Graph Anomaly Detection via Retrieval-Augmented Sequence Coherence Discrimination* (revised manuscript).
- **Reddit graph background:** Hamilton, Ying and Leskovec, *Inductive Representation Learning on Large Graphs*, NeurIPS 2017 (manuscript [23]); Zeng et al., *GraphSAINT: Graph Sampling Based Inductive Learning Method*, ICLR 2020 (Reddit2 upstream source).
- **Questions:** Platonov et al., *A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really Making Progress?*, ICLR 2023 (manuscript [24]).
- **Elliptic:** Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics*, KDD 2019 (manuscript [13]).
- **Heal-Fraud:** Ma et al., *Fighting Against Organized Fraudsters Using Risk Diffusion-Based Parallel Graph Neural Network*, IJCAI 2023 (manuscript [25]).
- **Amazon:** McAuley and Leskovec, *From Amateurs to Connoisseurs: Modeling the Evolution of User Expertise through Online Reviews*, WWW 2013 (manuscript [14]).
- **Epinions/GADBench:** Tang et al., *GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection*, NeurIPS 2023 (manuscript [26]).

Dataset licenses and upstream model licenses are separate from the repository's source-code license. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). **Code provenance:** The `NOTICE` included in this release describes the software as an **independent reconstruction**. This README does not override that notice or assert that the implementation reproduces every experiment in the manuscript. Please review `NOTICE` and the source license for the applicable attribution.
