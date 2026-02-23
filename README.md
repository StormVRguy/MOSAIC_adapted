# MOSAIC: Mapping Of Subjective Accounts into Interpreted Clusters

A comprehensive topic modeling pipeline for consciousness-related textual data across multiple datasets, using BERTopic, BERT embeddings, and UMAP-HDBSCAN clustering with multilingual support.

## Overview

MOSAIC is a research framework that analyzes subjective experiential reports from various consciousness studies through:
- Advanced NLP with BERT embeddings and multilingual models
- Dimensionality reduction via UMAP
- Density-based clustering with HDBSCAN
- Hyperparameter optimization with Optuna
- Topic coherence optimization
- Large Language Model integration with Llama CPP for deeper insights
- Support for multiple datasets and languages

## Supported Datasets

This repository contains analysis pipelines for several consciousness research datasets:

- **Dreamachine**: Stroboscopic light-induced altered states of consciousness
- **Inner Speech**: Japanese phenomenological reports on inner speech experiences
- **Depression/MPE**: Mental health and psychological experience reports  
- **NDE**: Near-death experience accounts
- **Ganzfeld**: Sensory deprivation experimental reports

### Downloading or obtaining ready-to-use data

**This repository does not ship or host any of the above datasets.** The code expects you to place your own CSV files under `DATA/` (see [Running the pipeline on your own trial database](#running-the-pipeline-on-your-own-trial-database)). There is no built-in download script for Dreamachine, Inner Speech, NDE, Ganzfeld, or MPE.

Ways to get data that is (or can be made) ready for analysis:

| Source | What to do |
|--------|------------|
| **Dreamachine / Perception Census** | The Dreamachine programme and [Perception Census](https://perceptioncensus.dreamachine.world/) (University of Sussex / Glasgow) have collected large-scale experiential reports. Access and use of data are subject to their policies. For data or collaboration enquiries, contact **perceptioncensus@sussex.ac.uk** or the authors (e.g. r.beaut@sussex.ac.uk). |
| **Your own trial or survey** | Export open-text responses to a CSV with a column named `cleaned_reflection`, `reflection_answer`, or `text`, then follow the [trial-database instructions](#running-the-pipeline-on-your-own-trial-database). |
| **Other public datasets** | Any CSV of short documents (e.g. survey responses, reviews, narratives) with one of the supported text column names can be dropped into `DATA/<dataset_name>/` and run through the pipeline. You can search [Hugging Face Datasets](https://huggingface.co/datasets) or discipline-specific repositories (e.g. OSF, Zenodo) for “experiential reports”, “free text”, or “survey responses” and adapt column names if needed. |
| **Paper / supplementary materials** | The MOSAIC methodology is described in *Beauté, R. et al. (2024). Analysing the phenomenology of stroboscopically induced phenomena using natural language topic modelling*. If a paper or preprint links to supplementary data or a replication package, that may provide example or benchmark data in a compatible format. |

To **try the pipeline quickly** without real study data, you can create a small CSV (e.g. a few dozen rows) with a `text` or `reflection_answer` column containing short paragraphs, save it as `DATA/demo/demo.csv`, add a minimal config (see template), and run the pipeline with `param_selection = "default"`.

## Project Structure

```
MOSAIC/
├── src/                           # Core functionality
│   ├── preprocessor.py            # Text cleaning, sentence splitting
│   ├── model.py                   # BERTopic configuration
│   ├── utils.py                   # Metrics and helper functions
│   ├── optuna_search.py           # Hyperparameter search with Optuna
│   └── optuna_search_allmetrics.py # Multi-objective optimization
├── configs/                       # Dataset-specific configurations
│   └── dreamachine2.py           # Dreamachine dataset settings
├── preproc/                       # Data preprocessing utilities
│   ├── prepare_data.ipynb        # Data preparation notebook
│   └── preprocess_data_*.ipynb   # Dataset-specific preprocessing
├── scripts/                       # Analysis notebooks and tools
├── EVAL/                          # Model evaluation and analysis
│   ├── dreamachine/              # Dreamachine-specific evaluations
│   │   ├── demographics.ipynb    # Demographic analysis
│   │   └── stability_tests/      # Model stability testing
│   ├── conditions_similarity.ipynb # Cross-condition comparisons
│   └── optuna_search/            # Hyperparameter optimization results
├── MULTILINGUAL/                  # Multilingual analysis pipeline
│   ├── DREAMACHINE/              # Multilingual Dreamachine analysis
│   ├── INNERSPEECH/              # Japanese inner speech analysis
│   │   ├── app.py               # Streamlit dashboard
│   │   ├── app_hosted.py        # Hosted version of dashboard
│   │   └── local_translator.py  # Local translation utilities
│   ├── translate/                # Translation utilities
│   └── prepare_data.ipynb        # Multilingual data preparation
├── DATA/                          # Local data storage
├── pyproject.toml                # Project configuration and dependencies
├── requirements.txt              # Python dependencies
└── .mosaicvenv/                  # Virtual environment
```

## Key Features

### Core Analysis Pipeline
- **Preprocessing**: Text cleaning, sentence splitting, duplicate removal
- **Embedding**: Support for multiple transformer models (Qwen, E5, BGE, etc.)
- **Clustering**: UMAP dimensionality reduction + HDBSCAN clustering  
- **Topic Modeling**: BERTopic with custom representation models
- **Evaluation**: Coherence metrics, stability testing, bootstrap analysis

### Multilingual Support
- Translation pipelines for non-English datasets
- Support for Japanese text processing
- API-based and local translation options using Llama models

### Interactive Dashboards
- Streamlit applications for real-time analysis ([`MULTILINGUAL/INNERSPEECH/app.py`](MULTILINGUAL/INNERSPEECH/app.py))
- Parameter tuning interfaces
- Visualization tools with datamapplot integration

### Hyperparameter Optimization
- Optuna-based search for optimal model parameters
- Multi-objective optimization across multiple metrics
- Dataset-specific parameter spaces

## Installation

1. Clone the repository:
```bash
git clone https://github.com/romybeaute/MOSAIC.git
cd MOSAIC
```

2. Create and activate virtual environment:
```bash
python3 -m venv .mosaicvenv
source .mosaicvenv/bin/activate  # On Windows: .mosaicvenv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

This will install all dependencies specified in [`pyproject.toml`](pyproject.toml) and make the MOSAIC package available for import.

## Usage

### Basic Analysis
The primary way to use MOSAIC is through Jupyter notebooks in the [`scripts/`](scripts/) directory and dataset-specific folders:

```bash
# Navigate to analysis notebooks
cd EVAL/dreamachine/
jupyter lab demographics.ipynb
```

### Hyperparameter Optimization
Run hyperparameter search using Optuna:

```bash
python src/optuna_search.py --dataset dreamachine --condition DL --sentences --n_trials 200
```

Parameters:
- `--dataset`: Dataset name (dreamachine, innerspeech, etc.)
- `--condition`: Experimental condition (HS, DL, HW)
- `--sentences`: Enable sentence-level analysis
- `--n_trials`: Number of optimization trials

### Interactive Dashboard
Launch the Streamlit dashboard for interactive analysis:

```bash
streamlit run MULTILINGUAL/INNERSPEECH/app.py
```

### Translation Pipeline
For multilingual datasets, use the translation utilities:

```bash
python MULTILINGUAL/translate/local_translator.py \
    --dataset nde \
    --input-csv NDE_reflection_reports.csv \
    --text-column reflection_answer \
    --model llama \
    --task translate \
    --num-samples 100
```

## Configuration

Dataset-specific configurations are stored in the [`configs/`](configs/) directory. Example configuration for Dreamachine dataset in [`configs/dreamachine2.py`](configs/dreamachine2.py):

```python
class DreamachineConfig:
    def __init__(self):
        self.transformer_model = "Qwen/Qwen3-Embedding-0.6B"
        self.ngram_range = (1, 3)
        self.top_n_words = 15
        # ... other parameters
```

## Running the pipeline on your own trial database

To run thematic (topic) analysis on a new dataset (e.g. a trial database of free-text responses), follow these steps.

### 1. Prepare your data

- **Format**: One CSV file with at least one **text column**.
- **Supported column names** (checked in this order): `cleaned_reflection`, `reflection_answer`, or `text`. Use one of these, or the pipeline will fail with a clear error.
- **Content**: One row per report (e.g. per participant response). The pipeline can split documents into sentences internally; you can also pre-split if you prefer.
- **Where to put the file**:
  - **Option A (recommended)**: `DATA/<dataset_name>/<your_file>.csv`  
    Example: `DATA/mytrial/mytrial_cleaned.csv`
  - **Option B**: `DATA/preprocessed/<dataset_name>_preprocessed.csv`  
    The Optuna script looks for names like `{dataset}_cleaned_API.csv`, `{dataset}_preprocessed.csv`, etc.

So for a dataset named `mytrial`, create `DATA/mytrial/` and put your CSV there (e.g. `mytrial_cleaned.csv` or `mytrial_cleaned_API.csv`).

### 2. Add a dataset config

Copy the template and adapt it for your trial:

```bash
cp src/mosaic/configs/template.py src/mosaic/configs/mytrial.py
```

Edit `src/mosaic/configs/mytrial.py`:

- Set `self.name = "mytrial"` (must match the folder name under `DATA/`).
- Set `self.transformer_model` (e.g. `"Qwen/Qwen3-Embedding-0.6B"` or `"sentence-transformers/all-mpnet-base-v2"`).
- Adjust `ngram_range`, `max_df`, `min_df`, `top_n_words` if needed.
- In `get_default_params(condition)` and `_get_full_params` / `_get_reduced_params`, define a **condition** that matches how you run the pipeline (e.g. one condition like `"main"` or several like `"arm1"`, `"arm2"`). The pipeline will use `condition` for loading data and saving results.

If your CSV has a different text column name, either rename the column to one of `cleaned_reflection`, `reflection_answer`, or `text`, or add that name to the list in `optuna_search.py` (`load_data`) and in the notebook where the CSV is read.

### 3. Run the pipeline from the notebook

1. **Install and kernel**: From the repo root, create/activate the venv and install the package (see [Installation](#installation)). Select the `.mosaicvenv` (or equivalent) kernel in Jupyter.

2. **Open the pipeline notebook**:  
   `notebooks/1_Run_pipeline/MOSAIC_pipeline.ipynb` or `MOSAIC_pipeline_extended.ipynb`.

3. **Set dataset and condition** in the first cells:
   - `dataset = "mytrial"`  (same as `self.name` in config and `DATA/mytrial/`).
   - `condition = "main"`  (or whatever you defined in your config; this is used for filenames and default params).

4. **Point the notebook at your data**:
   - The current notebooks assume a **Box** path like `~/Library/CloudStorage/Box-Box/TMDATA/<dataset>/`. For a local trial, change the data path to use the repo’s `DATA` folder, e.g.:
     - `DATA_DIR = os.path.join(project_root, "DATA", dataset)`  
     and ensure your CSV lives under `DATA/mytrial/` with a name the notebook will use, e.g. `main_reflections_APIcleaned.csv` or the same column names as above.
   - Alternatively, set `reports_path = os.path.join(project_root, "DATA", dataset, "mytrial_cleaned.csv")` and in the load cell use:
     - `df_reports = pd.read_csv(reports_path)[['cleaned_reflection'] or ['reflection_answer'] or ['text']].dropna().reset_index(drop=True)` (pick the column you have).

5. **Project root and imports**: The notebook should run from the **project root** (where `pyproject.toml` and `src/` are), or you must set `project_root` and `sys.path` so that `mosaic` is importable (e.g. `sys.path.insert(0, project_root)` after `pip install -e .`). Imports should use the installed package: `from mosaic.preprocessing.preprocessing import split_sentences`, `from mosaic.utils import ...`, `from mosaic.model import ...`, `from mosaic.configs.mytrial import config`.

6. **Parameter source**: In the “Train BERTopic model” section, set `param_selection = "default"` to use your config’s default UMAP/HDBSCAN parameters. That avoids needing Optuna or grid-search results for a first run.

7. Run all cells. The pipeline will: load text → (optionally) split into sentences → preprocess (min words, dedup) → embed → cluster (UMAP + HDBSCAN) → fit BERTopic → produce topic labels and visualisations.

### 4. (Optional) Hyperparameter search with Optuna

For better topics you can run Optuna first, then use the best trial in the notebook:

```bash
# From repo root, with venv activated
python -m mosaic.optuna_search --dataset mytrial --condition main --sentences --n_trials 50
```

- Optuna reads from `DATA/mytrial/` (or `DATA/preprocessed/`) using the same file discovery as above and the same text column logic (`cleaned_reflection`, `reflection_answer`, `text`).
- It uses your config from `mosaic.configs.mytrial` if `--use-config` is set (default).
- Results are written under `results/optuna/mytrial/` (path may vary; see script output). The notebook currently expects results under `EVAL/<dataset>/optuna_search/` and a CSV with a column named `embedding_coherence`. If your Optuna run writes different column names (e.g. `objective_embed_coherence`), either copy/rename the Optuna CSV into `EVAL/mytrial/optuna_search/OPTUNA_results_main_sentences_<sanitized_model>.csv` and ensure an `embedding_coherence` column exists (e.g. copy from `objective_embed_coherence`), or change the notebook to sort by the column your Optuna file actually has.

### 5. Outputs

- **Topic model**: Fitted BERTopic model and topic assignments.
- **Visualisations**: Document-topic plots, topic bar charts, hierarchy (and in extended pipeline, datamapplot, etc.).
- **Exports**: The notebook can save topic summaries to CSV (path set in the “save final topic summary” cell); adjust `summary_results_file` to a path under `RESULTS/` or `EVAL/mytrial/` as you prefer.

### Quick checklist for a new trial

| Step | Action |
|------|--------|
| 1 | CSV with column `cleaned_reflection`, `reflection_answer`, or `text` in `DATA/<dataset_name>/` |
| 2 | Copy `template.py` → `src/mosaic/configs/<dataset_name>.py`, set `name`, `transformer_model`, and conditions |
| 3 | In notebook: set `dataset`, `condition`, data path to `DATA/<dataset_name>/...`, imports from `mosaic.*` |
| 4 | Use `param_selection = "default"` for first run; optionally run Optuna and wire results into the notebook |
| 5 | Run notebook and inspect/save topic summaries and figures |

## Development

The project uses a modular structure:
- Core functionality in [`src/`](src/)
- Dataset-specific code in respective folders
- Shared utilities in [`DATA/helpers.py`](DATA/helpers.py)
- Path management through `mosaic.path_utils`

## Citation

If using this code, please cite:
- Beauté, R. et al. (2024). Analysing the phenomenology of stroboscopically induced phenomena using natural language topic modelling
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure

## License

See [`LICENSE`](LICENSE) for details.