#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : grid_search.py
# description     : Exhaustive grid search over BERTopic hyperparameters.
#                   Saves results to EVAL/{dataset}/GS_results_{condition}_{sent_suffix}.csv
#                   which is exactly the file the MOSAIC_pipeline notebook reads when
#                   param_selection = "grid_search".
# usage           : python grid_search.py --dataset trial --condition Report_meditazione_cleaned --sentences
# ==============================================================================

import argparse
import csv
import importlib
import os
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_str = str(Path(script_dir).parent.parent)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

try:
    from mosaic.model import run_bertopic
    from mosaic.preprocessing.preprocessing import split_sentences
    from mosaic.utils import get_params_grid
except ImportError:
    from src.mosaic.model import run_bertopic
    from src.mosaic.preprocessing.preprocessing import split_sentences
    from src.mosaic.utils import get_params_grid

os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

RESULTS_COLUMNS = [
    'n_components', 'n_neighbors', 'min_dist',
    'min_cluster_size', 'min_samples',
    'embedding_coherence', 'coherence_score', 'n_topics',
]


def _load_config(dataset: str):
    try:
        config_module = importlib.import_module(f"mosaic.configs.{dataset}")
        print(f"Loaded config from mosaic.configs.{dataset}")
        return config_module.config
    except (ImportError, AttributeError):
        print(f"Config for '{dataset}' not found – using dreamachine defaults.")
        from mosaic.configs.dreamachine import config as default_config
        return default_config


def _find_data_file(project_root: Path, dataset: str, condition: str) -> Path:
    candidates = [
        project_root / "DATA" / "raw" / f"{condition}.csv",
        project_root / "DATA" / dataset / f"{dataset}_cleaned_API.csv",
        project_root / "DATA" / "preprocessed" / f"{dataset}_preprocessed.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Wildcard fallback in DATA/raw
    matches = list((project_root / "DATA" / "raw").glob(f"{dataset}*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No data file found for dataset='{dataset}', condition='{condition}'.\n"
        f"Checked: {[str(c) for c in candidates]}"
    )


def _load_texts(data_path: Path, use_sentences: bool) -> list:
    df = pd.read_csv(data_path)
    text_col = next(
        (c for c in ['cleaned_reflection', 'reflection_answer', 'Report', 'text'] if c in df.columns),
        None,
    )
    if not text_col:
        raise ValueError(f"No recognised text column in {list(df.columns)}")
    texts = df[text_col].dropna().reset_index(drop=True).tolist()

    if use_sentences:
        texts, _ = split_sentences(texts)
        texts = [s for s in texts if len(s.split()) >= 2]
        seen: set = set()
        texts = [s for s in texts if not (s in seen or seen.add(s))]

    print(f"Loaded {len(texts)} texts (sentences={use_sentences})")
    return texts


def run_grid_search(dataset: str, condition: str, use_sentences: bool, reduced: bool):
    project_root = Path(__file__).resolve().parent.parent.parent

    config = _load_config(dataset)
    data_path = _find_data_file(project_root, dataset, condition)
    print(f"✓ Using data file: {data_path}")

    texts = _load_texts(data_path, use_sentences)

    # Build embedding model and pre-compute embeddings once
    embedding_model = SentenceTransformer(config.transformer_model)
    print("Generating embeddings …")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    vectorizer_model = CountVectorizer(
        ngram_range=getattr(config, 'ngram_range', (1, 2)),
        stop_words=list(getattr(config, 'extended_stop_words', 'english')),
        max_df=getattr(config, 'max_df', 0.95),
        min_df=getattr(config, 'min_df', 2),
        lowercase=True,
    )
    top_n_words = getattr(config, 'top_n_words', 15)

    umap_combinations, hdbscan_combinations = get_params_grid(config, condition, reduced=reduced)

    # Output path expected by the notebook:
    # EVAL/{dataset}/GS_results_{condition}_{sent_suffix}.csv
    sent_suffix = 'sentences' if use_sentences else ''
    out_dir = project_root / "EVAL" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"GS_results_{condition}_{sent_suffix}.csv"

    total = len(umap_combinations) * len(hdbscan_combinations)
    print(f"Running {total} combinations – results → {results_path}")

    with open(results_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()

        run_idx = 0
        for n_comp, n_neigh, min_dist in umap_combinations:
            for min_cls, min_samp in hdbscan_combinations:
                run_idx += 1
                print(
                    f"[{run_idx}/{total}] "
                    f"n_comp={n_comp} n_neigh={n_neigh} min_dist={min_dist} "
                    f"min_cls={min_cls} min_samp={min_samp}",
                    end=" … ",
                    flush=True,
                )
                try:
                    _, topics, cv_score, emb_coh = run_bertopic(
                        data=texts,
                        embeddings=embeddings,
                        vectorizer_model=vectorizer_model,
                        embedding_model=embedding_model,
                        n_neighbors=n_neigh,
                        n_components=n_comp,
                        min_dist=min_dist,
                        min_cluster_size=min_cls,
                        min_samples=min_samp,
                        top_n_words=top_n_words,
                        random_seed=42,
                    )
                    n_topics = len(set(t for t in topics if t != -1))
                    print(f"emb_coh={float(emb_coh):.4f} cv={float(cv_score):.4f} topics={n_topics}")
                    writer.writerow({
                        'n_components': n_comp,
                        'n_neighbors': n_neigh,
                        'min_dist': min_dist,
                        'min_cluster_size': min_cls,
                        'min_samples': min_samp,
                        'embedding_coherence': float(emb_coh),
                        'coherence_score': float(cv_score),
                        'n_topics': n_topics,
                    })
                    f.flush()
                except Exception as e:
                    print(f"FAILED ({e})")

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exhaustive grid search over BERTopic hyperparameters."
    )
    parser.add_argument('--dataset', type=str, default='trial',
                        help="Dataset name (e.g. trial, dreamachine)")
    parser.add_argument('--condition', type=str, required=True,
                        help="Condition name (e.g. Report_meditazione_cleaned, DL, HS)")
    parser.add_argument('--sentences', action='store_true',
                        help="Split documents into sentences before modelling")
    parser.add_argument('--reduced', action='store_true',
                        help="Use reduced parameter grid (fast smoke-test)")
    args = parser.parse_args()

    run_grid_search(
        dataset=args.dataset,
        condition=args.condition,
        use_sentences=args.sentences,
        reduced=args.reduced,
    )
