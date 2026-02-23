# Trial dataset for MOSAIC thematic analysis

- **File**: `trial_cleaned.csv`
- **Rows**: 111 reflections
- **Column**: `reflection_answer` (required by pipeline). Optional `topic` column (affection / engineering / sports) for reference only.
- **Constraint**: Each reflection has at most 70 characters when spaces are ignored.
- **Usage**: In the pipeline notebook set `dataset = "trial"`, point `DATA_DIR` to `DATA/trial`, and load `trial_cleaned.csv` (or use Optuna: it will discover `trial*.csv` here).
- **Regenerate**: Run `python generate_trial_data.py` from this directory (random seed 42 for reproducibility).
