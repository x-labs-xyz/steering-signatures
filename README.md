# steering-signatures

Replication files for Signatures of Steerability in Activation Space of Language Models. The main entry points are:

- `steering_analysis_gemma_big.py`: downloads Model-Written Evaluation datasets, builds the multiple-choice prompts, caches layer activations to disk, extracts a steering vector, and runs the steering sweep that produces steerability results.
- `representation_association_search.py`: reads steering-result pickles plus cached activations, computes a focused set of representation metrics, and writes feature/correlation CSVs.
- `synthetic_superposition_suite.py`: runs the synthetic superposition sweep from command-line grid arguments, writes the raw sweep table, and then writes `corr_df` as the final output artifact.

Run any script from inside this folder with standard Python:

```bash
python steering_analysis_gemma_big.py --help
python representation_association_search.py --help
python synthetic_superposition_suite.py --help
```

Typical steering run:

```bash
python steering_analysis_gemma_big.py \
  --model google/gemma-3-4b-it \
  --dataset-category persona \
  --dataset-names power-seeking-2 political-neutrality \
  --vector-method diff_means \
  --activation-cache-dir activation_cache_torch
```

Steering options:

- `--model` selects the Hugging Face model to load.
- `--vector-method` controls how the steering vector is extracted. Supported values are `diff_means`, `scaled_diff_means`, and `fisher_mean`.
- `--layers` lets you restrict the run to specific layers; omit it to run all layers.
- `--results-path` chooses the pickle file used for incremental saving and resume.

Representation-metric run:

```bash
python representation_association_search.py \
  --steering-pkls google_gemma-3-4b-it_diff_means_steering_results.pkl \
  --cache-dir activation_cache_torch \
  --out-dir representational_correlations
```

`representation_association_search.py` only computes:

- `fisher_trace_ratio`
- `mean_cosine_alignment_diff`
- `glue_capacity`
- `twonn_intrinsic_dimension`

This produces:

- `activation_representation_metrics.csv`: per `(model, dataset, layer)` feature table.
- `correlations_all_models.csv` and `correlations_<model>.csv`: feature/steerability correlation summaries.

Synthetic superposition run:

```bash
python synthetic_superposition_suite.py \
  --d 32 64 \
  --n-over-d 1.5 2.5 4.0 \
  --epsilon 0.06 0.12 0.20 \
  --k-active 2 4 8 \
  --dist-gap 0.3 0.7 1.1 \
  --support-mode same_support different_support \
  --n-pos 256 \
  --n-neg 256 \
  --reps 3 \
  --out-dir synthetic_superposition_outputs
```


Synthetic outputs:

- `synthetic_superposition_results.csv`: one row per synthetic grid cell and replicate.
- `synthetic_superposition_params.json`: the grid and sweep settings used for the run.
- `synthetic_superposition_correlations.csv`: the `corr_df` summary saved as the final step.

