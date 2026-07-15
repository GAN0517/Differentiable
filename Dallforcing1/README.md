# Dallforcing1 Workflow

Dallforcing1 combines a differentiable physical core with a deep-learning tendency module to advance the AOD state. The workflow is:

1. **Load configuration**: Read data locations, time ranges, training settings, and physical-process settings from `Difference/settings.py`.
2. **Prepare data**: Load normalized NetCDF data, organize AOD, wind, meteorological, and radiation variables, and create chronological training, validation, and evaluation samples.
3. **Advance the physical state**: Use the current AOD and wind fields to calculate physical tendencies and the physics-advanced state through semi-Lagrangian advection, wind-field compression, and diffusion.
4. **Estimate the DL correction**: Apply the process-decomposed network to the physics-advanced state to estimate source and deposition tendencies that compensate for unresolved processes. During multi-step prediction, only AOD is updated recursively, while the background forcing variables remain anchored.
5. **Update the coupled state**: Combine the physical and deep-learning tendencies to obtain the next AOD state. The complete update remains differentiable and participates in end-to-end training.
6. **Train and evaluate**: Train and validate the model chronologically, select the checkpoint through early stopping, and export predictions and process components for evaluation.
7. **Interpret the model**: Use the modules under `Difference/analysis` to perform SHAP analysis for the total tendency and the source and deposition components.

Main modules:

- `Difference/data`: Physical-tendency data construction.
- `Difference/physics`: Spherical physical operators and AOD advancement.
- `Difference/dl`: Dataset and deep-learning models.
- `Difference/models`: Coupling of physical and deep-learning tendencies.
- `Difference/train`: Losses, training, validation, and result export.
- `Difference/analysis`: Model interpretation and result analysis.
