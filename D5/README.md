# D5 Workflow

D5 is a reference model that performs multi-step AOD prediction using only a deep-learning tendency. The physical tendency does not participate in the forward state update. The workflow is:

1. **Load configuration**: Read data locations, time ranges, training settings, and output settings from `Difference/settings.py`.
2. **Prepare data**: Load normalized NetCDF data, organize AOD, meteorological, and radiation variables, and construct continuous temporal windows.
3. **Initialize the rollout**: Use AOD at the beginning of each window as the initial recursive state and prepare the corresponding background variables.
4. **Predict the DL tendency**: Estimate the AOD tendency from the current AOD and background variables without adding a physical tendency to the forward calculation.
5. **Advance multiple steps**: Update the next AOD state with the predicted tendency and feed the updated AOD into the following prediction step until the requested forecast horizon is reached.
6. **Train and evaluate**: Aggregate the supervised losses over all forecast steps, perform chronological training, validation, and evaluation, and save checkpoints and prediction outputs.
7. **Interpret the model**: Use the SHAP module under `Difference/analysis` to quantify the contributions of the input variables to the predicted tendency.

Main modules:

- `Difference/dl`: Dataset and deep-learning models.
- `Difference/models`: Tendency-based AOD state update.
- `Difference/train`: Multi-step losses, training, validation, and result export.
- `Difference/physics` and `Difference/data`: Retained physical-computation and data-construction utilities.
- `Difference/analysis`: Model interpretation and result analysis.
