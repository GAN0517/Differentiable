# D5PL Workflow

D5PL adds a physics-consistency loss to a pure deep-learning AOD rollout. By default, the physical tendency constrains training but does not directly enter the forward prediction state. The workflow is:

1. **Load configuration**: Read data locations, time ranges, training settings, physical-constraint weights, and output settings from `Difference/settings.py`.
2. **Prepare data**: Load normalized NetCDF data, organize AOD, wind, meteorological, and radiation variables, and construct continuous multi-step samples.
3. **Calculate two tendency paths**: Predict the AOD tendency with the deep-learning network and independently calculate advection, compression, and diffusion tendencies from the current AOD and wind fields.
4. **Advance the forward state**: By default, use the deep-learning tendency to update the next AOD state and feed the updated AOD into the following forecast step.
5. **Apply the physics constraint**: Compare the deep-learning-advanced state with the physics-advanced state and use their difference as a physics-residual loss alongside the supervised prediction loss.
6. **Train and evaluate**: Aggregate supervised and physics-consistency losses over all forecast steps, perform training, validation, and early-stopping selection, and export evaluation results.
7. **Interpret the model**: Use the SHAP module under `Difference/analysis` to analyze the contributions of the input variables to the predicted tendency.

Main modules:

- `Difference/physics`: Spherical advection, compression, and diffusion calculations.
- `Difference/dl`: Dataset and deep-learning models.
- `Difference/models`: Tendency fusion and state update.
- `Difference/train`: Multi-step supervised losses, physics constraints, training, and evaluation.
- `Difference/data`: Physical-tendency data-construction utilities.
- `Difference/analysis`: Model interpretation and result analysis.
