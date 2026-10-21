##  Conclusion and reporting notes

Use the actual `results_df` output to write your findings. Do not copy example metrics from earlier runs.

In your conclusion, report:
1. Which model performed best on the held-out test set and its MAE, RMSE, and R².
2. Which features had the highest Random Forest importance.
3. Important limitations: historical data may not represent current markets; missing scores are imputed; high-sales games are difficult to predict; and feature importance is not proof of causation.
4. The prediction target is global sales in millions. Regional sales were excluded from predictors to reduce target leakage.

**Reproducibility:** the split and models use `random_state=42`. Restart the kernel and run all cells from top to bottom before submitting.
