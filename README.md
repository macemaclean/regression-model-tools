# Regression model tools

Python scripts for regression models, using the Scikit-Learn framework:
* Diagnostic plots
* Approximate Shapley values
* Bootstrapped error margins for predictions

## Diagnostic plots
While ML models do not generally have the same residual distribution assumptions as for classical linear regression, there is still value in examining residual plots.

```python
import lightgbm as lgb
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from regression_diagnostics import RegressionDiagnostics
import warnings
warnings.filterwarnings('ignore')

# Load the boston house-prices dataset and fit a regression model
boston = load_boston()

X = pd.DataFrame(boston["data"], columns=boston.feature_names)
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)

# Generate diagnostic plots
diagnostics = RegressionDiagnostics(lgb_model)
diagnostics.fit(X_test, y_test)
```
```python
# Fitted values against actual values
diagnostics.fitted_actual()
```
![Fitted values against actual values](https://github.com/macemaclean/regression-model-tools/blob/master/docs/images/diagnostics_fitted_actual.png)
```python
# Residuals against fitted values
diagnostics.residuals_fitted()
```
![Residuals against fitted values](https://github.com/macemaclean/regression-model-tools/blob/master/docs/images/diagnostics_residuals_fitted.png)
```python
# Histogram of residuals
diagnostics.hist_residuals()
```
![Residuals against fitted values](https://github.com/macemaclean/regression-model-tools/blob/master/docs/images/diagnostics_residuals_histogram.png)
```python
# QQ plot of residuals
diagnostics.qq_plot()
```
![Residuals against fitted values](https://github.com/macemaclean/regression-model-tools/blob/master/docs/images/diagnostics_qq_plot.png)
