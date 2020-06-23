"""
Diagnostics for regression models
"""

import numpy as np
import pandas as pd

from plotnine import ggplot, geom_point, geom_abline, geom_vline, \
    geom_histogram, aes, labs, theme
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score

class RegressionDiagnostics(BaseEstimator):
    """Class to generate diagnostic plots for a regression model
    """
    def __init__(self, model):
        """Initialise diagnostics object

        Parameters
        ----------
        model : regressor (or pipeline)
            Model to evaluate
        """
        self.model = model

        # Initialise other attributes
        self.X = None
        self.y = None
        self.yhat = None
        self.r2_score = None
        self.df = None

    def fit(self, X, y):
        """Build a data frame of y and predicted y (yhat) and calculate
        residuals

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input array/data frame for self.model
        y : array-like of shape (n_samples, 1)
            Observed values

        Returns
        -------
        self
        """
        self.X = X
        if isinstance(y, pd.DataFrame):
            # Convert y to array if data frame passed
            self.y = y.values.reshape(-1)
        else:
            self.y = y

        self.yhat = self.model.predict(X)
        self.r2_score = r2_score(y_true=y, y_pred=self.yhat)

        # Combine y and yhat into a data frame and calculate residuals
        # (preserve index from X input)
        self.df = pd.DataFrame(zip(self.y, self.yhat),
                               columns=["y", "yhat"],
                               index=self.X.index).assign(
                                   residual=lambda x: x["y"] - x["yhat"])
        return self

    def fitted_actual(self, figure_size=(4, 4), sample_frac=1.0):
        """Plot fitted values against actual values

        Parameters
        ----------
        figure_size : tuple(int, int), optional default=(4, 4)
            Plot size (width, height)

        sample_frac : float, optional default=1.0
            Fraction of data points to plot

        Returns
        -------
        plot : ggplot object
        """
        return (ggplot(self.df.sample(frac=sample_frac),
                       aes(x="y", y="yhat")) +
                geom_point(alpha=0.25) +
                geom_abline(slope=1, intercept=0,
                            color="red", linetype="dashed") +
                labs(title="Fitted vs Actual (R2 = {:.3f})".format(
                    self.r2_score),
                     x="Actual",
                     y="Fitted") +
                theme(figure_size=figure_size))


    def residuals_fitted(self, figure_size=(4, 4), sample_frac=1.0):
        """Plot residuals against fitted values

        Parameters
        ----------
        figure_size : tuple(int, int), optional default=(4, 4)
            Plot size (width, height)

        sample_frac : float, optional default=1.0
            Fraction of data points to plot

        Returns
        -------
        plot : ggplot object
        """
        return (ggplot(self.df.sample(frac=sample_frac),
                       aes(x="yhat", y="residual")) +
                geom_point(alpha=0.25) +
                geom_abline(slope=0, intercept=0,
                            color="red", linetype="dashed") +
                labs(title="Residuals vs fitted",
                     x="Fitted",
                     y="Residuals") +
                theme(figure_size=figure_size))

    def hist_residuals(self, figure_size=(8, 4), sample_frac=1.0):
        """Histogram of residuals

        Parameters
        ----------
        figure_size : tuple(int, int), optional default=(8, 4)
            Plot size (width, height)

        sample_frac : float, optional default=1.0
            Fraction of data points to plot

        Returns
        -------
        plot : ggplot object
        """
        return (ggplot(self.df.sample(frac=sample_frac),
                       aes(x="residual")) +
                geom_histogram(fill="lightblue", colour="grey") +
                geom_vline(xintercept=0,
                           color="red", linetype="dashed") +
                labs(title="Residuals",
                     x="Residuals") +
                theme(figure_size=figure_size))

    def qq_plot(self, figure_size=(6, 4), sample_frac=1.0):
        """QQ plot of residuals

        Parameters
        ----------
        figure_size : tuple(int, int), optional default=(6, 4)
            Plot size (width, height)

        sample_frac : float, optional default=1.0
            Fraction of data points to plot

        Returns
        -------
        plot : ggplot object
        """
        # Normal distribution quantiles
        q = stats.norm.ppf(
            [(x + 1) / (len(self.y) + 1) for x in range(len(self.y))])

        # Get gradient and intercept of QQ line
        r_quantiles = np.quantile(self.df.residual, [0.25, 0.75])
        norm_quantiles = stats.norm.ppf([0.25, 0.75])
        qq_grad = (r_quantiles[1] - r_quantiles[0]) / (
            norm_quantiles[1] - norm_quantiles[0])
        qq_int = r_quantiles[0] - qq_grad * norm_quantiles[0]

        # data frame to hold the plot data
        qq = pd.DataFrame(zip(self.df.residual.sort_values(ascending=True), q),
                          columns=["x", "norm_q"])

        return (ggplot(qq.sample(frac=sample_frac),
                       aes(x="norm_q", y="x")) +
                geom_point(alpha=0.25) +
                geom_abline(intercept=qq_int, slope=qq_grad,
                            color="red", linetype="dashed") +
                labs(title="QQ Plot",
                     x="Normal Quantiles",
                     y="Sample Quantiles") +
                theme(figure_size=figure_size))

    def largest_residuals(self, n_residuals=10):
        """Return list of largest n residuals

        Parameters
        ----------
        n_residuals : int, optional default=10

        Returns
        -------
        residuals : Int64Index
        """
        return self.df.assign(
            residual_abs=lambda x: abs(x["residual"])).sort_values(
                by="residual_abs", ascending=False).drop(
                    "residual_abs", axis=1)[0:n_residuals]
