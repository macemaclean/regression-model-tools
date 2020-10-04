"""
Bootstrap error and prediction intervals

Estimate bootstrapped error/prediction intervals for a data set based
on observed residuals in a reference data set
"""

import numpy as np
import pandas as pd
import umap

from category_encoders import TargetEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def boot_err(x, reps, alpha):
    """ Bootstrapped quantiles

    Calculate
        a) a bootstrapped quantile of the mean of x and
        b) a bootstrapped quantile of x

    Parameters
    ----------
    x : array-like, shape(,1)
        Input array
    reps : int
        Number of (re)samples to generate (int)
    alpha : float
        quantile (0 <= alpha <= 1)

    Returns
    -------
    quantiles : list[float, float]
        quantile for the bootstrapped mean of x, and quantile for x
    """
    # Checks
    if not isinstance(x, np.ndarray):
        raise Exception("Input 'x' must be an array")

    if (not isinstance(reps, int)) or (reps <= 0):
        raise Exception("Number of bootstrap replications must be a " + \
                        "positive integer")

    if (alpha < 0) or (alpha > 1):
        raise Exception("alpha must be in the range 0 <= alpha <= 1")

    # Generate <reps> resamples of x (with replacement)
    r = np.random.default_rng().choice(x, size=(reps, len(x)),
                                       replace=True, shuffle=False)

    # get <alpha> quantile for the mean of each row of resampled data
    q_err = np.quantile(np.mean(np.abs(r), 1), (alpha))

    # the the <alpha> quantile for (all of) the resampled data
    q_pred = np.quantile(np.abs(r), (alpha))

    return([q_err, q_pred])


def bootstrap_intervals(model, X_ref, y_ref, X_new,
                        categorical_features=[],
                        nn_weights=None,
                        reps=1000,
                        k=None,
                        alpha=0.95,
                        verbose=False,
                        umap_params=None):
    """
    Get bootstrapped error intervals for a regression model

    Calculate error and prediction intervals for a regression model, using
    reference data and return a data frame of predicted values and intervals
    for new data.
    If k is provided, intervals are calculated for each point using k nearest
    neighbours. Otherwise intervals are calculated across the entire
    reference data.

    Parameters
    ----------
    model : regressor
        The model (pipeline) used for predictions
    X_ref: array-like, shape(n_samples, n_features)
        X data to use for calculating error and prediction intervals
    y_ref : array_like, shape(n_samples, 1)
        y data to use for calculating error and prediction intervals
    X_new : array-like, shape(n_samples, n_features)
        New data to be predicted and intervals applied to
    categorical_features : list of strings
        categorical features in the model
    nn_weights : dict(str: number)
        dict of weights for each feature (used in the nearest
        neighbour matching) {feature: weight}
    reps : int
        number of bootstrap replications
    k : int
        number of data points to use for nearest neighbour matching.
        If k is None, calculate global average error margin.
    alpha : float
        quantile to calculate intervals (0 <= alpha <= 1)
    verbose : boolean
        Print quantiles (only applies if nearest neighbour method is
                                not used)

    Returns
    -------
    output : pandas DataFrame
        Return predicted values for X_new along with error margins and
        prediction intervals
    """
    # Checks
    if (not isinstance(reps, int)) or (reps <= 0):
        raise Exception("Number of bootstrap replications must be a " + \
                        "positive integer")

    if (alpha < 0) or (alpha > 1):
        raise Exception("alpha must be in the range 0 <= alpha <= 1")

    # If k is None or k >= count of data points then use all data
    use_all = (k is None or k >= len(y_ref))

    # Get predicted values and residuals for the reference data
    yhat_ref = model.predict(X_ref).reshape(-1,)
    residuals_ref = yhat_ref - np.array(y_ref).reshape(-1)

    # Get predicted values for the new data
    yhat_new = model.predict(X_new).reshape(-1,)

    if use_all:
        # Calculate global error (average error across all data points)

        # Return mean of absolute values from each bootstrap sample
        q_err = boot_err(residuals_ref, reps, alpha)

        result = pd.DataFrame(yhat_new, columns=["yhat"])
        result = result.assign(
            lower_err=yhat_new - q_err[0],
            upper_err=yhat_new + q_err[0],
            lower_pred=yhat_new - q_err[1],
            upper_pred=yhat_new + q_err[1])

        # Print quantiles if requested
        if verbose:
            print('{:2.2%} Quantile for mean absolute error: {:-4f}'.format(
                alpha, q_err[0]))
            print('{:2.2%} Quantile for prediction error: {:-4f}'.format(
                alpha, q_err[1]))
    else:
        # Calculate local error (average error for each data point)

        # Standardise X_ref for nearest neighbour matching

        # Build individual transformers for each feature so that weights
        # can be applied
        transformers_cat = [(x,
                             Pipeline(steps=[
                                 ("target", TargetEncoder()),
                                 ("scaler", MinMaxScaler())
                             ]),
                             [x]) for x in categorical_features]

        non_categorical_features = [x for x in X_ref.columns
                                    if x not in categorical_features]

        transformers_other = [(x,
                               MinMaxScaler(),
                               [x]) for x in
                              non_categorical_features]

        # Build preprocessor
        prep_nn = ColumnTransformer(transformers_cat + transformers_other,
                                    transformer_weights=nn_weights)

        X_ref_nn = prep_nn.fit_transform(X_ref, y_ref)
        X_new_nn = prep_nn.transform(X_new)

        # Reduce dimensionality using UMAP
        num_features = X_ref.shape[1]
        
        if num_features >= 10:
            if umap_params is None:
                umap_params = {
                    "n_neighbors": 15,
                    "min_dist": 0,
                    "n_components": min(np.ceil(np.sqrt(num_features)), 6),
                    "random_state": 99
                }
        
            umap_trans = umap.UMAP(**umap_params).fit(X_ref_nn)
        
            X_ref_dim = umap_trans.embedding_
            X_new_dim = umap_trans.transform(X_new_nn)
        else:
            X_ref_dim = X_ref_nn
            X_new_dim = X_new_nn
        
        # Get k nearest neighbours
        nn = NearestNeighbors(n_neighbors=k,
                              algorithm='ball_tree').fit(X_ref_dim)
        _, nn_ix = nn.kneighbors(X_new_dim)

        # Get residuals for each of the neighbours
        residuals_nn = lambda z: residuals_ref[z]

        # Get bootstrapped errors
        q_err = np.apply_along_axis(lambda x:
                                    boot_err(x, reps=reps, alpha=alpha), 1,
                                    residuals_nn(nn_ix))

        result = pd.DataFrame(yhat_new, columns=["yhat"])
        result = result.assign(
            lower_err=yhat_new - q_err[:, 0],
            upper_err=yhat_new + q_err[:, 0],
            lower_pred=yhat_new - q_err[:, 1],
            upper_pred=yhat_new + q_err[:, 1])

    return result
