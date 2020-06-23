"""
Calculate estimated (or exact) Shapley values for features in a model
"""

import itertools
import random

import datetime as dt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from numba import njit
from plotnine import ggplot, aes, theme, labs, \
    geom_line, geom_point, geom_smooth, scale_colour_gradient2, \
    geom_bar, facet_wrap, coord_flip


def get_permutation_from_index(elements, perm_ix):
    """ Get a permutation corresponding to an index

    If all permutations of a list of elements are represented as a tree
    structure, we can identify a unique permutation from a given index position
    by traversing that tree.

    Parameters
    ----------
    elements : list
        List of elements that the permutation is derived from
    perm_ix : int
        Index position of permutation

    Returns
    -------
    perm: list
        Permutation corresponding to the index
    """
    # Check ix is an integer
    if not isinstance(perm_ix, int):
        raise Exception("Index is not an integer")

    # Check index exists in set of possible permutations
    if perm_ix < 0 or perm_ix > np.math.factorial(len(elements)):
        raise Exception("Index does not exist: {}".format(perm_ix))

    # List to hold result
    perm = []

    elements_copy = elements.copy()

    # Search the tree structure for the permutation that corresponds to the
    # index
    while elements_copy:
        k = len(elements_copy)
        leaves_in_branch = np.math.factorial(k - 1)
        branch = np.int(perm_ix / leaves_in_branch)
        perm.append(elements_copy.pop(branch))
        perm_ix -= branch * leaves_in_branch

    return perm


def random_sample_int_naive(upper_limit, num_samples):
    """ Generate a list of random integers without replacement using a naive
    algorithm

    This is inefficient as it stores already seen values in a set. However,
    unlike the built in functions the upper limit is not restricted by
    a maximum value.

    Parameters
    ----------
    upper_limit : int
        Upper limit of random integers to generate
    num_samples : int
        Number of random integers to return

    Returns
    -------
    sample: list
        List of random integers
    """
    sample = set()
    while len(sample) < num_samples:
        random_int = random.randrange(upper_limit)
        if random_int not in sample:
            sample.add(random_int)
    return list(sample)


def random_sample_int(upper_limit, num_samples):
    """ Build list of random integers (without replacement)

    Parameters
    ----------
    upper_limit : int
        Upper limit of random integers to generate
    num_samples : int
        Number of random integers to return

    Returns
    -------
    sample: list
        List of random integers
    """
    # Use random.sample function if the upper limit is <20!
    if upper_limit <= np.math.factorial(20):
        sample = random.sample(range(upper_limit), k=num_samples)
    else:
        # upper_limit is too large: Use naive method otherwise
        sample = random_sample_int_naive(upper_limit, num_samples)
    return sample


def random_sample_permutations(elements, num_samples):
    """ Generate random permutations without replacement

    Generates a list of random integers without replacement, and then matches
    them to permutations

    Parameters
    ----------
    elements : list
        List of elements
    num_samples : int
        Number of random permutations to return

    Returns
    -------
    random_ints : generator
        Generator
    """
    k = len(elements)

    # Check that sample size does not exceed count of possible permutations
    num_possible_perms = np.math.factorial(k)
    if num_samples > num_possible_perms:
        str_ex = "Requested number of samples ({}) exceeds possible " + \
            "permutations: {}"
        raise Exception(str_ex.format(num_samples, num_possible_perms))

    # When sample size == number of possible permutations use itertools to
    # generate all permutations
    if num_samples == np.math.factorial(k):
        return itertools.permutations(elements)

    # Get random integers and then create generator matching them to
    # permutations
    random_ints = random_sample_int(np.math.factorial(k), num_samples)
    return (get_permutation_from_index(elements, ix) for ix in random_ints)


@njit()
def percent_abs_diff(arr1, arr2):
    """
    Sum of absolute differences between two arrays as a % of the magnitude
    of the first array

    Parameters
    ----------
    arr1 : array, shape(datapoints, features)
        Array against which to measure the difference

    arr2 : array, shape(datapoints, features)
        Array

    Returns
    -------
    diff : float
    """
    diff = np.sum(np.abs(arr1 - arr2), axis=0) / np.sum(np.abs(arr1), axis=0)
    return diff


@njit()
def running_mean(arr, arr_new, num_obs):
    """ Takes an existing running mean array and recalculates using new data

    Parameters
    ----------
    arr : array, shape(,1)
        DESCRIPTION.
    arr_new : array, shape(,1)
        New data to add to x and calculate new mean values
    num_obs : int
        Number of observations used to calculate arr

    Returns
    -------
    result : array, shape(,1)
        New running mean
    """
    return (arr * num_obs + arr_new) / (num_obs + 1)


class ShapleyEstimator(BaseEstimator):
    """
    Estimate Shapley values
    """
    def __init__(self, model, tol=0.001, tol_counter_threshold=5,
                 max_iter=100, base_features=None, feature_groups=None,
                 default_values=None):
        """ Initialise the class

        Parameters
        ----------
        model : regressor
            Model to use in estimating the Shapley values

        tol : float, optional default=0.005
            Stopping tolerance for iterations (% difference in
            convergence metric since previous iteration)

        tol_counter_threshold : int, optional default=5
            Number of consecutive iterations required below the
            tolerance level to trigger early stopping

        max_iter : int, optional default=100
            Maximum number of iterations to run

        base_features : list, optional default=[]
            Features to 'fix' in the estimator. These features are
            input first into the model and are not included in the
            permutations

        feature_groups : dict, optional default=None
            Dict of feature groups {feature_group: list of features}
            If not None, this enableAs shapley values to be calculated
            for groups rather than their constituent features. This
            might be desirable, for example, if a feature group is not
            of interest, and computation time is an issue.

        default_values : dict, optional default=None {str: float}
            Dict of features to default value lookup {feature: value}.
            Default value to use when feature is 'excluded' from the
            model.

        Notes
        -----
        The convergence metric used is the sum of absolute differences
        in the Shapley values for each datapoint in X between iterations,
        divided by the sum of Shapley values for the previous iteration.
        """
        self.model = model
        self.tol = tol
        self.tol_counter_threshold = tol_counter_threshold
        self.max_iter = max_iter
        if base_features is None:
            self.base_features = []
        else:
            self.base_features = base_features
        self.feature_groups = feature_groups
        self.default_values = default_values

        # Initialize other parameters
        self._shapley_values = None
        self._feature_vars = []
        self._convergence_history = {}
        self._shapley_history = {}
        self._base_yhat = None
        #self._full_yhat = None
        self._already_seen = dict()

        self.X = None

    def fit(self, X, y=None):
        """Calculate Shapley values for X.

        Parameters
        ----------
        X : pandas DataFrame, shape(n_samples, n_features)
            Input array

        preprocessor: sklearn preprocessor, optional default=None
            Preprocessor to transform X before inputting to the model
        """
        print("Start time: {}".format(
            dt.datetime.now().strftime("%H:%M:%S.%f")))

        # Store X within the class
        self.X = X

        # Counter used for convergence early-stopping
        tol_counter = 0

        # Get features to iterate over (i.e. non-base features)
        shapley_features = [x for x in self.X.columns
                            if x not in self.base_features]

        # Set default values to use for the Shapley features
        if self.default_values is None:
            self.default_values = {x:0 for x in shapley_features}

        # Get feature(group) list if supplied
        if self.feature_groups:
            self._feature_vars = list(self.feature_groups.keys())
        else:
            self._feature_vars = shapley_features

        num_feature_vars = len(self._feature_vars)

        # Check that maximum iterations parameter is not larger than
        # the total amount of possible permutations
        possible_permutations = np.math.factorial(num_feature_vars)
        if possible_permutations < self.max_iter:
            print("Maximum iterations exceeds possible permutations.\n" +
                  "Using all permutations instead.")
            max_iter_to_use = possible_permutations
        else:
            max_iter_to_use = self.max_iter

        # Get the permutations to iterate over
        perms = random_sample_permutations(list(range(num_feature_vars)),
                                           max_iter_to_use)

        # Get number of all possible coalitions
        num_coalitions = 2**num_feature_vars

        # Set up arrays to hold Shapley values and convergence history
        self._shapley_values = np.zeros((X.shape[0], num_feature_vars))
        shapley_values_prev = self._shapley_values.copy()
        self._shapley_history = {}

        # Get copy of data and set non-base features to default values
        X_shapley = self.X.copy()

        for f in shapley_features:
            X_shapley[f] = self.default_values[f]

        self._base_yhat = self.model.predict(X_shapley)

        for i, current_perm in enumerate(perms):
            # Initialise X
            X_shapley = self.X.copy()

            for f in shapley_features:
                X_shapley[f] = self.default_values[f]

            current_yhat = self._base_yhat

            # List to hold each element as it is introduced (the elements are
            # sorted and converted into a tuple to see if this subset
            # combination has already been seen)
            combo = []

            # Add each feature in turn and get predicted value
            for j in current_perm:
                # Copy across X values for next feature (group) in permutation
                if self.feature_groups:
                    new_features = self.feature_groups[self._feature_vars[j]]
                else:
                    new_features = self._feature_vars[j]

                X_shapley[new_features] = self.X[new_features].copy()

                # Calculate marginal effect of adding this feature (group).
                # Check whether coalition has already been seen. If True then
                # use cached values
                combo.append(j)
                combo.sort()
                # Convert list to tuple (so that it can be a dict key)
                tuple_combo = tuple(combo)
                # Check if combination has already been seen
                if tuple_combo in self._already_seen:
                    yhat = self._already_seen[tuple_combo]
                else:
                    yhat = self.model.predict(X_shapley)
                    self._already_seen[tuple_combo] = yhat
                marginal_effect = yhat - current_yhat

                # Calculate running mean
                if i == 0:
                    self._shapley_values[:, j] = marginal_effect
                else:
                    self._shapley_values[:, j] = running_mean(
                        self._shapley_values[:, j], marginal_effect, i)

                current_yhat = yhat

            # Convergence : calculate % abs difference from last iteration
            conv = np.nan_to_num(percent_abs_diff(self._shapley_values,
                                                  shapley_values_prev))

            # Reset snapshot of current shapley values
            shapley_values_prev = self._shapley_values.copy()

            # Log current sum of absolute shapley values (for testing)
            if (i + 1) % max(10**np.int(np.log10(i + 1) - 1), 1) == 0:
                # Log convergence metric for this iteration
                self._convergence_history[i] = conv

            # Print convergence (intervals increase logarithmically)
            mean_conv = np.mean(conv)
            if (i + 1) % 10**np.int(np.log10(i + 1)) == 0:
                # Print current status
                print(("Iteration: {:6d}, Difference: {:f}, Time: {}, " + \
                       "Coalitions seen: {:1.2f}%").format(
                           i + 1, mean_conv,
                           dt.datetime.now().strftime("%H:%M:%S.%f"),
                           len(self._already_seen) * 100 /
                           (num_coalitions - 1)))

                # Log current shapley values (for testing)
                self._shapley_history[i] = self._shapley_values.copy()

            # Early stopping if convergence measure is under the threshold 'tol'
            # for 'tol_counter_threshold' iterations
            if mean_conv < self.tol:
                tol_counter += 1
            else:
                tol_counter = 0

            if tol_counter >= self.tol_counter_threshold:
                print("Early stopping at iteration {}".format(i + 1))
                self._convergence_history[i] = conv
                self._shapley_history[i] = self._shapley_values.copy()
                break

        print("Finished")
        print(("Iteration: {:6d}, Difference: {:f}, Time: {}, " + \
               "Coalitions seen: {:1.2f}%").format(
                   i + 1, np.mean(conv),
                   dt.datetime.now().strftime("%H:%M:%S.%f"),
                   len(self._already_seen) * 100 / (num_coalitions - 1)))

    def get_shapley_values(self):
        """Get the estimated Shapley values

        Creates a data frame of the Shapley values

        Returns
        -------
        shapley values : pandas DataFrame
            Data frame containing the estimated Shapley values
        """
        # Check Shapley values exist
        if self._shapley_values is None:
            raise Exception("No Shapley values are available")

        shapley_values_df = pd.DataFrame(
            self._shapley_values, columns=self._feature_vars,
            index=self.X.index)
        shapley_values_df["BASE"] = self._base_yhat
        shapley_values_df = shapley_values_df[["BASE"] +
                                              self._feature_vars]
        return shapley_values_df

    def get_shapley_history(self):
        """Get a data frame containing logged values of shapley history
        """
        # Check Shapley history exists
        if not self._shapley_history:
            raise Exception("No Shapley values are available")

        shapley_history_df = pd.concat(
            [pd.DataFrame(v, columns=self._feature_vars).assign(
                BASE=self._base_yhat,
                ITERATION=k) for k, v in self._shapley_history.items()])
        return shapley_history_df[["ITERATION", "BASE"] + self._feature_vars]

    def get_convergence_history(self):
        """Get a data frame containing the convergence metric history

        Returns
        -------
        conv_history : pandas DataFrame
            Data frame containing the convergence metric history
        """
        conv_history = pd.DataFrame(self._convergence_history,
                                    index=self._feature_vars).transpose()
        conv_history.index.rename("Iteration", inplace=True)
        return conv_history

    def plot_convergence_history(self, start_iter=5, figure_size=(10, 5)):
        """Plot of the convergence history

        Plots convergence metric for each feature in grey, overlaid with
        the mean.

        Parameters
        ----------
        start_iter : int, optional default=5
            Iteration to start plotting from (initial differences between
            interations will likely be large)

        Returns
        -------
        g : ggplot object
        """
        # Check convergence history exists
        if self._convergence_history is None:
            raise Exception("Convergence history is not available")

        conv_history = self.get_convergence_history()

        conv_history_melt = conv_history.reset_index().melt(
            id_vars=["Iteration"],
            var_name="variable",
            value_name="value")

        conv_history_mean = pd.DataFrame(
            conv_history.mean(axis=1)).reset_index()
        conv_history_mean.columns = ["Iteration", "value"]
        conv_history_mean = conv_history_mean.assign(
            variable="mean")

        g = (ggplot(
            conv_history_melt[conv_history_melt["Iteration"] >= start_iter],
            aes(x="Iteration", y="value", group="variable")) +
            geom_line(colour="grey", alpha=0.5) +
            geom_line(data=conv_history_mean[
                conv_history_melt["Iteration"] >= start_iter],
                mapping=aes(x="Iteration", y="value"), size=1) +
            labs(title="Convergence history",
                 x="Iteration",
                 y="Value") +
            theme(figure_size=figure_size)
            )

        return g

    def plot_partial_dependency(self, var1, var2=None,
                                facet_var=None, figure_size=(6, 4)):
        """Produce partial dependency plot

        Plots of features against the corresponding Shapley values,
        produced with plotnine

        Parameters
        ----------
        var1 : string
            Feature to plot along the x-axis (main feature of interest)

        var2 : string, optional default=None
            If provided, the data points' colour is linked to this feature

        figure_size : tuple, optional default=(6, 4)
            Figure size to use for plot (width, height)

        Returns
        -------
        g : ggplot object
        """
        # Check Shapley values exist
        if self._shapley_values is None:
            raise Exception("No Shapley values are available")

        # Build list of features
        if var2:
            if var1 != var2:
                var_list = [var1, var2]
            else:
                var_list = [var1]
        else:
            var_list = [var1]

        # Add facet wrap variable if provided
        if facet_var is not None:
            var_list += [facet_var]

        # Get original feature values and their Shapley values
        X_pdp = pd.concat([self.X[var_list],
                           self.get_shapley_values()[[var1]].rename(
                               columns=lambda x: x + "_shapley")],
                          axis=1)

        # Build plots
        if var2:
            # Calculate midpoint for colour gradient (otherwise default
            # is zero)
            midpoint = (max(X_pdp[var2]) -
                        min(X_pdp[var2])) / 2 + \
                min(X_pdp[var2])

            g = (ggplot(X_pdp,
                        aes(x=var1, y=var1 + "_shapley", colour=var2)) +
                 geom_point(alpha=0.5) +
                 geom_smooth(se=False, colour="grey") +
                 scale_colour_gradient2(low="blue", mid="yellow",
                                        high="red", midpoint=midpoint))
        else:
            g = (ggplot(X_pdp,
                        aes(x=var1, y=var1 + "_shapley")) +
                 geom_point(alpha=0.5) +
                 geom_smooth(se=False, colour="grey"))

        # Add labels and set figure size
        g += labs(y="Shapley value",
                  title="Partial dependency plot")
        g += theme(figure_size=figure_size)

        # Add facet wrap if required
        if facet_var is not None:
            g += facet_wrap("~" + facet_var)

        return g

    def plot_data_point(self, data_point_ix, use_base=True,
                        figure_size=(8, 6)):
        """ Plot Shapley values for an individual data point

        Parameters
        ----------
        data_point_ix :  int
        use_base : boolean, optional default=True

        Returns
        -------
        g : ggplot object
        """
        # Check Shapley values exist
        if self._shapley_values is None:
            raise Exception("No Shapley values are available")

        d = self.get_shapley_values().loc[[data_point_ix]]

        if not use_base:
            d = d.drop("BASE", axis=1)

        g = (ggplot(d.reset_index(drop=False).melt(
            id_vars="index"),
            aes(x="variable", y="value", fill="variable")) +
            geom_bar(stat="identity") +
            labs(title="Shapley values (Index: " + str(data_point_ix) + ")",
                 x="Feature",
                 y="Shapley value",
                 fill="Feature") +
            coord_flip())
        g += theme(figure_size=figure_size)

        return g
