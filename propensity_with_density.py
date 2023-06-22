from abstract import AbstractOverlapEstimator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


class PropensityDensity(AbstractOverlapEstimator):
    classifier = None
    epsilon = 0.05
    density_estimator = None
    # P(T = 1) is by default 0.5
    p_T1 = 0.5

    """
        - Intialize PropensityDensity with classifier_type & epsilon
        - Set P(T = 1) if not 0.5 with set_p_T1()
        - Tune hyperparameters (the method tune_hyperparameters() already fits the data, no need to call fit() again)
        - Call predict or score
    """

    def __init__(self, classifier_type="lr", epsilon_in=0.05):
        """Initialize classifier

        Args:
            classifier_type (String): which type of classifier to initialize
                "lr" = LogisticRegression
                "rfc" = RandomForestClassifier
                "dtc" = DecisionTreeClassifier
            epsilon (float): what interval to deem as overlap [a, 1-a]
        """
        self.epsilon = epsilon_in
        if classifier_type == "lr":
            self.classifier = LogisticRegression()
        if classifier_type == "rfc":
            self.classifier = RandomForestClassifier()
        if classifier_type == "dtc":
            self.classifier = DecisionTreeClassifier()

        self.density_estimator = KernelDensity(bandwidth=0.2, kernel='gaussian')

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the overlap estimator to the data.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features) containing the input data.
            y (np.ndarray): Array of shape (n_samples, ) containing the group labels for each data point.
        """
        self.classifier.fit(X, y)
        self.density_estimator.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict whether each data point is in the overlap region.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features) containing the input data.

        Returns:
            np.ndarray: Array of shape (n_samples, ) with 0 or 1 for each data point.
                        0 means the data point is not in the overlap region,
                        and 1 means the data point is in the overlap region.
        """
        ps = self.classifier.predict_proba(X)[:, 1]

        # log(P(X=x))
        log_likelihood = self.density_estimator.score_samples(X)
        density = np.exp(log_likelihood)

        overlap_t1 = (ps * density) / self.p_T1
        overlap_t0 = (self.classifier.predict_proba(X)[:, 0] * density) / (1 - self.p_T1)

        overlap_ps = np.where((overlap_t1 > self.epsilon) & (overlap_t0 > self.epsilon), 1, 0)
        return overlap_ps

    def get_overlap_region(self) -> object:
        """Return the overlap region identified by the overlap estimator.
        #TODO Have to figure out what representation to use for the overlap region.

        Returns:
            object: The overlap region.
        """
        pass

    def score(
            self, X: np.ndarray, y: np.ndarray, score_type: str = "accuracy"
    ) -> float:
        """Compute the score of the model with X and y as test data.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features) containing the test data points.
            y (np.ndarray): Array of shape (n_samples, ) containing the group labels for each data point.
            score_type (str, optional): The type of score to compute. Defaults to "accuracy".

        Returns:
            float: The score of the model.
        """
        if score_type.lower() == "accuracy":
            prediction = self.predict(X)
            sum = 0.0
            for p, t in zip(prediction, y):
                if p == t:
                    sum += 1.0
            return sum / len(y)
        if score_type.lower() == "iou":
            prediction = self.predict(X)
            tp = fp = fn = 0.0
            for p, t in zip(prediction, y):
                if p == 1:
                    if t == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if t == 1:
                        fn += 1
            if (tp + fp + fn) == 0:
                return 0
            return tp / (tp + fp + fn)
        else:
            pass

    def get_params(self) -> dict:
        """Return the parameters of the overlap estimator.

        Returns:
            dict: The parameters of the overlap estimator.
        """
        params = self.classifier.get_params()
        params["epsilon"] = self.epsilon
        return params

    def set_params(self, **params) -> None:
        """Set the parameters of the overlap estimator.

        Args:
            **params: The parameters to set.
        """
        if "epsilon" in params:
            self.epsilon = params["epsilon"]
            del params["epsilon"]
        self.classifier.set_params(**params)

    def set_params_kde(self, **params) -> None:
        """Set the parameters of the kernel density estimator.

        Args:
            **params: The parameters to set.
        """
        self.density_estimator.set_params(**params)

    def set_p_T1(self, v):
        """Set P(T = 1)
        """
        self.p_T1 = v

    def get_overlap_samples(self, X: np.ndarray) -> np.ndarray:
        """Return only the data points in the overlap region.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features) containing the input data.

        Returns:
            np.ndarray: Array of shape (n_samples, n_features)
        """
        prediction = self.predict(X)
        overlap_ps = np.array([x for (x, p) in zip(X, prediction) if p == 1])

        return overlap_ps

    def tune_hyperparameters(self, classifier, X, y):
        """Tune hyperparameters of the density estimator and classifier

        Args:
            classifier (String): indicates which classifier is used
                "lr" = LogisticRegression
                "rfc" = RandomForestClassifier
                "dtc" = DecisionTreeClassifier
            X (np.ndarray): Array of shape (n_samples, n_features) containing the input data.
            y (np.ndarray): Array of shape (n_samples, ) containing the group labels for each data point.
        """

        # Optimize kde parameters
        param_grid = {
            'bandwidth': np.linspace(0.2, 0.8, 15)
            # 'bandwidth': np.linspace(0.1, 1, 100)
        }

        # Grid search with cross-validation
        grid_search = GridSearchCV(estimator=KernelDensity(), param_grid=param_grid)
        grid_search.fit(X, y)
        self.set_params_kde(**grid_search.best_params_)

        # Optimize classifier parameters
        cccv_classifier = None
        if classifier == "lr":
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10]
            }
            cccv_classifier = LogisticRegression()
        elif classifier == "dtc":
            param_grid = {
                'max_depth': [2, 3, 5, 10, 20],
                'criterion': ["gini", "entropy"]
            }
            cccv_classifier = DecisionTreeClassifier()

        elif classifier == "rfc":
            param_grid = {
                'n_estimators': [64, 128, 256],
                'max_depth': [2, 4, 6, 8]
            }
            cccv_classifier = RandomForestClassifier()

        grid_search = GridSearchCV(estimator=cccv_classifier, param_grid=param_grid, scoring='neg_log_loss')
        grid_search.fit(X, y)

        self.classifier = grid_search.best_estimator_
        self.density_estimator.fit(X, y)

    def propensity_plot(self, x1, x2):
        """Plot the propensity scores of each class.

        Args:
            x1 (np.ndarray): Array of shape (n_samples, n_features) containing the input data of class 1.
            x2 (np.ndarray): Array of shape (n_samples, n_features) containing the input data of class 2.
            x2 (np.ndarray): Array of shape (n_samples, n_features) containing the input data of class 2.

        """
        dim = np.shape(x1)[1]
        plt.figure(2)
        plt.hist(self.classifier.predict_proba(x1.reshape(-1, dim))[:, 1], bins=30, alpha=0.5, label='Treatment group')
        plt.hist(self.classifier.predict_proba(x2.reshape(-1, dim))[:, 1], bins=30, alpha=0.5, label='Control group')
        plt.legend()
        plt.xlabel("Probability of being in treatment group")
        plt.ylabel("Samples")
        plt.title("Propensity scoring of 500 samples")
        plt.show()