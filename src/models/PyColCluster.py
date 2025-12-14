import warnings

import numpy as np
from deslib.dcs import APosteriori
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y

from src.utils.utils import get_pycol_complexity_vector

warnings.filterwarnings("ignore")


class AdaptiveClusterPycol(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        pool_classifiers,
        fallback_k=7,
        cluster_min=2,
        cluster_max=15,
        random_state=42,
        algorithm=APosteriori,
        meta_model=None,
        threshold=0.55,
        selected_feature_names=None,
        complexity_scaler=None,
        k_mapping=None,
    ):
        self.pool = pool_classifiers
        self.fallback_k = fallback_k
        self.cluster_min = cluster_min
        self.cluster_max = cluster_max
        self.random_state = random_state
        self.algorithm = algorithm
        self.meta_model = meta_model
        self.selected_feature_names = selected_feature_names
        self.threshold = threshold
        self.complexity_scaler = complexity_scaler
        self.k_mapping = k_mapping

    def find_optimal_k_for_cluster(self, X_scaled):
        n_samples = X_scaled.shape[0]
        best_k, best_score = max(self.cluster_min, 2), -1.0
        for k in range(self.cluster_min, self.cluster_max + 1):
            if n_samples >= k:
                labels = KMeans(n_clusters=k, random_state=self.random_state).fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score, best_k = score, k
        return best_k

    def _predict_k_for_cluster(self, local_X, local_y):
        if self._meta_model_ is None:
            raise Exception("Meta model is not provided")

        comp_vec, _ = get_pycol_complexity_vector(local_X, local_y, self.selected_feature_names)

        if comp_vec is None or np.any(np.isnan(comp_vec)):
            return self.fallback_k

        try:
            comp_vec_array = np.asarray(comp_vec).reshape(1, -1)            
            if self._complexity_scaler_ is not None:
                comp_vec_array = self._complexity_scaler_.transform(comp_vec_array)
            
            proba = self._meta_model_.predict_proba(comp_vec_array)
            max_proba = proba.max(axis=1)[0]

            if max_proba > self.threshold:
                k_pred = int(self._meta_model_.predict(comp_vec_array)[0])
                if self.k_mapping is not None:
                    k_pred = int(self.k_mapping.get(k_pred, self.fallback_k))
                return k_pred
            else:
                return self.fallback_k
        except Exception:
            return self.fallback_k

    def _prepare_dcs_models(self, unique_ks):
        unique_ks = set(unique_ks) | {self.fallback_k}
        self._dcs = {}
        for k in sorted(unique_ks):
            self._dcs[k] = self.algorithm(
                pool_classifiers=self.pool, k=k, random_state=self.random_state
            ).fit(self.X_dsel_, self.y_dsel_)

    def fit(self, X_dsel, y_dsel):
        self.X_dsel_, self.y_dsel_ = check_X_y(X_dsel, y_dsel)
        self._meta_model_ = self.meta_model
        self._complexity_scaler_ = self.complexity_scaler

        self._scaler_X = StandardScaler().fit(self.X_dsel_)
        X_scaled = self._scaler_X.transform(self.X_dsel_)
        n_clusters = self.find_optimal_k_for_cluster(X_scaled)
        self._kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state).fit(X_scaled)
        labels = self._kmeans.labels_

        self._cluster_k = {}
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_size = np.sum(cluster_mask)
            n_classes = len(np.unique(self.y_dsel_[cluster_mask]))

            if cluster_size < 30 or n_classes < 2:
                self._cluster_k[i] = self.fallback_k
                continue

            local_X = self.X_dsel_[cluster_mask]
            local_y = self.y_dsel_[cluster_mask]
            self._cluster_k[i] = self._predict_k_for_cluster(local_X, local_y)

        self._prepare_dcs_models(set(self._cluster_k.values()))
        return self

    def predict(self, X_test):
        check_is_fitted(self)
        X_test_np = np.asarray(X_test)

        X_scaled = self._scaler_X.transform(X_test_np)
        cluster_ids = self._kmeans.predict(X_scaled)

        k_to_indices = {}
        for idx, cid in enumerate(cluster_ids):
            k_sel = self._cluster_k.get(cid, self.fallback_k)
            k_to_indices.setdefault(k_sel, []).append(idx)

        preds = np.zeros(X_test_np.shape[0], dtype=int)
        for k_sel, indices in k_to_indices.items():
            model = self._dcs.get(k_sel, self._dcs[self.fallback_k])
            preds[indices] = model.predict(X_test_np[indices])

        return preds

