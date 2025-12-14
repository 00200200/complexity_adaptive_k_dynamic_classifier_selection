import warnings

import numpy as np
from deslib.dcs import APosteriori
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted, check_X_y

from src.utils.utils import get_pycol_complexity_vector

warnings.filterwarnings("ignore")


class AdaptivePyColKNN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        pool_classifiers,
        k_comp=30,
        fallback_k=None,
        random_state=42,
        algorithm=APosteriori,
        meta_model=None,
        threshold=0.55,
        selected_feature_names=None,
        complexity_scaler=None,
        k_mapping=None,
    ):
        self.pool = pool_classifiers
        self.k_comp = k_comp
        self.fallback_k = fallback_k
        self.random_state = random_state
        self.algorithm = algorithm
        self.selected_feature_names = selected_feature_names
        self.meta_model = meta_model
        self.threshold = threshold
        self.complexity_scaler = complexity_scaler
        self.k_mapping = k_mapping

    def _prepare_dcs_models(self, unique_ks):
        unique_ks = set(unique_ks) | {self.fallback_k}
        self._dcs = {}
        for k in sorted(unique_ks):
            self._dcs[k] = self.algorithm(
                pool_classifiers=self.pool, k=k, random_state=self.random_state
            ).fit(self.X_dsel_, self.y_dsel_)

    def _get_valid_comp_vec(self, X_test_point):
        max_k = min(50, len(self.X_dsel_) - 1)
        for k_try in range(self.k_comp, max_k + 1, 10):
            try:
                nbrs = NearestNeighbors(n_neighbors=min(k_try, len(self.X_dsel_) - 1))
                nbrs.fit(self.X_dsel_)
                local_idx = nbrs.kneighbors(X_test_point.reshape(1, -1), return_distance=False)[0]
                comp_vec, _ = get_pycol_complexity_vector(
                    self.X_dsel_[local_idx], self.y_dsel_[local_idx], self.selected_feature_names
                )
                if comp_vec is not None and not np.any(np.isnan(comp_vec)):
                    return comp_vec
            except Exception:
                pass
        return None

    def fit(self, X_dsel, y_dsel):
        self.X_dsel_, self.y_dsel_ = check_X_y(X_dsel, y_dsel)
        self._complexity_scaler_ = self.complexity_scaler
        self._prepare_dcs_models({3, 5, 7, 9, 11, 13, 15})
        return self

    def predict(self, X_test):
        check_is_fitted(self)
        X_test_np = np.asarray(X_test)

        if self.meta_model is None:
            raise Exception("Meta model is not provided")

        k_to_indices = {}
        for i in range(X_test_np.shape[0]):
            comp_vec = self._get_valid_comp_vec(X_test_np[i])

            if comp_vec is None or np.any(np.isnan(comp_vec)):
                k_sel = self.fallback_k
            else:
                try:
                    comp_vec_array = np.asarray(comp_vec).reshape(1, -1)
                    
                    if self._complexity_scaler_ is not None:
                        comp_vec_array = self._complexity_scaler_.transform(comp_vec_array)
                    
                    proba = self.meta_model.predict_proba(comp_vec_array)
                    max_proba = proba.max(axis=1)[0]
                    if max_proba > self.threshold:
                        k_sel = int(self.meta_model.predict(comp_vec_array)[0])
                        if self.k_mapping is not None:
                            k_sel = int(self.k_mapping.get(k_sel, self.fallback_k))
                        if k_sel < 3 or k_sel > 15:
                            k_sel = self.fallback_k
                    else:
                        k_sel = self.fallback_k
                except Exception:
                    k_sel = self.fallback_k
            
            model_k = k_sel if k_sel in self._dcs else self.fallback_k
            k_to_indices.setdefault(model_k, []).append(i)

        preds = np.zeros(X_test_np.shape[0], dtype=int)
        for k_sel, indices in k_to_indices.items():
            model = self._dcs.get(k_sel, self._dcs[self.fallback_k])
            preds[indices] = model.predict(X_test_np[indices])
        
        return preds

