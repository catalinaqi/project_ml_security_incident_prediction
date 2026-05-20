from __future__ import annotations
from typing import Type, Dict, Callable, Any, Optional
import importlib

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor,GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from src.crispdm.common.logging_adapter_common import get_logger
from src.crispdm.configuration.enum_registry_config import ProblemType

log = get_logger(__name__)


class ModelRegistry:

    _CLUSTERING_MODELS: Dict[str, Type] = {
        "kmeans": KMeans,
        "dbscan": DBSCAN,
        "hierarchical": AgglomerativeClustering,
        "agglomerative": AgglomerativeClustering,
    }

    _CLASSIFICATION_MODELS: Dict[str, Type] = {
        "random_forest": RandomForestClassifier,
        "rf": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "xgb": XGBClassifier,
        "logistic": LogisticRegression,
        "logistic_regression": LogisticRegression,
        "svm": SVC,
        "svc": SVC,
        "gradient_boosting": GradientBoostingClassifier,
        "gb": GradientBoostingClassifier,
    }

    _REGRESSION_MODELS: Dict[str, Type] = {
        "linear": LinearRegression,
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "random_forest": RandomForestRegressor,
        "rf": RandomForestRegressor,
        "xgboost": XGBRegressor,
        "xgb": XGBRegressor,
        "svr": SVR,
        "gradient_boosting": GradientBoostingRegressor,
    }

    _TIMESERIES_MODELS: Dict[str, Type] = {
        # Placeholder - add ARIMA, Prophet, LSTM when implemented
    }

    @classmethod
    def get_model_class(
            cls,
            problem_type: str,
            model_name: str
    ) -> Type:

        log.debug(f"Getting model class: problem_type={problem_type}, model_name={model_name}")

        # Normalize inputs
        problem_type = problem_type.lower()
        model_name = model_name.lower()

        # Select registry based on problem type
        if problem_type in ["clustering", "cluster"]:
            registry = cls._CLUSTERING_MODELS
        elif problem_type in ["classification", "classify"]:
            registry = cls._CLASSIFICATION_MODELS
        elif problem_type in ["regression", "regress"]:
            registry = cls._REGRESSION_MODELS
        elif problem_type in ["timeseries", "time_series", "ts"]:
            registry = cls._TIMESERIES_MODELS
        else:
            log.error(f"Unknown problem type: {problem_type}")
            raise ValueError(f"Unknown problem type: {problem_type}")

        # Get model class
        if model_name not in registry:
            available = list(registry.keys())
            log.error(f"Model '{model_name}' not found for {problem_type}. Available: {available}")
            raise KeyError(f"Model '{model_name}' not found. Available: {available}")

        model_class = registry[model_name]
        log.info(f"Retrieved model class: {model_class.__name__}")
        return model_class

    @classmethod
    def get_available_models(cls, problem_type: str) -> list[str]:

        problem_type = problem_type.lower()

        if problem_type in ["clustering", "cluster"]:
            registry = cls._CLUSTERING_MODELS
        elif problem_type in ["classification", "classify"]:
            registry = cls._CLASSIFICATION_MODELS
        elif problem_type in ["regression", "regress"]:
            registry = cls._REGRESSION_MODELS
        elif problem_type in ["timeseries", "time_series", "ts"]:
            registry = cls._TIMESERIES_MODELS
        else:
            log.warning(f"Unknown problem type: {problem_type}")
            return []

        available = list(registry.keys())
        log.debug(f"Available models for {problem_type}: {available}")
        return available

    @classmethod
    def register_model(
            cls,
            problem_type: str,
            model_name: str,
            model_class: Type
    ) -> None:

        log.info(f"Registering custom model: {model_name} for {problem_type}")

        problem_type = problem_type.lower()
        model_name = model_name.lower()

        if problem_type in ["clustering", "cluster"]:
            cls._CLUSTERING_MODELS[model_name] = model_class
        elif problem_type in ["classification", "classify"]:
            cls._CLASSIFICATION_MODELS[model_name] = model_class
        elif problem_type in ["regression", "regress"]:
            cls._REGRESSION_MODELS[model_name] = model_class
        elif problem_type in ["timeseries", "time_series", "ts"]:
            cls._TIMESERIES_MODELS[model_name] = model_class
        else:
            log.error(f"Cannot register for unknown problem type: {problem_type}")
            raise ValueError(f"Unknown problem type: {problem_type}")

        log.info(f"Model registered successfully: {model_name}")


class TechniqueRegistry:

    @staticmethod
    def get_technique_function(
            technique_name: str,
            module_path: Optional[str] = None
    ) -> Callable:

        log.debug(f"Getting technique function: {technique_name}")

        # Convention: technique_name -> run_{technique_name}
        function_name = f"run_{technique_name}"

        if module_path is None:
            log.error(f"Module path required to load technique: {technique_name}")
            raise ValueError(f"Module path required for technique: {technique_name}")

        try:
            module = importlib.import_module(module_path)

            if not hasattr(module, function_name):
                log.error(f"Function '{function_name}' not found in {module_path}")
                raise AttributeError(f"Function '{function_name}' not found in {module_path}")

            func = getattr(module, function_name)
            log.info(f"Retrieved technique function: {function_name} from {module_path}")
            return func

        except ImportError as e:
            log.error(f"Failed to import module {module_path}: {e}")
            raise

    @staticmethod
    def execute_technique(
            technique_name: str,
            module_path: str,
            **kwargs: Any
    ) -> Any:

        log.info(f"Executing technique: {technique_name}")

        func = TechniqueRegistry.get_technique_function(technique_name, module_path)

        try:
            result = func(**kwargs)
            log.info(f"Technique executed successfully: {technique_name}")
            return result
        except Exception as e:
            log.error(f"Technique execution failed: {technique_name} - {e}")
            raise


def get_model_class(problem_type: str, model_name: str) -> Type:
    return ModelRegistry.get_model_class(problem_type, model_name)


def get_available_models(problem_type: str) -> list[str]:
    return ModelRegistry.get_available_models(problem_type)


def register_custom_model(problem_type: str, model_name: str, model_class: Type) -> None:
    ModelRegistry.register_model(problem_type, model_name, model_class)