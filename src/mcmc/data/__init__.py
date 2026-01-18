from .io import Dataset as Dataset
from .io import load_dataset as load_dataset
from .registry import DatasetSpec as DatasetSpec
from .registry import list_datasets as list_datasets
from .registry import get_spec as get_spec

__all__ = ["Dataset", "load_dataset", "DatasetSpec", "list_datasets", "get_spec"]
