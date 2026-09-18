"""Portable aggregate analysis of task-based educational chat logs."""

from .analysis import analyze
from .models import AnalysisConfig, AggregateReport, Dataset
from .sources import load_source

__all__ = ["AggregateReport", "AnalysisConfig", "Dataset", "analyze", "load_source"]
__version__ = "0.1.0"
