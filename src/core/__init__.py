"""
Core Module for PlanGPT-VL

This module contains core configurations and prompt templates.
"""

from .prompts import PROMPTS, PromptManager
from .config import Config

__all__ = ['PROMPTS', 'PromptManager', 'Config']
