"""
Configuration Module for PlanGPT-VL

Centralized configuration management for the entire system.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InferenceConfig:
    """Configuration for inference servers"""
    server_url: str = "http://localhost:8000"
    server_urls: Optional[List[str]] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 200
    timeout: float = 600.0


@dataclass
class PathConfig:
    """Configuration for file paths"""
    output_dir: str = "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/tmp2"
    checkpoint_enabled: bool = True


@dataclass
class FilterConfig:
    """Configuration for image filtering"""
    filter_percent: float = 5.0  # Percentage of highest resolution images to filter
    min_resolution: Optional[int] = None
    max_resolution: Optional[int] = None


@dataclass
class Config:
    """Main configuration class"""
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create Config from dictionary"""
        return cls(
            inference=InferenceConfig(**config_dict.get('inference', {})),
            paths=PathConfig(**config_dict.get('paths', {})),
            filter=FilterConfig(**config_dict.get('filter', {}))
        )
