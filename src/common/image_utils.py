"""
Image Processing Utilities
"""

import os
from typing import List


def process_image_directory(directory_path: str,
                            extensions: List[str] = None) -> List[str]:
    """
    Process image directory and return all image paths

    Args:
        directory_path: Path to image directory
        extensions: List of valid image extensions (default: common image formats)

    Returns:
        List of image file paths
    """
    if not os.path.exists(directory_path):
        return []

    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    image_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths
