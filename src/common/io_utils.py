"""
I/O Utilities for JSON and JSONLINES files
"""

import json
from typing import List, Dict, Any


def load_jsonlines(file_path: str) -> List[Dict[str, Any]]:
    """
    Load jsonlines file and convert to list of dictionaries

    Args:
        file_path: Path to jsonlines file

    Returns:
        List of dictionaries
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_json(file_path: str) -> Any:
    """
    Load JSON file

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2):
    """
    Save data to JSON file

    Args:
        data: Data to save
        file_path: Path to output JSON file
        indent: JSON indentation level
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def save_jsonlines(data: List[Dict[str, Any]], file_path: str):
    """
    Save data to jsonlines file

    Args:
        data: List of dictionaries to save
        file_path: Path to output jsonlines file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
