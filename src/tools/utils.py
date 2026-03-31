"""
Shared Utilities (I/O, image, text)
"""

import json
import os
import re
from typing import Any, Dict, List


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


def parse_sections(text: str) -> Dict[str, str]:
    """
    Parse thinking and summary sections from VLM response

    Args:
        text: Response text containing <think> and <summary> tags

    Returns:
        Dictionary with 'thinking' and 'summary' keys
    """
    result = {"thinking": "", "summary": ""}

    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        result["thinking"] = think_match.group(1).strip()

    summary_match = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
    if summary_match:
        result["summary"] = summary_match.group(1).strip()

    return result


def parse_questions(text: str) -> List[Dict[str, str]]:
    """
    Parse questions from formatted text

    Format: 【问题类型】[维度] 问题内容

    Args:
        text: Text containing formatted questions

    Returns:
        List of dictionaries with 'type', 'dimension', and 'content' keys
    """
    question_matches = re.findall(
        r'【([^】]+)】\s*\[([^\]]+)\]\s*(.*?)$',
        text,
        re.MULTILINE
    )

    parsed_questions = []
    for match in question_matches:
        if len(match) == 3:
            question_type, dimension, content = match
            parsed_questions.append({
                "type": question_type,
                "dimension": dimension,
                "content": content.strip()
            })

    return parsed_questions


def extract_thinking_content(response: str) -> str:
    """
    Extract content from <thinking> tags

    Args:
        response: Response text

    Returns:
        Thinking content or empty string
    """
    thinking_pattern = re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL)
    thinking_match = thinking_pattern.search(response)
    if thinking_match:
        return thinking_match.group(1).strip()
    return ""


def combine_thinking_answer(thinking: str, answer: str) -> str:
    """
    Combine thinking and answer sections

    Args:
        thinking: Thinking content
        answer: Answer content

    Returns:
        Combined formatted string
    """
    return f"<thinking>\n{thinking}\n</thinking>\n{answer}"
