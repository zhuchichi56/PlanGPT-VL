"""
Text Processing Utilities
"""

import re
from typing import Dict, List


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
