"""
Data Analyzer and Post-processor

Analyzes and visualizes question-answer dataset statistics.
"""

import os
import re
from collections import Counter
from typing import Dict
import numpy as np

# Optional: Import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns  # noqa: F401
    from matplotlib.font_manager import FontProperties  # noqa: F401
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

from tools.utils import load_json, save_json  # noqa: F401


class DataAnalyzer:
    """Analyzer for question-answer datasets"""

    def __init__(self, data_path: str):
        """
        Initialize analyzer

        Args:
            data_path: Path to data JSON file
        """
        self.data = load_json(data_path)
        self.total_questions = len(self.data)

    def analyze_types(self) -> Dict:
        """Analyze question types (basic vs difficult)"""
        question_types = []
        for item in self.data:
            if "基础" in item.get("type", ""):
                question_types.append("Basic")
            else:
                question_types.append("Difficult")

        type_counter = Counter(question_types)
        return {
            "basic_count": type_counter.get("Basic", 0),
            "difficult_count": type_counter.get("Difficult", 0),
            "basic_percentage": type_counter.get("Basic", 0) / self.total_questions * 100,
            "difficult_percentage": type_counter.get("Difficult", 0) / self.total_questions * 100
        }

    def analyze_dimensions(self) -> Dict:
        """Analyze dimension distribution"""
        dimensions = [item.get("dimension", "Unknown") for item in self.data]
        dimension_counter = Counter(dimensions)

        return {
            "labels": list(dimension_counter.keys()),
            "values": list(dimension_counter.values()),
            "percentages": [count / self.total_questions * 100 for count in dimension_counter.values()]
        }

    def analyze_lengths(self) -> Dict:
        """Analyze question and response lengths"""
        question_lengths = [len(item.get("question", "")) for item in self.data]
        response_lengths = []

        for item in self.data:
            response = item.get("response", "")
            # Extract only summary part if exists
            summary_match = re.search(r'<summary>(.*?)</summary>', response, re.DOTALL)
            if summary_match:
                summary = summary_match.group(1).strip()
                response_lengths.append(len(summary))
            else:
                response_lengths.append(len(response))

        return {
            "question_lengths": question_lengths,
            "response_lengths": response_lengths,
            "question_avg": np.mean(question_lengths),
            "question_median": np.median(question_lengths),
            "response_avg": np.mean(response_lengths),
            "response_median": np.median(response_lengths)
        }

    def generate_report(self, output_path: str = "analysis_report.txt"):
        """Generate text analysis report"""
        type_stats = self.analyze_types()
        dimension_stats = self.analyze_dimensions()
        length_stats = self.analyze_lengths()

        report = []
        report.append(f"=" * 60)
        report.append("Dataset Analysis Report")
        report.append(f"=" * 60)
        report.append(f"Total questions: {self.total_questions}")
        report.append("")

        report.append("Question Types:")
        report.append(f"  Basic: {type_stats['basic_count']} ({type_stats['basic_percentage']:.2f}%)")
        report.append(f"  Difficult: {type_stats['difficult_count']} ({type_stats['difficult_percentage']:.2f}%)")
        report.append("")

        report.append("Dimensions:")
        for label, value, pct in zip(dimension_stats['labels'],
                                     dimension_stats['values'],
                                     dimension_stats['percentages']):
            report.append(f"  {label}: {value} ({pct:.2f}%)")
        report.append("")

        report.append("Lengths:")
        report.append(f"  Question - Avg: {length_stats['question_avg']:.1f}, Median: {length_stats['question_median']:.1f}")
        report.append(f"  Response - Avg: {length_stats['response_avg']:.1f}, Median: {length_stats['response_median']:.1f}")

        report_text = "\n".join(report)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        return report_text

    def plot_visualizations(self, output_dir: str = "analysis_results"):
        """Generate visualization plots (requires matplotlib)"""
        if not HAS_VIZ:
            print("Visualization libraries not available. Skipping plots.")
            return

        os.makedirs(output_dir, exist_ok=True)

        type_stats = self.analyze_types()
        dimension_stats = self.analyze_dimensions()
        length_stats = self.analyze_lengths()

        # Configure matplotlib
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = plt.cm.viridis(np.linspace(0, 1, 10))

        # 1. Question type pie chart
        plt.figure(figsize=(10, 6))
        plt.pie([type_stats['basic_count'], type_stats['difficult_count']],
                labels=['Basic', 'Difficult'],
                autopct='%1.1f%%',
                startangle=90,
                colors=[colors[1], colors[5]],
                wedgeprops={'edgecolor': 'w', 'linewidth': 1.5},
                textprops={'fontsize': 14})
        plt.title('Question Type Distribution', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/question_type_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Dimension distribution
        plt.figure(figsize=(12, 8))
        sorted_indices = np.argsort(dimension_stats['values'])
        sorted_dimensions = [dimension_stats['labels'][i] for i in sorted_indices]
        sorted_values = [dimension_stats['values'][i] for i in sorted_indices]
        sorted_percentages = [dimension_stats['percentages'][i] for i in sorted_indices]

        bars = plt.barh(sorted_dimensions, sorted_values, color=colors[3], alpha=0.8)
        plt.xlabel('Number of Questions', fontsize=12)
        plt.ylabel('Dimension Type', fontsize=12)
        plt.title('Question Dimension Distribution', fontsize=16, pad=20)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        for bar, percentage in zip(bars, sorted_percentages):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{percentage:.1f}%', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/dimension_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualizations saved to {output_dir}/")


def analyze_dataset(data_path: str,
                   output_dir: str = "analysis_results",
                   generate_plots: bool = True) -> DataAnalyzer:
    """
    Convenience function to analyze dataset

    Args:
        data_path: Path to data JSON
        output_dir: Output directory for results
        generate_plots: Whether to generate plots

    Returns:
        DataAnalyzer instance
    """
    analyzer = DataAnalyzer(data_path)
    analyzer.generate_report(os.path.join(output_dir, "analysis_report.txt"))

    if generate_plots:
        analyzer.plot_visualizations(output_dir)

    return analyzer
