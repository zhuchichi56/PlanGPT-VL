import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
from matplotlib.font_manager import FontProperties
import os

def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: list, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Load data
data_path = "Your Data"
data = load_json(data_path)

# Create output directory
output_dir = "analysis_results"
os.makedirs(output_dir, exist_ok=True)

# Data analysis
total_questions = len(data)
print(f"Total questions: {total_questions}")

# 1. Analyze question types (basic vs difficult)
question_types = []
for item in data:
    if "基础" in item.get("type", ""):
        question_types.append("Basic")
    else:
        question_types.append("Difficult")

type_counter = Counter(question_types)
basic_count = type_counter.get("Basic", 0)
difficult_count = type_counter.get("Difficult", 0)
basic_percentage = basic_count / total_questions * 100
difficult_percentage = difficult_count / total_questions * 100

print(f"Basic questions: {basic_count} ({basic_percentage:.2f}%)")
print(f"Difficult questions: {difficult_count} ({difficult_percentage:.2f}%)")

# 2. Analyze dimension distribution
dimensions = [item.get("dimension", "Unknown") for item in data]
dimension_counter = Counter(dimensions)
dimension_labels = list(dimension_counter.keys())
dimension_values = list(dimension_counter.values())
dimension_percentages = [count / total_questions * 100 for count in dimension_values]

# 3. Analyze question and answer lengths
question_lengths = [len(item.get("question", "")) for item in data]
response_lengths = []

for item in data:
    response = item.get("response", "")
    # Remove thinking process, keep only summary part
    summary_match = re.search(r'<summary>(.*?)</summary>', response, re.DOTALL)
    if summary_match:
        summary = summary_match.group(1).strip()
        response_lengths.append(len(summary))
    else:
        response_lengths.append(len(response))

# Plot settings
plt.rcParams['font.sans-serif'] = ['SimHei']  # For displaying Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # For displaying minus sign
plt.style.use('seaborn-v0_8-whitegrid')
colors = plt.cm.viridis(np.linspace(0, 1, 10))

# 1. Question type pie chart
plt.figure(figsize=(10, 6))
plt.pie([basic_count, difficult_count], 
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

# 2. Dimension distribution horizontal bar chart
plt.figure(figsize=(12, 8))
sorted_indices = np.argsort(dimension_values)
sorted_dimensions = [dimension_labels[i] for i in sorted_indices]
sorted_values = [dimension_values[i] for i in sorted_indices]
sorted_percentages = [dimension_percentages[i] for i in sorted_indices]

bars = plt.barh(sorted_dimensions, sorted_values, color=colors[3], alpha=0.8)
plt.xlabel('Number of Questions', fontsize=12)
plt.ylabel('Dimension Type', fontsize=12)
plt.title('Question Dimension Distribution', fontsize=16, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add percentage labels
for i, (bar, percentage) in enumerate(zip(bars, sorted_percentages)):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{percentage:.1f}%', 
             va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{output_dir}/dimension_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Question length distribution
plt.figure(figsize=(10, 6))
sns.histplot(question_lengths, kde=True, bins=20, color=colors[2])
plt.axvline(np.mean(question_lengths), color='r', linestyle='--', label=f'Mean: {np.mean(question_lengths):.1f} chars')
plt.axvline(np.median(question_lengths), color='g', linestyle='-.', label=f'Median: {np.median(question_lengths):.1f} chars')
plt.xlabel('Question Length (characters)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Question Length Distribution', fontsize=16, pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/question_length_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Answer length distribution
plt.figure(figsize=(10, 6))
sns.histplot(response_lengths, kde=True, bins=20, color=colors[6])
plt.axvline(np.mean(response_lengths), color='r', linestyle='--', label=f'Mean: {np.mean(response_lengths):.1f} chars')
plt.axvline(np.median(response_lengths), color='g', linestyle='-.', label=f'Median: {np.median(response_lengths):.1f} chars')
plt.xlabel('Response Length (characters)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Response Length Distribution', fontsize=16, pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/response_length_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Relationship between question type and response length
plt.figure(figsize=(10, 6))
type_response_lengths = {
    "Basic": [response_lengths[i] for i in range(len(response_lengths)) if "基础" in question_types[i]],
    "Difficult": [response_lengths[i] for i in range(len(response_lengths)) if "基础" not in question_types[i]]
}


sns.boxplot(data=[type_response_lengths["Basic"], type_response_lengths["Difficult"]], 
            palette=[colors[1], colors[5]])
plt.xticks([0, 1], ['Basic', 'Difficult'])
plt.ylabel('Response Length (characters)', fontsize=12)
plt.title('Relationship Between Question Type and Response Length', fontsize=16, pad=20)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/question_type_vs_response_length.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Relationship between dimension and response length (heatmap)
dimension_response_length = {}
for i, dim in enumerate(dimensions):
    if dim not in dimension_response_length:
        dimension_response_length[dim] = []
    dimension_response_length[dim].append(response_lengths[i])

avg_lengths = {dim: np.mean(lengths) for dim, lengths in dimension_response_length.items()}
sorted_dims = sorted(avg_lengths.keys(), key=lambda x: avg_lengths[x], reverse=True)
sorted_avgs = [avg_lengths[dim] for dim in sorted_dims]

plt.figure(figsize=(12, 8))
bars = plt.barh(sorted_dims, sorted_avgs, color=plt.cm.viridis(np.linspace(0, 1, len(sorted_dims))))
plt.xlabel('Average Response Length (characters)', fontsize=12)
plt.ylabel('Dimension Type', fontsize=12)
plt.title('Average Response Length by Dimension', fontsize=16, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{output_dir}/dimension_vs_response_length.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Analysis results saved to {output_dir} directory")
