import json
import os
import argparse

def convert_format(input_file, output_file):
    """
    Convert from the original format:
    [
        {
            "image": "/path/to/image.jpeg",
            "question": "Question text?",
            "thinking": "",
            "summary": "Summary text"
        },
        ...
    ]
    
    To the target format:
    [
        {
            "messages": [
                {
                    "content": "<image>Question text?",
                    "role": "user"
                },
                {
                    "content": "Summary text",
                    "role": "assistant"
                }
            ],
            "images": [
                "new/path/to/image.jpeg"
            ]
        },
        ...
    ]
    """
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    output_data = []
    
    for item in input_data:
        # Extract the image path and use it directly as input
        image_path = item["image"]
        # Use the image path as is for the output
        new_image_path = image_path
        
        # Create the new format
        new_item = {
            "messages": [
                {
                    "content": f"<image>{item['question']}",
                    "role": "user"
                },
                {
                    "content": item["summary"],
                    "role": "assistant"
                }
            ],
            "images": [
                new_image_path
            ]
        }
        
        output_data.append(new_item)
    
    # Write the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete. Output saved to {output_file}")

