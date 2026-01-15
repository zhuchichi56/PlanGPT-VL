import argparse
import base64
import os
from openai import OpenAI


def encode_image(image_path: str) -> tuple[str, str]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image_type = os.path.splitext(image_path)[1].lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(image_type, "image/jpeg")

    return base64.b64encode(content).decode("utf-8"), mime_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Test proxy with text or VQA.")
    parser.add_argument("--base-url", default="http://localhost:8081/v1")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--question", default="What is in this image?")
    parser.add_argument("--image", default=None, help="Local image path for VQA.")
    args = parser.parse_args()

    client = OpenAI(
        api_key="dummy",
        base_url=args.base_url,
    )

    if args.image:
        base64_image, mime_type = encode_image(args.image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": args.question},
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": "Hello!"}]

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
