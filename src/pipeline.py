"""
Data Synthesis Pipeline

Consolidated generation pipeline for questions, responses, CPT, and RLAIF-V.
"""

from dataclasses import dataclass, field
import os
import re
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from tqdm import tqdm

from core.prompts import PROMPTS
from tools.utils import (
    process_image_directory,
    load_json,
    save_json,
    parse_questions,
    extract_thinking_content,
    combine_thinking_answer,
)
from tools.inference_utils import parallel_image_inference, InferenceParams
from tools.filtering import filter_planning_maps


class QuestionGenerator:
    """Generator for urban planning questions from images"""

    def __init__(self, prompt_key: str = "context_question_generation_od"):
        """
        Initialize question generator

        Args:
            prompt_key: Key for question generation prompt
        """
        self.prompt_key = prompt_key

    def generate(self, image_paths: List[str],
                inference_params: Optional[InferenceParams] = None) -> List[Dict]:
        """
        Generate questions from images

        Args:
            image_paths: List of image paths
            inference_params: Inference configuration

        Returns:
            List of question dictionaries with structure:
                {
                    "image": image_path,
                    "question": question_content,
                    "type": question_type,
                    "dimension": question_dimension
                }
        """
        logger.info(f"Generating questions for {len(image_paths)} images")

        # Run inference
        template = PROMPTS.get(self.prompt_key, self.prompt_key)
        prompts = [template] * len(image_paths)
        results = parallel_image_inference(
            prompts,
            image_paths,
            inference_params=inference_params
        )

        # Parse questions
        all_questions = []
        for image_path, result in zip(image_paths, results):
            if not result:
                logger.warning(f"No result for {image_path}")
                continue

            parsed_questions = parse_questions(result)

            for question in parsed_questions:
                question_data = {
                    "image": image_path,
                    "question": question["content"],
                    "type": question["type"],
                    "dimension": question["dimension"]
                }
                all_questions.append(question_data)

        logger.info(f"Generated {len(all_questions)} questions")
        return all_questions


def generate_questions(image_paths: List[str],
                      inference_params: Optional[InferenceParams] = None) -> List[Dict]:
    """
    Convenience function to generate questions

    Args:
        image_paths: List of image paths
        inference_params: Inference configuration

    Returns:
        List of question dictionaries
    """
    generator = QuestionGenerator()
    return generator.generate(image_paths, inference_params=inference_params)


class ResponseGenerator:
    """Generator for urban planning question responses"""

    def __init__(self, inference_params: Optional[InferenceParams] = None):
        self.inference_params = inference_params

    def generate(self,
                question_results: List[Dict],
                mode: str = "direct_cpt",
                question_key: str = "question",
                caption_key: str = "caption",
                inference_params: Optional[InferenceParams] = None) -> List[str]:
        """
        Generate responses for questions

        Args:
            question_results: List of dictionaries containing 'image' and question field
            mode: Generation mode ('direct', 'with_caption', 'direct_cpt', 'with_caption_cpt')
            question_key: Key for question field in input data
            caption_key: Key for caption field (if using caption mode)
            inference_params: Inference configuration

        Returns:
            List of generated responses
        """
        # Select prompt template based on mode
        template_map = {
            "direct": "answer_direct",
            "with_caption": "answer_with_caption",
            "direct_cpt": "answer_direct_cpt",
            "with_caption_cpt": "answer_with_caption_cpt",
            "cpt": "answer_direct_cpt"  # Alias
        }

        template_key = template_map.get(mode, "answer_direct_cpt")
        logger.info(f"Using prompt template: {template_key}")

        # Prepare image paths and parameters
        image_paths = [item["image"] for item in question_results]

        # Prepare parameters for prompt formatting
        params = []
        for item in question_results:
            param = {"question": item[question_key]}
            if "caption" in template_key and caption_key in item:
                param["caption"] = item[caption_key]
            params.append(param)

        # Run inference
        inference_settings = inference_params or self.inference_params
        template = PROMPTS.get(template_key, template_key)
        prompts = [template.format(**p) for p in params]
        results = parallel_image_inference(
            prompts,
            image_paths,
            inference_params=inference_settings
        )

        return results


def generate_responses(question_results: List[Dict],
                      mode: str = "direct_cpt",
                      inference_params: Optional[InferenceParams] = None) -> List[str]:
    """
    Convenience function to generate responses

    Args:
        question_results: List of question dictionaries
        mode: Generation mode
        inference_params: Inference configuration

    Returns:
        List of responses
    """
    generator = ResponseGenerator(inference_params=inference_params)
    return generator.generate(
        question_results,
        mode,
        inference_params=inference_params
    )


class CPTGenerator:
    """Generator for critical point thinking"""

    def __init__(self, version: str = "critical_version"):
        """
        Initialize CPT generator

        Args:
            version: CPT version to use
        """
        self.version = version

    def generate(self,
                data: List[Dict],
                question_key: str = "question",
                answer_key: str = "answer",
                inference_params: Optional[InferenceParams] = None) -> List[Dict]:
        """
        Generate critical thinking points

        Args:
            data: List of dictionaries with 'image', question and answer
            question_key: Key for question field
            answer_key: Key for answer field
            inference_params: Inference configuration

        Returns:
            Updated data with CPT responses
        """
        logger.info(f"Generating CPT for {len(data)} items")

        # Prepare data
        image_paths = [item["image"] for item in data]
        params = [
            {"question": item[question_key], "answer": item[answer_key]}
            for item in data
        ]

        # Run inference
        template = PROMPTS.get(self.version, self.version)
        prompts = [template.format(**p) for p in params]
        results = parallel_image_inference(
            prompts,
            image_paths,
            inference_params=inference_params
        )

        # Process results
        for result, item in zip(results, data):
            if not result:
                logger.warning("No CPT result for item")
                continue

            thinking = extract_thinking_content(result)
            item[f"{self.version}_response"] = combine_thinking_answer(thinking, item[answer_key])

        return data


def generate_cpt(data: List[Dict],
                version: str = "critical_version",
                question_key: str = "question",
                answer_key: str = "answer",
                inference_params: Optional[InferenceParams] = None) -> List[Dict]:
    """
    Convenience function to generate CPT

    Args:
        data: List of question-answer dictionaries
        version: CPT version
        question_key: Key for question field
        answer_key: Key for answer field
        inference_params: Inference configuration

    Returns:
        Updated data with CPT
    """
    generator = CPTGenerator(version)
    return generator.generate(
        data,
        question_key=question_key,
        answer_key=answer_key,
        inference_params=inference_params
    )


class CaptionRefiner:
    """Refines image captions iteratively"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 inference_params: Optional[InferenceParams] = None):
        """
        Initialize caption refiner

        Args:
            api_key: API key for evaluator (if using external evaluation)
            api_base: API base URL for evaluator
        """
        self.api_key = api_key
        self.api_base = api_base
        self.response_gen = ResponseGenerator(inference_params=inference_params)

    def parse_key_points(self, caption: str) -> List[str]:
        """
        Parse numbered key points from caption

        Args:
            caption: Caption text with [1], [2], etc.

        Returns:
            List of key points
        """
        key_points = []
        pattern = r'\\[(\\d+)\\](.*?)(?=\\[\\d+\\]|$)'
        matches = re.findall(pattern, caption, re.DOTALL)

        for _, content in matches:
            key_point = content.strip()
            if key_point:
                key_points.append(key_point)

        # Fallback: split by paragraphs
        if not key_points:
            paragraphs = [p.strip() for p in caption.split('\\n\\n') if p.strip()]
            if paragraphs:
                key_points = paragraphs

        return key_points

    def generate_caption(self, item: Dict) -> str:
        """
        Generate initial caption for an item

        Args:
            item: Dictionary with 'image' and optionally 'question'

        Returns:
            Generated caption
        """
        temp_item = {
            "image": item["image"],
            "question": PROMPTS["caption_generation"]
        }

        results = self.response_gen.generate(
            [temp_item],
            mode="direct",
            question_key="question"
        )

        return results[0] if results else ""

    def improve_key_point(self, item: Dict, key_point: str, index: int) -> str:
        """
        Improve a single key point

        Args:
            item: Item dictionary with 'image' and 'question'
            key_point: Current key point text
            index: Index of the key point

        Returns:
            Improved key point or empty string if no improvement needed
        """
        prompt = PROMPTS["caption_improvement"].format(
            question=item.get("question", ""),
            index=index,
            key_point=key_point
        )

        temp_item = {
            "image": item["image"],
            "question": prompt
        }

        results = self.response_gen.generate(
            [temp_item],
            mode="direct",
            question_key="question"
        )

        result = results[0] if results else ""

        if "无需改进" in result:
            return ""

        # Extract improved version
        improved_point = ""
        patterns = [
            r'改进版本：\\s*(.*?)$',
            r'改进的描述点：\\s*(.*?)$',
            r'建议修改为：\\s*(.*?)$'
        ]

        for pattern in patterns:
            matches = re.search(pattern, result, re.DOTALL)
            if matches:
                improved_point = matches.group(1).strip()
                break

        # Fallback: use lines after keywords
        if not improved_point:
            lines = result.strip().split('\\n')
            for i, line in enumerate(lines):
                if "改进" in line or "建议" in line:
                    if i + 1 < len(lines):
                        improved_point = lines[i + 1].strip()
                        break

        return improved_point

    def improve_caption(self, item: Dict, current_caption: str) -> str:
        """
        Improve entire caption by refining each key point

        Args:
            item: Item dictionary
            current_caption: Current caption text

        Returns:
            Improved caption
        """
        key_points = self.parse_key_points(current_caption)
        improved_points = []

        for i, point in enumerate(key_points):
            improved = self.improve_key_point(item, point, i + 1)
            if improved:
                improved_points.append(f"[{i + 1}] {improved}")
            else:
                improved_points.append(f"[{i + 1}] {point}")

        return "\\n\\n".join(improved_points)

    def iterate_caption(self,
                       item: Dict,
                       max_iterations: int = 3) -> Tuple[str, List[str]]:
        """
        Iteratively improve caption

        Args:
            item: Item dictionary with 'image' and optionally 'question'
            max_iterations: Maximum improvement iterations

        Returns:
            Tuple of (best_caption, all_captions)
        """
        current_caption = self.generate_caption(item)
        all_captions = [current_caption]

        for i in range(max_iterations - 1):
            print(f"Iteration {i+1}/{max_iterations-1} for image {os.path.basename(item['image'])}")
            improved_caption = self.improve_caption(item, current_caption)
            all_captions.append(improved_caption)
            current_caption = improved_caption

        # Return last caption as best (in simple version without scoring)
        return current_caption, all_captions

    def process_dataset(self,
                       data: List[Dict],
                       max_iterations: int = 3,
                       max_workers: int = 8) -> List[Dict]:
        """
        Process entire dataset with caption refinement

        Args:
            data: List of item dictionaries
            max_iterations: Maximum iterations per caption
            max_workers: Maximum parallel workers

        Returns:
            Updated data with refined captions
        """
        def process_item(item):
            try:
                best_caption, all_captions = self.iterate_caption(item, max_iterations)

                result = item.copy()
                result["caption"] = best_caption
                result["all_captions"] = all_captions

                print(f"Processed {os.path.basename(item['image'])}")
                return result

            except Exception as e:
                print(f"Error processing item: {str(e)}")
                return None

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [executor.submit(process_item, item) for item in data]

            for future in tqdm(tasks, desc="Processing images"):
                result = future.result()
                if result:
                    results.append(result)

        print(f"Processed {len(results)} items")
        return results


def refine_captions(data_path: str,
                   output_path: str,
                   max_iterations: int = 3,
                   max_workers: int = 8) -> List[Dict]:
    """
    Convenience function to refine captions

    Args:
        data_path: Path to input data JSON
        output_path: Path to output JSON
        max_iterations: Maximum iterations per caption
        max_workers: Maximum parallel workers

    Returns:
        List of results with refined captions
    """
    data = load_json(data_path)
    refiner = CaptionRefiner()
    results = refiner.process_dataset(data, max_iterations, max_workers)
    save_json(results, output_path)
    print(f"Results saved to {output_path}")
    return results


@dataclass
class PipelineConfig:
    """Configuration for data synthesis pipeline"""
    output_dir: str
    inference: InferenceParams = field(default_factory=InferenceParams)
    filter_images: bool = False
    keep_unfiltered: bool = False
    response_mode: str = "direct_cpt"
    run_cpt: bool = True
    run_rlaifv: bool = False
    caption_max_iterations: int = 3
    caption_max_workers: int = 8
    checkpoint_dir: Optional[str] = None


class DataSynthesisPipeline:
    """End-to-end data synthesis pipeline"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _output_path(self, filename: str) -> str:
        return os.path.join(self.config.output_dir, filename)

    def _checkpoint_path(self, step_name: str) -> Optional[str]:
        if not self.config.checkpoint_dir:
            return None
        path = os.path.join(self.config.checkpoint_dir, step_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _maybe_filter(self, image_paths: List[str]) -> List[str]:
        if not self.config.filter_images:
            return image_paths

        logger.info("Running planning map filter")
        filter_results = filter_planning_maps(
            image_paths,
            output_dir=self._checkpoint_path("filter"),
            inference_params=self.config.inference
        )
        save_json(filter_results, self._output_path("filter_results.json"))

        if self.config.keep_unfiltered:
            return image_paths

        filtered_paths = [
            item["image"]
            for item in filter_results
            if item.get("is_planning_map") == 1
        ]
        logger.info(f"Filtered down to {len(filtered_paths)} images")
        return filtered_paths

    def _unique_images(self, items: List[Dict]) -> List[str]:
        seen = set()
        unique = []
        for item in items:
            image = item.get("image")
            if image and image not in seen:
                seen.add(image)
                unique.append(image)
        return unique

    def _maybe_refine_captions(self, questions: List[Dict]) -> List[Dict]:
        if not self.config.run_rlaifv:
            if "caption" in self.config.response_mode:
                raise ValueError(
                    "Response mode requires captions. Enable --run_rlaifv to generate captions."
                )
            return questions

        logger.info("Running RLAIF-V caption refinement")
        unique_images = self._unique_images(questions)
        refiner = CaptionRefiner(inference_params=self.config.inference)
        caption_items = [{"image": img} for img in unique_images]
        refined = refiner.process_dataset(
            caption_items,
            max_iterations=self.config.caption_max_iterations,
            max_workers=self.config.caption_max_workers
        )
        captions_path = self._output_path("captions.json")
        save_json(refined, captions_path)
        logger.info(f"Saved captions to {captions_path}")

        caption_map = {item["image"]: item.get("caption", "") for item in refined}
        all_captions_map = {item["image"]: item.get("all_captions", []) for item in refined}
        for item in questions:
            image = item.get("image")
            if image in caption_map:
                item["caption"] = caption_map[image]
                item["all_captions"] = all_captions_map.get(image, [])

        return questions

    def run(self, image_dir: str) -> Dict[str, str]:
        """
        Run the pipeline.

        Returns:
            Dict with output file paths.
        """
        image_paths = process_image_directory(image_dir)
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")

        logger.info(f"Loaded {len(image_paths)} images from {image_dir}")
        image_paths = self._maybe_filter(image_paths)
        if not image_paths:
            raise ValueError("No images left after filtering")

        questions = generate_questions(
            image_paths,
            inference_params=self.config.inference
        )
        questions = self._maybe_refine_captions(questions)

        questions_path = self._output_path("questions.json")
        save_json(questions, questions_path)
        logger.info(f"Saved questions to {questions_path}")

        responses = generate_responses(
            questions,
            mode=self.config.response_mode,
            inference_params=self.config.inference
        )
        for item, response in zip(questions, responses):
            item["response"] = response
        responses_path = self._output_path("responses.json")
        save_json(questions, responses_path)
        logger.info(f"Saved responses to {responses_path}")

        outputs = {
            "questions": questions_path,
            "responses": responses_path,
        }

        if self.config.run_cpt:
            cpt_results = generate_cpt(
                questions,
                version="critical_version",
                answer_key="response",
                inference_params=self.config.inference
            )
            cpt_path = self._output_path("responses_with_cpt.json")
            save_json(cpt_results, cpt_path)
            logger.info(f"Saved CPT results to {cpt_path}")
            outputs["cpt"] = cpt_path

        return outputs
