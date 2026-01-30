#!/usr/bin/env python3
"""
MedMCQA Evaluation Script

Evaluates an LLM via OpenAI-compatible API on the MedMCQA medical QA benchmark.
Dataset: https://huggingface.co/datasets/openlifescienceai/medmcqa_formatted
"""

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from datasets import load_dataset
from tqdm.asyncio import tqdm


VALID_ANSWERS = {"A", "B", "C", "D"}

SYSTEM_PROMPT = """You are a medical expert taking a multiple choice exam. 
For each question, analyze the options carefully and select the single best answer.
Respond with ONLY the letter of your answer (A, B, C, or D) followed by a brief explanation.
Format: [ANSWER] followed by explanation."""


def format_question(sample: dict) -> str:
    """Format a MedMCQA sample into a question prompt."""
    data = sample["data"]
    question = data["Question"]
    options = data["Options"]
    
    options_text = "\n".join(f"{key}) {value}" for key, value in sorted(options.items()))
    
    prompt = f"""Question: {question}

{options_text}

What is the correct answer? Respond with the letter (A, B, C, or D) followed by your reasoning."""
    
    return prompt


def extract_answer(response: str) -> Optional[str]:
    """Extract the answer letter from model response."""
    if not response:
        return None
    
    response = response.strip()
    
    # Pattern 1: Answer starts with letter
    if response and response[0].upper() in VALID_ANSWERS:
        return response[0].upper()
    
    # Pattern 2: [A], [B], [C], [D] format
    bracket_match = re.search(r'\[([ABCD])\]', response, re.IGNORECASE)
    if bracket_match:
        return bracket_match.group(1).upper()
    
    # Pattern 3: "Answer: A" or "The answer is A" format
    answer_match = re.search(
        r'(?:answer|option|choice)[\s:]*(?:is\s+)?([ABCD])\b', 
        response, 
        re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).upper()
    
    # Pattern 4: Just look for first standalone A, B, C, or D
    letter_match = re.search(r'\b([ABCD])\b', response)
    if letter_match:
        return letter_match.group(1).upper()
    
    return None


async def query_llm_async(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    question: str,
    headers: Optional[dict] = None,
    max_retries: int = 2,
    retry_delay: float = 1.0,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> tuple[str, Optional[str]]:
    """Query the LLM asynchronously."""
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract response - handle both 'content' and 'reasoning' fields
            message = data["choices"][0]["message"]
            response_text = (
                message.get("content") 
                or message.get("reasoning_content") 
                or message.get("reasoning") 
                or ""
            )
            
            # Retry on empty response
            if not response_text.strip():
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    return "", None
            
            answer = extract_answer(response_text)
            return response_text, answer
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)
            else:
                return "", None
    
    return "", None


async def process_sample(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    sample: dict,
    headers: Optional[dict],
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Process a single sample with concurrency control."""
    async with semaphore:
        question_prompt = format_question(sample)
        response_text, predicted_answer = await query_llm_async(
            client=client,
            url=url,
            model=model,
            question=question_prompt,
            headers=headers,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        data = sample["data"]
        correct_answer = data["Correct Option"]
        subject = sample.get("subject_name", "Unknown")
        is_correct = predicted_answer == correct_answer if predicted_answer else False
        
        return {
            "id": sample["id"],
            "question": data["Question"],
            "options": data["Options"],
            "correct_answer": correct_answer,
            "correct_answer_text": data["Correct Answer"],
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "response": response_text,
            "subject": subject,
            "topic": sample.get("topic_name"),
            "explanation": sample.get("explanation"),
        }


async def run_evaluation_async(
    base_url: str,
    model: str,
    dataset,
    api_key: Optional[str] = None,
    num_samples: Optional[int] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    output_file: Optional[str] = None,
    concurrency: int = 10,
) -> dict:
    """Run evaluation on the dataset with concurrent requests."""
    
    results = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_samples": num_samples,
            "concurrency": concurrency,
        },
        "samples": [],
        "metrics": {},
    }
    
    samples = list(dataset)
    if num_samples:
        samples = samples[:num_samples]
    
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    semaphore = asyncio.Semaphore(concurrency)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [
            process_sample(
                client=client,
                url=url,
                model=model,
                sample=sample,
                headers=headers,
                temperature=temperature,
                max_tokens=max_tokens,
                semaphore=semaphore,
            )
            for sample in samples
        ]
        
        sample_results = await tqdm.gather(*tasks, desc="Evaluating")
    
    # Aggregate results
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_answered = 0
    
    for sample_result in sample_results:
        results["samples"].append(sample_result)
        
        if sample_result["predicted_answer"]:
            total_answered += 1
        if sample_result["is_correct"]:
            total_correct += 1
        
        subject = sample_result["subject"]
        subject_stats[subject]["total"] += 1
        if sample_result["is_correct"]:
            subject_stats[subject]["correct"] += 1
    
    total_samples = len(sample_results)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    answer_rate = total_answered / total_samples if total_samples > 0 else 0
    
    results["metrics"] = {
        "total_samples": total_samples,
        "total_correct": total_correct,
        "total_answered": total_answered,
        "accuracy": accuracy,
        "accuracy_pct": f"{accuracy * 100:.2f}%",
        "answer_rate": answer_rate,
        "answer_rate_pct": f"{answer_rate * 100:.2f}%",
    }
    
    # Subject-wise accuracy
    subject_metrics = {}
    for subject, stats in sorted(subject_stats.items()):
        subj_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        subject_metrics[subject] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": subj_acc,
            "accuracy_pct": f"{subj_acc * 100:.2f}%",
        }
    results["metrics"]["by_subject"] = subject_metrics
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    
    return results


def run_evaluation(
    base_url: str,
    model: str,
    dataset,
    api_key: Optional[str] = None,
    num_samples: Optional[int] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    output_file: Optional[str] = None,
    concurrency: int = 10,
) -> dict:
    """Wrapper to run async evaluation."""
    return asyncio.run(
        run_evaluation_async(
            base_url=base_url,
            model=model,
            dataset=dataset,
            api_key=api_key,
            num_samples=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            output_file=output_file,
            concurrency=concurrency,
        )
    )


def print_summary(results: dict):
    """Print evaluation summary."""
    metrics = results["metrics"]
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {results['model']}")
    print(f"Timestamp: {results['timestamp']}")
    print("-" * 60)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Correctly Answered: {metrics['total_correct']}")
    print(f"Answer Rate: {metrics['answer_rate_pct']}")
    print(f"Accuracy: {metrics['accuracy_pct']}")
    print("-" * 60)
    print("\nAccuracy by Subject:")
    print("-" * 60)
    
    for subject, stats in sorted(metrics["by_subject"].items(), 
                                  key=lambda x: x[1]["accuracy"], 
                                  reverse=True):
        print(f"  {subject:30s}: {stats['accuracy_pct']:>8s} ({stats['correct']}/{stats['total']})")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM on MedMCQA benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # API Configuration
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (or set OPENAI_API_KEY env var). Use --no-auth to skip.",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Skip authentication (for endpoints that don't require it)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name/ID to evaluate",
    )
    
    # Dataset Configuration
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)",
    )
    
    # Generation Configuration
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens in response",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests (higher = faster, but may overload server)",
    )
    
    # Output Configuration
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    print(f"Loading MedMCQA formatted dataset ({args.split} split)...")
    dataset = load_dataset("openlifescienceai/medmcqa_formatted", split=args.split)
    print(f"Loaded {len(dataset)} samples")
    
    # Determine output file
    output_file = args.output
    if not output_file:
        model_name = args.model.replace("/", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{args.output_dir}/{model_name}_{args.split}_{timestamp}.json"
    
    print("\nStarting evaluation...")
    print(f"  Model: {args.model}")
    print(f"  Base URL: {args.base_url}")
    print(f"  Split: {args.split}")
    print(f"  Samples: {args.num_samples or 'all'}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Concurrency: {args.concurrency}")
    print()
    
    # Determine API key (None if --no-auth is specified)
    api_key = None if args.no_auth else args.api_key
    
    results = run_evaluation(
        base_url=args.base_url,
        model=args.model,
        dataset=dataset,
        api_key=api_key,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_file=output_file,
        concurrency=args.concurrency,
    )
    
    print_summary(results)


if __name__ == "__main__":
    main()
