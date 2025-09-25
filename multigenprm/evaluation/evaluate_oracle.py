import argparse
import ast
import json
import math
import numpy as np
import os
from datasets import load_dataset
from collections import defaultdict, Counter

def exact_match(pred, gold):
    return str(pred).strip().lower() == str(gold).strip().lower()

def evaluate_methods(dataset, N_max):
    majority_vote_correct = 0
    oracle_correct = 0
    total = len(dataset)

    for entry in dataset:
        parsed_answers = entry['parsed_answers']
        indices = np.random.permutation(len(parsed_answers))[:N_max]
        parsed_answers = [ parsed_answers[idx] for idx in indices ]
        gold_answer = entry['answer']

        # Simple majority vote (unweighted)
        answer_counter = Counter()
        for ans in parsed_answers:
            answer_counter[ans.strip().lower()] += 1
        majority_pred = answer_counter.most_common(1)[0][0]
        if exact_match(majority_pred, gold_answer):
            majority_vote_correct += 1

        # Oracle: check if any answer matches the gold answer
        answer_in_parsed_answer = False
        for a in parsed_answers:
            answer_in_parsed_answer = answer_in_parsed_answer or exact_match(a, gold_answer)
        if answer_in_parsed_answer:
            oracle_correct += 1

    accuracy = {
        'majority_vote_accuracy': majority_vote_correct / total,
        'oracle_accuracy': oracle_correct / total
    }
    return accuracy    
    

def main():
    parser = argparse.ArgumentParser(description="Generate text with VLLM and extract Yes/No probabilities")    
    parser.add_argument("--category", type=str, default="law", choices={
        'law', 'psychology', 'chemistry', 'biology', 'physics', 
        'history', 'economics', 'math', 'business', 'philosophy', 
        'health', 'engineering', 'computer_science', 'other'
    }, help="Category of problems to process")
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--N_max", type=int, default=16)
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset for category: {args.category}...")
    dataset = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Eval", split="test")
    category_name = "computer science" if args.category == "computer_science" else args.category
    dataset = [d for d in dataset if d["category"] == category_name]
    q_ids = list(set([d["id"] for d in dataset]))

    # Preprocess dataset as list
    preprocessed_dataset = []
    for q_id in q_ids:
        subset = [d for d in dataset if d["id"] == q_id]
        data = {
            "q_id": q_id,
            "question": subset[0]["question"],
            "cots": [],
            "parsed_answers": [],
            "answer": subset[0]["answer"]
        }
        for d in subset:
            data["cots"].append(ast.literal_eval(d["chain_of_thoughts"]))
            data["parsed_answers"].append(d["parsed_answer"])
        preprocessed_dataset.append(data)    
    dataset = preprocessed_dataset
    
    print(f"Total questions: {len(dataset)}")

    accuracy = {
        'majority_vote_accuracy': [],
        'oracle_accuracy': []
    }
    for _ in range(args.num_runs):
        results = evaluate_methods(dataset, args.N_max)
        for key in accuracy:
            accuracy[key].append(results[key])
    for key in accuracy:
        array = np.array(accuracy[key])*100
        mean = np.round(np.mean(array), 2)
        std = np.round(np.std(array), 2)
        max = np.round(np.max(array), 2)
        min = np.round(np.min(array), 2)
        print(f"{key}, mean: {mean}, std: {std}, max: {max}, min: {min}")        
    
if __name__ == "__main__":
    main()