import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score

def exact_match(pred, gold):
    return str(pred).strip().lower() == str(gold).strip().lower()

def get_reward_value(rewards, strategy):
    rewards = np.array(rewards)
    rewards[np.isnan(rewards)] = 0.5
    if strategy == "min":
        return np.min(rewards).item()
    elif strategy == "max":
        return np.max(rewards).item()
    elif strategy == "mean":
        return np.mean(rewards).item()
    elif strategy == "prod":
        return np.prod(rewards).item()
    else:  # strategy == "last"
        return rewards[-1]

def evaluate_verification_with_threshold(dataset, strategy="mean", threshold=0.5):
    predictions, true_labels, scores = [], [], []
    for entry in dataset:        
        answers, rewards, gold = entry['parsed_answers'], entry['rewards'], entry['answer']
        for answer, reward_list in zip(answers, rewards):
            is_correct = 1 if exact_match(answer, gold) else 0
            true_labels.append(is_correct)
            reward_score = get_reward_value(reward_list, strategy)
            scores.append(reward_score)
            predictions.append(1 if reward_score > threshold else 0)
    return predictions, true_labels, scores

def calculate_metrics(predictions, true_labels, scores):
    if len(set(true_labels)) <= 1:
        f1 = f1_score(true_labels, predictions, zero_division=0)
        auprc = 0.0
    else:
        f1 = f1_score(true_labels, predictions, zero_division=0)
        auprc = average_precision_score(true_labels, scores)
    
    return {
        'f1': f1 * 100,
        'auprc': auprc * 100,
        'accuracy': np.mean(np.array(predictions) == np.array(true_labels)) * 100
    }

def evaluate_single_category(input_dir, category, strategy, threshold):
    eval_file = os.path.join(input_dir, f"eval_{category}_dataset.json")
    if not os.path.exists(eval_file):
        print(f"Warning: {eval_file} not found, skipping {category}")
        return None
        
    with open(eval_file) as f:
        dataset = json.load(f)
    
    predictions, true_labels, scores = evaluate_verification_with_threshold(dataset, strategy, threshold)
    metrics = calculate_metrics(predictions, true_labels, scores)
    
    return predictions, true_labels, scores, metrics, len(dataset), len(predictions)

def main():
    parser = argparse.ArgumentParser(description="Evaluate outcome verification using F1 score and AUPRC")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing eval_*_dataset.json files")
    parser.add_argument("--strategy", type=str, default="mean", choices=['min', 'max', 'mean', 'last', 'prod'])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args()

    categories = ['law', 'psychology', 'chemistry', 'biology', 'physics', 
                    'history', 'economics', 'math', 'business', 'philosophy', 
                    'health', 'engineering', 'computer_science', 'other']
    
    print(f"Evaluating all categories with strategy={args.strategy}, threshold={args.threshold}")
    
    all_predictions, all_true_labels, all_scores = [], [], []
    category_results = {}
    
    for category in tqdm(categories, desc="Processing categories"):
        result = evaluate_single_category(args.input_dir, category, args.strategy, args.threshold)
        if result is not None:
            predictions, true_labels, scores, metrics, distinct_questions, total_samples = result
            all_predictions.extend(predictions)
            all_true_labels.extend(true_labels)
            all_scores.extend(scores)
            category_results[category] = (metrics, distinct_questions, total_samples)
    
    if not category_results:
        print("No valid categories found!")
        return
    
    # Calculate overall metrics on combined data
    overall_metrics = calculate_metrics(all_predictions, all_true_labels, all_scores)
    total_samples = len(all_predictions)
    
    # Determine column width
    col_width = 6 + args.decimals if args.decimals > 0 else 6
    
    # Print results table
    print(f"\n" + "="*50)
    print(f"RESULTS BY CATEGORY")
    print(f"="*50)
    
    valid_categories = list(category_results.keys())
    cat_short = [cat[:4] for cat in valid_categories]
    header_cats = " ".join(f"{cat:<{col_width}}" for cat in cat_short)
    header = f"{'Metric':<8} {'Overall':<{col_width}} {header_cats}"
    print(header)
    print("-" * len(header))
    
    # F1 scores
    cat_f1_results = []
    for cat in valid_categories:
        metrics, _, _ = category_results[cat]
        cat_f1_results.append(f"{metrics['f1']:.{args.decimals}f}"[:col_width].ljust(col_width))
    
    overall_f1 = f"{overall_metrics['f1']:.{args.decimals}f}"[:col_width].ljust(col_width)
    print(f"{'F1':<8} {overall_f1} {' '.join(cat_f1_results)}")
    
    # AUPRC scores
    cat_auprc_results = []
    for cat in valid_categories:
        metrics, _, _ = category_results[cat]
        cat_auprc_results.append(f"{metrics['auprc']:.{args.decimals}f}"[:col_width].ljust(col_width))
    
    overall_auprc = f"{overall_metrics['auprc']:.{args.decimals}f}"[:col_width].ljust(col_width)
    print(f"{'AUPRC':<8} {overall_auprc} {' '.join(cat_auprc_results)}")

if __name__ == "__main__":
    main()