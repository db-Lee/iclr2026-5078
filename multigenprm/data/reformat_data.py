import argparse
import json
import ast
import os
from datasets import load_dataset

def load_train_dataset(category: str) -> list[dict]:
    print(f"Loading train dataset for category: {category}...")
    dataset = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train")        
    
    category_name = "computer science" if category == "computer_science" else category
    dataset = [d for d in dataset if d["category"] == category_name]

    # Preprocess dataset as list
    preprocessed_dataset, q_cot_set = [], set()
    for data in dataset:
        preprocessed_data = {
            "q_id": data["id"], 
            "question": data["question"], 
            "cot_id": data["cot_id"], 
            "parsed_answer": data["parsed_answer"],
            "answer": data["answer"],                                     
        }
        
        cot, _cot = [], ast.literal_eval(data["chain_of_thoughts"])
        labels, _labels = [], ast.literal_eval(data["labels"])
        
        # empty string
        for step, l in zip(_cot, _labels):
            if step == "":
                continue
            cot.append(step); labels.append(l)
        
        parsed_answer, answer = data["parsed_answer"], data["answer"]
        
        # The last reasoning step is an answer prediction
        # If the answer prediction is wrong, labels[-1] should be -1. but there are some 1s
        if parsed_answer != answer and \
            labels[-1] == 1 and \
                "The answer is" in cot[-1]:
            labels[-1] = -1
        
        # If the answer prediction is correct, labels[-1] should be 1 unless the previous step is wrong.
        if parsed_answer == answer and \
            (labels[-1] == -1 and labels[-2] == 1) and \
                "The answer is" in cot[-1]:
            labels[-1] = 1
            
        preprocessed_data["cot"] = cot
        preprocessed_data["labels"] = labels
        
        assert len(preprocessed_data["cot"]) == len(preprocessed_data["labels"])
        
        preprocessed_dataset.append(preprocessed_data)
        q_cot_set.add((preprocessed_data["q_id"], preprocessed_data["cot_id"]))
    
    assert len(preprocessed_dataset) == len(q_cot_set)
    
    return preprocessed_dataset

def load_test_dataset(category: str) -> list[dict]:
    print(f"Loading test dataset for category: {category}...")
    dataset = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Eval", split="test")
    
    category_name = "computer science" if category == "computer_science" else category
    dataset = [d for d in dataset if d["category"] == category_name]
    q_ids = list(set([d["id"] for d in dataset]))

    # Preprocess dataset as list
    preprocessed_dataset = []
    for q_id in q_ids:
        subset = [data for data in dataset if data["id"] == q_id]
        
        assert len(set(data["question"] for data in subset)) == 1, \
            f"Multiple questions for q_id {q_id}"
        assert len(set(data["answer"] for data in subset)) == 1, \
            f"Multiple answers for q_id {q_id}"
        
        preprocessed_data = {
            "q_id": q_id,
            "question": subset[0]["question"],
            "cot_ids": [],
            "cots": [],
            "parsed_answers": [],
            "answer": subset[0]["answer"]
        }
        for d in subset:
            preprocessed_data["cot_ids"].append(data["cot_id"])
            cot, _cot = [], ast.literal_eval(data["chain_of_thoughts"])
            # empty string
            for step in _cot:
                if step == "":
                    continue
                cot.append(step)
            preprocessed_data["cots"].append(cot)
            preprocessed_data["parsed_answers"].append(data["parsed_answer"])
        preprocessed_dataset.append(preprocessed_data)    
    return preprocessed_dataset

def main():
    parser = argparse.ArgumentParser(description="Reformatting datasets")
    parser.add_argument("--output_dir", type=str, default="./datasets", help="Directory to save results")
    parser.add_argument("--category", type=str, default="law", 
                       choices=['law', 'psychology', 'chemistry', 'biology', 'physics', 
                               'history', 'economics', 'math', 'business', 'philosophy', 
                               'health', 'engineering', 'computer_science', 'other', 'all'],
                       help="Category of problems to process")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)    
    os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)    
    
    categories = (['law', 'psychology', 'chemistry', 'biology', 'physics', 
                   'history', 'economics', 'math', 'business', 'philosophy', 
                   'health', 'engineering', 'computer_science', 'other'] 
                  if args.category == 'all' else [args.category])
    
    for category in categories:
        args.category = category
        print(f"Reformatting category: {category}")
        
        # train dataset
        train_dataset = load_train_dataset(category)
        with open(os.path.join(args.output_dir, "train", f"{category}_dataset.json"), "w") as f:
            json.dump(train_dataset, f, indent=4)
            
        # test dataset
        test_dataset = load_test_dataset(category)
        with open(os.path.join(args.output_dir, "test", f"{category}_dataset.json"), "w") as f:
            json.dump(test_dataset, f, indent=4)

if __name__ == "__main__":
    main()