import pandas as pd
import litellm
import argparse
import os
import re
from evals.grader_prompts import GRADER_TEMPLATE
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def grade_row(row_data):
    idx, row = row_data
    question = row['original_question']
    predicted_answer = row['answer']
    gold_answer = row['true_answer']
    
    input_prompt = GRADER_TEMPLATE.format(
        question=question,
        predicted_answer=predicted_answer,
        target=gold_answer
    )
    
    try:
        output = litellm.completion(
            model="fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
            messages=[{"role": "user", "content": input_prompt}],
            temperature=0.0
        )['choices'][0]['message']['content']
        return idx, output
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return idx, "Error"

def compute_accuracy(evaluations):
    """
    Compute accuracy from a list of evaluation texts using A, B, C labels
    A = CORRECT, B = INCORRECT, C = NOT_ATTEMPTED
    """
    # Count the occurrences of each grade
    grade_a_count = sum(1 for eval_text in evaluations if re.search(r'\bA\b', eval_text))
    grade_b_count = sum(1 for eval_text in evaluations if re.search(r'\bB\b', eval_text))
    grade_c_count = sum(1 for eval_text in evaluations if re.search(r'\bC\b', eval_text))
    
    total_evaluated = grade_a_count + grade_b_count + grade_c_count
    
    if total_evaluated == 0:
        print("Warning: No valid evaluation results found (no A or B grades).")
        return 0.0
    
    accuracy = grade_a_count / total_evaluated
    
    print(f"Evaluation summary:")
    print(f"- CORRECT (A): {grade_a_count}")
    print(f"- INCORRECT (B): {grade_b_count}")
    print(f"- NOT_ATTEMPTED (C): {grade_c_count}")
    print(f"- Total evaluated (A+B+C): {total_evaluated}")
    
    return accuracy

def autograde_df(df_path, num_cpus=4):
    # Read the dataframe
    df = pd.read_json(df_path, lines=True)
    
    # Prepare data for parallel processing
    row_data = list(df.iterrows())
    
    # Use specified number of CPU cores
    n_processes = max(1, min(num_cpus, cpu_count()))
    print(f"Using {n_processes} processes")
    
    # Create process pool and process rows in parallel
    with Pool(n_processes) as pool:
        # Use tqdm for progress bar
        results = list(tqdm(
            pool.imap(grade_row, row_data),
            total=len(row_data),
            desc="Grading"
        ))
    
    # Sort results by index and extract grades
    results.sort(key=lambda x: x[0])
    final_grades = [grade for _, grade in results]
    
    # Add the grades as a new column
    df['final_grade'] = final_grades
    
    # Calculate and print accuracy
    accuracy = compute_accuracy(final_grades)
    print(f"Final accuracy: {accuracy:.2%}")
    
    # Create a simplified dataframe with only the needed columns
    simplified_df = pd.DataFrame({
        'question': df['original_question'],
        'answer': df['answer'],
        'true_answer': df['true_answer'],
        'evaluation': df['final_grade'],
    })
    
    # Create a new output filename
    base_path, ext = os.path.splitext(df_path)
    output_path = f"{base_path}_graded{ext}"
    
    # Save the simplified dataframe to the new file
    simplified_df.to_json(output_path, orient='records', lines=True)
    print(f"Grading completed and results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto-grade answers in a DataFrame')
    parser.add_argument('--df_path', type=str, default="/Users/hannesstahlin/Documents/OpenDeepSearch/output/fireworks_ai__accounts__fireworks__models__llama-v3p3-70b-instruct/codeact/frames_test_set/fireworks_ai__accounts__fireworks__models__llama-v3p3-70b-instruct__codeact__frames_test_set__trial0.jsonl", help='Path to the DataFrame JSON file')
    parser.add_argument('--num_cpus', type=int, default=4, help='Number of CPU cores to use')
    
    args = parser.parse_args()
    autograde_df(args.df_path, args.num_cpus)