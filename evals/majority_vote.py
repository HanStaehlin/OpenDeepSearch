import json
import os
import argparse
import glob
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# --- Define patterns for filtering non-attempts (lowercase for case-insensitive matching) ---
NOT_ATTEMPTED_PATTERNS = [
    "i don't know", "i do not know", "unknown", "cannot answer", "can't answer",
    "unable to answer", "need more context", "unable to provide", "no information found",
    "i'm not sure", "error", "timed out", "could not find", "no definitive answer",
    "i cannot fulfill this request", "as an ai",
]
# ---

def find_trial_files(task_dir):
    """Finds all JSONL files matching the trial pattern in a directory."""
    pattern = os.path.join(task_dir, '*__trial*.jsonl')
    trial_files = glob.glob(pattern)
    trial_files.sort() 
    print(f"Found {len(trial_files)} trial files in {task_dir}. Sorted alphabetically.")
    if trial_files:
         print(f"Processing order starts with: {Path(trial_files[0]).name}")
    return trial_files

def is_not_attempted(answer_text):
    """Checks if the answer text matches any defined non-attempt patterns."""
    if not isinstance(answer_text, str) or not answer_text.strip():
        return True
    answer_lower = answer_text.lower()
    for pattern in NOT_ATTEMPTED_PATTERNS:
        if pattern in answer_lower:
            return True
    return False

def aggregate_and_vote(task_dir):
    """
    Reads trial files, aggregates, filters, votes (tie-breaking by filename),
    constructs final records, and collects voting statistics.
    Returns tuple: (final_output_records, statistics_dict) or (None, None)
    """
    trial_files = find_trial_files(task_dir)
    num_trial_files = len(trial_files)
    if not trial_files:
        print(f"No trial files found in {task_dir}. Skipping.")
        return None, None # Return None for both results and stats

    aggregated_data = defaultdict(list)
    total_records_read = 0 # Track total lines/records read across all files
    
    print("Reading and aggregating data from trial files...")
    for trial_file in tqdm(trial_files, desc="Reading Files"):
        file_path = Path(trial_file)
        filename = file_path.name 
        try:
            with open(trial_file, 'r', encoding='utf-8') as f:
                trial_index_str = filename.split('__trial')[-1].split('.')[0] 
                try:
                    trial_index = int(trial_index_str)
                except ValueError:
                    trial_index = -1 

                for line_num, line in enumerate(f, 1):
                    total_records_read += 1
                    try:
                        data = json.loads(line)
                        question = data.get('original_question')
                        answer = data.get('answer', None) 

                        if isinstance(answer, str):
                            processed_answer = answer.strip()
                        elif answer is None:
                             processed_answer = "" 
                        else:
                            processed_answer = str(answer)

                        if question:
                            aggregated_data[question].append({
                                'processed_answer': processed_answer,
                                'trial_index': trial_index,
                                'filename': filename, 
                                'full_record': data 
                            })
                            
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in {trial_file} (line {line_num})")
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num} in {trial_file} ({filename}): {e}")

        except Exception as e:
            print(f"Error reading file {trial_file}: {e}")

    # --- Initialize Statistics Counters ---
    stats = {
        "total_unique_questions": len(aggregated_data),
        "total_records_read": total_records_read,
        "num_trial_files_found": num_trial_files,
        "questions_clear_majority": 0,
        "questions_tie_breaker": 0,
        "questions_all_filtered": 0,
        "total_valid_records_voted": 0, # Sum of valid records across all questions voted on
    }
    # ---

    final_output_records = []
    print(f"\nFiltering non-attempts and performing majority vote for {stats['total_unique_questions']} unique questions...")

    for question, all_trial_records in tqdm(aggregated_data.items(), desc="Voting"):
        
        valid_trial_records = [
            r for r in all_trial_records 
            if not is_not_attempted(r['processed_answer'])
        ]
        
        num_valid_records = len(valid_trial_records)
        stats["total_valid_records_voted"] += num_valid_records # Add count for this question

        default_source_record = all_trial_records[0]['full_record'] 
        majority_answer = "FILTERED_AS_NON_ATTEMPT" 
        source_record = default_source_record 
        is_tie = False
        # vote_counts = {} # Can uncomment if needed in optional details

        if not valid_trial_records:
            # All answers were filtered out
            stats["questions_all_filtered"] += 1
            print(f"Warning: All answers for question '{question[:50]}...' were filtered as non-attempts.")
            pass 
        else:
            # Proceed with voting
            source_record = valid_trial_records[0]['full_record'] # Update default source to first valid one
            
            answers_for_voting = [r['processed_answer'] for r in valid_trial_records]
            vote_counts = Counter(answers_for_voting)
            most_common = vote_counts.most_common()

            if not most_common:
                 majority_answer = "ERROR_VOTING_FAILED"
            elif len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                # Clear majority among valid answers
                stats["questions_clear_majority"] += 1 # Increment stat
                majority_answer = most_common[0][0]
                is_tie = False
                matching_records = sorted(
                    [r for r in valid_trial_records if r['processed_answer'] == majority_answer],
                    key=lambda x: x['filename'] 
                )
                if matching_records:
                    source_record = matching_records[0]['full_record']
            else:
                # Tie detected among valid answers
                stats["questions_tie_breaker"] += 1 # Increment stat
                is_tie = True
                top_count = most_common[0][1]
                tied_answers = {ans for ans, count in most_common if count == top_count}
                
                tied_records_candidates = [
                    r for r in valid_trial_records 
                    if r['processed_answer'] in tied_answers
                ]
                
                tied_records_sorted_by_filename = sorted(
                    tied_records_candidates,
                    key=lambda x: x['filename'] 
                )
                
                if tied_records_sorted_by_filename:
                     chosen_record = tied_records_sorted_by_filename[0]
                     majority_answer = chosen_record['processed_answer']
                     source_record = chosen_record['full_record']
                     print(f"Warning: Tie detected for question '{question[:50]}...'. Used answer from file '{chosen_record['filename']}'.")
                else: 
                     majority_answer = most_common[0][0] 
                     print(f"Warning: Tie detected for question '{question[:50]}...' but failed to select tiebreaker. Using first most common.")

        # Construct the final output record
        final_record = source_record.copy() 
        final_record['answer'] = majority_answer 
        
        # --- Optional: Add extra fields ---
        # final_record['majority_vote_details'] = {
        #      'vote_counts': dict(vote_counts), 
        #      'total_trials_for_q': len(all_trial_records),
        #      'valid_trials_for_q': num_valid_records,
        #      'is_tie': is_tie,
        #      'source_filename': source_record.get('filename', 'N/A') # Assuming filename is stored in source_record if needed
        # }
        # ---

        final_output_records.append(final_record)

    # Return both the processed records and the collected statistics
    return final_output_records, stats

def save_results(results, output_file):
    """Saves the results to a JSONL file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in results:
                f.write(json.dumps(entry) + '\n')
        print(f"\nMajority vote results saved to: {output_file}")
    except Exception as e:
        print(f"\nError saving results to {output_file}: {e}")

def display_statistics(stats):
    """Prints a formatted summary of the voting statistics."""
    if not stats:
        print("\nNo statistics available.")
        return

    print("\n--- Aggregation and Voting Statistics ---")
    print(f"Trial Files Found:          {stats.get('num_trial_files_found', 'N/A')}")
    print(f"Total Records Read:         {stats.get('total_records_read', 'N/A')}")
    print(f"Total Unique Questions:     {stats.get('total_unique_questions', 'N/A')}")
    
    total_q = stats.get('total_unique_questions', 0)
    if total_q > 0:
        clear_maj = stats.get('questions_clear_majority', 0)
        tie_break = stats.get('questions_tie_breaker', 0)
        all_filt = stats.get('questions_all_filtered', 0)
        total_voted_on = clear_maj + tie_break # Questions that had at least one valid answer

        print("\nVoting Outcomes:")
        print(f"- Clear Majority:           {clear_maj} ({clear_maj/total_q:.1%})")
        print(f"- Decided by Tie-Breaker:   {tie_break} ({tie_break/total_q:.1%})")
        print(f"- All Answers Filtered:     {all_filt} ({all_filt/total_q:.1%})")
        
        total_valid_records = stats.get('total_valid_records_voted', 0)
        if total_voted_on > 0:
             avg_valid_per_q = total_valid_records / total_voted_on
             print(f"\nAverage valid answers per voted question: {avg_valid_per_q:.2f}")
        else:
             print("\nNo questions had valid answers to vote on.")
             
    print("-----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate trial results using majority vote after filtering non-attempts. Tie-breaks use alphabetically first filename. Displays summary statistics.')
    parser.add_argument(
        '--task_dir', 
        type=str, 
        required=True,
        help='Directory containing the trial JSONL files (e.g., .../codeact/frames_test_set/)'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default=None, 
        help='Path for the output JSONL file. If None, defaults to [task_dir_name]_majority_vote.jsonl in the parent directory.'
    )

    args = parser.parse_args()

    output_file = args.output_file
    if output_file is None:
        task_dir_path = Path(args.task_dir)
        output_filename = f"{task_dir_path.name}_majority_vote.jsonl"
        # Place output file in the parent directory of task_dir
        output_file = task_dir_path.parent / output_filename 

    # Run aggregation and voting, capture results and stats
    final_data, statistics = aggregate_and_vote(args.task_dir)

    # Save the results if any were generated
    if final_data:
        save_results(final_data, output_file)
    else:
        print("No data aggregated or processed.")

    # Display the collected statistics
    display_statistics(statistics)