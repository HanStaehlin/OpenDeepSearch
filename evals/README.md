# Evaluation Scripts

This repository contains scripts for running evaluations and autograding on model outputs.

## Available Commands

### Autograde DataFrame Evaluation
To evaluate and autograde DataFrame outputs:

```bash
python -m evals.autograde_dataframe --csv_path <path_to_csv> --output_path <path_to_output_csv>
```

Example:

```bash
python evals/autograde_df.py output/fireworks_ai__accounts__fireworks__models__qwq-32b/codeact/simple_qa_test_set/fireworks_ai__accounts__fireworks__models__qwq-32b__codeact__simple_qa_test_set__trial1.jsonl
```

This command processes the specified JSONL file and performs automated grading on DataFrame outputs.

### Run Task Evaluations
To run evaluations on a dataset with parallel processing:

```bash
python ./evals/eval_tasks.py --parallel-workers=8 --num-trials=1 --eval-tasks=./evals/datasets/frames_test_set.csv ./evals/datasets/simple_qa_test_set.csv
```

Parameters:
- `--date`: Optional date for the evaluation
- `--eval-tasks`: List of paths to CSV files containing evaluation tasks (default: ["./evals/datasets/frames_test_set.csv", "./evals/datasets/simple_qa_test_set.csv"])
- `--search-model-id`: Model ID for the search tool (default: "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct")
- `--model-type`: Type of model to use, either "LiteLLMModel" or "HfApiModel" (default: "LiteLLMModel")
- `--model-id`: ID of the model to use (default: "fireworks_ai/accounts/fireworks/models/qwq-32b")
- `--agent-action-type`: Type of agent action: "codeact", "tool-calling", or "vanilla" (default: "codeact")
- `--parallel-workers`: Number of parallel workers to use (default: 8)
- `--num-trials`: Number of evaluation trials to run (default: 1)

The results will be saved as a DataFrame in the `evals` directory.

## Output
Evaluation results are stored in the following locations:
- Task evaluation results: `evals/` directory
- DataFrame autograding results: Generated in the script's output


