import pandas as pd
import json
from utils.jsonl import read_jsonl, write_jsonl


def gen_summary(results_path: str, summary_path: str, model_stats: dict = None):
    results = pd.DataFrame(read_jsonl(results_path))

    if "api_calls" not in results:
        results["api_calls"] = 1

    solved = len(results.query("is_solved == True"))
    unsolved = len(results.query("is_solved == False"))

    accuracy = solved / (solved + unsolved)

    # normal_solved = len(results.query("is_solved == True & api_calls == 2"))
    # our_solved = len(results.query("is_solved == True & api_calls > 2"))

    total_prompt_tokens = results['run_details'].apply(lambda x: sum(run['prompt_tokens'] for run in x)).sum()
    total_completion_tokens = results['run_details'].apply(lambda x: sum(run['completion_tokens'] for run in x)).sum()
    total_taken_time = results['run_details'].apply(lambda x: sum(run['taken_time'] for run in x)).sum()
    # total_cost = results['run_details'].apply(lambda x: sum(run['cost'] for run in x)).sum()

    average_prompt_tokens = total_prompt_tokens / len(results)
    average_completion_tokens = total_completion_tokens / len(results)
    average_taken_time = total_taken_time / len(results)
    
    total_api_calls = results['run_details'].apply(lambda x: sum(run['api_calls'] for run in x)).sum()
    max_api_calls = results['run_details'].apply(lambda x: sum(run['api_calls'] for run in x)).max()
    min_api_calls = results['run_details'].apply(lambda x: sum(run['api_calls'] for run in x)).min()
    average_api_calls = total_api_calls / len(results)

    false_results = results.query("is_solved == False")['run_details'].apply(lambda x: sum(run['api_calls'] for run in x)).value_counts()
    true_results = results.query("is_solved == True")['run_details'].apply(lambda x: sum(run['api_calls'] for run in x)).value_counts()

    with open(summary_path, mode="w", encoding="utf-8") as summary_file:
        # Define a width for alignment
        name_width = 30
        value_width = 10

        # Write model configuration if model_stats is provided (dual-model mode)
        if model_stats and model_stats.get('model2'):
            summary_file.write("=" * 70 + "\n")
            summary_file.write("=== Model Configuration ===\n")
            summary_file.write(f"Model 1: {model_stats.get('model1_path', 'N/A')}\n")
            summary_file.write(f"Model 2: {model_stats.get('model2_path', 'N/A')}\n")
            summary_file.write("=" * 70 + "\n")
            summary_file.write("\n")

        summary_file.write(f"{'Accuracy:':<{name_width}} {accuracy*100:>{value_width}.01f}\n")
        summary_file.write(f"{'Solved:':<{name_width}} {solved:>{value_width}}\n")
        summary_file.write(f"{'Unsolved:':<{name_width}} {unsolved:>{value_width}}\n")
        # summary_file.write(f"\n")
        # summary_file.write(f"{'Normal Solved:':<{name_width}} {normal_solved:>{value_width}}\n")
        # summary_file.write(f"{'Our Solved:':<{name_width}} {our_solved:>{value_width}}\n")
        summary_file.write(f"\n")
        summary_file.write(f"\n")
        summary_file.write(f"{'Total Prompt Tokens:':<{name_width}} {total_prompt_tokens:>{value_width}}\n")
        summary_file.write(f"{'Average Prompt Tokens:':<{name_width}} {average_prompt_tokens:>{value_width}.0f}\n")
        summary_file.write(f"\n")
        summary_file.write(f"{'Total Completion Tokens:':<{name_width}} {total_completion_tokens:>{value_width}}\n")
        summary_file.write(f"{'Average Completion Tokens:':<{name_width}} {average_completion_tokens:>{value_width}.0f}\n")
        summary_file.write(f"\n")
        summary_file.write(f"{'Total Taken Time:':<{name_width}} {total_taken_time:>{value_width}.02f}s\n")
        summary_file.write(f"{'Average Taken Time:':<{name_width}} {average_taken_time:>{value_width}.02f}s\n")
        summary_file.write(f"\n")
        # summary_file.write(f"{'Total Cost:':<{name_width}} {total_cost:>{value_width}.02f}\n")
        summary_file.write(f"\n")
        summary_file.write(f"{'Total Api Calls:':<{name_width}} {total_api_calls:>{value_width}.02f}\n")
        summary_file.write(f"{'Max Api Calls:':<{name_width}} {max_api_calls:>{value_width}}\n")
        summary_file.write(f"{'Min Api Calls:':<{name_width}} {min_api_calls:>{value_width}}\n")
        summary_file.write(f"{'Average Api Calls:':<{name_width}} {average_api_calls:>{value_width}.02}\n")
        summary_file.write(f"\n")
        summary_file.write(f"\n")
        summary_file.write(f"{'Solved Api Calls':<{name_width}}\n")
        summary_file.write(f"{'Api calls':<{name_width}} {'Solved':>{value_width}}\n")
        # Printing all keys and their values (Solved)
        for key, value in true_results.items():
            summary_file.write(f"{key:<{name_width}} {value:>{value_width}}\n")
        summary_file.write(f"\n")
        summary_file.write(f"{'Unsolved Api Calls':<{name_width}}\n")
        summary_file.write(f"{'Api calls':<{name_width}} {'Unsolved':>{value_width}}\n")
        # Printing all keys and their values (Unsolved)
        for key, value in false_results.items():
            summary_file.write(f"{key:<{name_width}} {value:>{value_width}}\n")
        
        # Write model-wise statistics if dual-model mode
        if model_stats and model_stats.get('model2'):
            summary_file.write(f"\n")
            summary_file.write("=" * 70 + "\n")
            summary_file.write("=== Model-wise Statistics ===\n")
            summary_file.write(f"\n")
            
            # Calculate model statistics from Results.jsonl
            model1_calls = 0
            model1_prompt_tokens = 0
            model1_completion_tokens = 0
            model2_calls = 0
            model2_prompt_tokens = 0
            model2_completion_tokens = 0
            
            for _, row in results.iterrows():
                for rd in row['run_details']:
                    model1_calls += rd.get('model1_calls', 0)
                    model1_prompt_tokens += rd.get('model1_prompt_tokens', 0)
                    model1_completion_tokens += rd.get('model1_completion_tokens', 0)
                    model2_calls += rd.get('model2_calls', 0)
                    model2_prompt_tokens += rd.get('model2_prompt_tokens', 0)
                    model2_completion_tokens += rd.get('model2_completion_tokens', 0)
            
            # Model 1 Statistics
            summary_file.write("Model 1 Statistics:\n")
            summary_file.write(f"  {'Total Api Calls:':<{name_width-2}} {model1_calls:>{value_width}}\n")
            summary_file.write(f"  {'Total Prompt Tokens:':<{name_width-2}} {model1_prompt_tokens:>{value_width}}\n")
            summary_file.write(f"  {'Total Completion Tokens:':<{name_width-2}} {model1_completion_tokens:>{value_width}}\n")
            summary_file.write(f"\n")
            
            # Model 2 Statistics
            summary_file.write("Model 2 Statistics:\n")
            summary_file.write(f"  {'Total Api Calls:':<{name_width-2}} {model2_calls:>{value_width}}\n")
            summary_file.write(f"  {'Total Prompt Tokens:':<{name_width-2}} {model2_prompt_tokens:>{value_width}}\n")
            summary_file.write(f"  {'Total Completion Tokens:':<{name_width-2}} {model2_completion_tokens:>{value_width}}\n")

