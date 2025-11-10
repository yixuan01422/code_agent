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

    total_prompt_tokens = results['run_details'].apply(lambda x: sum(run.get('prompt_tokens', 0) for run in x)).sum()
    total_completion_tokens = results['run_details'].apply(lambda x: sum(run.get('completion_tokens', 0) for run in x)).sum()
    total_taken_time = results['run_details'].apply(lambda x: sum(run.get('taken_time', 0) for run in x)).sum()
    # total_cost = results['run_details'].apply(lambda x: sum(run['cost'] for run in x)).sum()

    average_prompt_tokens = total_prompt_tokens / len(results) if total_prompt_tokens > 0 else 0
    average_completion_tokens = total_completion_tokens / len(results) if total_completion_tokens > 0 else 0
    average_taken_time = total_taken_time / len(results) if total_taken_time > 0 else 0
    
    total_api_calls = results['run_details'].apply(lambda x: sum(run.get('api_calls', 0) for run in x)).sum()
    max_api_calls = results['run_details'].apply(lambda x: sum(run.get('api_calls', 0) for run in x)).max()
    min_api_calls = results['run_details'].apply(lambda x: sum(run.get('api_calls', 0) for run in x)).min()
    average_api_calls = total_api_calls / len(results) if total_api_calls > 0 else 0

    false_results = results.query("is_solved == False")['run_details'].apply(lambda x: sum(run.get('api_calls', 0) for run in x)).value_counts()
    true_results = results.query("is_solved == True")['run_details'].apply(lambda x: sum(run.get('api_calls', 0) for run in x)).value_counts()

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
        summary_file.write(f"\n")
        
        # Only show token/time/api statistics if data exists
        if total_prompt_tokens > 0 or total_completion_tokens > 0:
            summary_file.write(f"\n")
            summary_file.write(f"{'Total Prompt Tokens:':<{name_width}} {total_prompt_tokens:>{value_width}}\n")
            summary_file.write(f"{'Average Prompt Tokens:':<{name_width}} {average_prompt_tokens:>{value_width}.0f}\n")
            summary_file.write(f"\n")
            summary_file.write(f"{'Total Completion Tokens:':<{name_width}} {total_completion_tokens:>{value_width}}\n")
            summary_file.write(f"{'Average Completion Tokens:':<{name_width}} {average_completion_tokens:>{value_width}.0f}\n")
            summary_file.write(f"\n")
        
        if total_taken_time > 0:
            summary_file.write(f"{'Total Taken Time:':<{name_width}} {total_taken_time:>{value_width}.02f}s\n")
            summary_file.write(f"{'Average Taken Time:':<{name_width}} {average_taken_time:>{value_width}.02f}s\n")
            summary_file.write(f"\n")
        
        if total_api_calls > 0:
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
                if key > 0:  # Only show non-zero api calls
                    summary_file.write(f"{key:<{name_width}} {value:>{value_width}}\n")
            summary_file.write(f"\n")
            summary_file.write(f"{'Unsolved Api Calls':<{name_width}}\n")
            summary_file.write(f"{'Api calls':<{name_width}} {'Unsolved':>{value_width}}\n")
            # Printing all keys and their values (Unsolved)
            for key, value in false_results.items():
                if key > 0:  # Only show non-zero api calls
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
            
            # Calculate loss statistics if available
            planning_losses = []
            codegen_losses = []
            debug_losses = []
            
            for _, row in results.iterrows():
                for rd in row['run_details']:
                    # Collect planning loss
                    if 'planning_loss' in rd:
                        planning_losses.append(rd['planning_loss'])
                    
                    # Collect code generation loss
                    if 'codegen_loss' in rd:
                        codegen_losses.append(rd['codegen_loss'])
                    
                    # Collect all debug losses (debug_1_loss, debug_2_loss, etc.)
                    for key, val in rd.items():
                        if key.startswith('debug_') and key.endswith('_loss'):
                            debug_losses.append(val)
            
            # Write loss statistics if any loss data exists
            if planning_losses or codegen_losses or debug_losses:
                summary_file.write(f"\n")
                summary_file.write("=" * 70 + "\n")
                summary_file.write("=== Loss Statistics (Model2 evaluating Model1) ===\n")
                summary_file.write("=" * 70 + "\n")
                summary_file.write(f"\n")
                
                # Planning Loss
                if planning_losses:
                    avg_planning = sum(planning_losses) / len(planning_losses)
                    min_planning = min(planning_losses)
                    max_planning = max(planning_losses)
                    summary_file.write("Planning Phase:\n")
                    summary_file.write(f"  {'Total samples:':<{name_width-2}} {len(planning_losses):>{value_width}}\n")
                    summary_file.write(f"  {'Average Loss:':<{name_width-2}} {avg_planning:>{value_width}.4f}\n")
                    summary_file.write(f"  {'Min Loss:':<{name_width-2}} {min_planning:>{value_width}.4f}\n")
                    summary_file.write(f"  {'Max Loss:':<{name_width-2}} {max_planning:>{value_width}.4f}\n")
                    summary_file.write(f"\n")
                
                # Code Generation Loss
                if codegen_losses:
                    avg_codegen = sum(codegen_losses) / len(codegen_losses)
                    min_codegen = min(codegen_losses)
                    max_codegen = max(codegen_losses)
                    summary_file.write("Code Generation Phase:\n")
                    summary_file.write(f"  {'Total samples:':<{name_width-2}} {len(codegen_losses):>{value_width}}\n")
                    summary_file.write(f"  {'Average Loss:':<{name_width-2}} {avg_codegen:>{value_width}.4f}\n")
                    summary_file.write(f"  {'Min Loss:':<{name_width-2}} {min_codegen:>{value_width}.4f}\n")
                    summary_file.write(f"  {'Max Loss:':<{name_width-2}} {max_codegen:>{value_width}.4f}\n")
                    summary_file.write(f"\n")
                
                # Debug Loss
                if debug_losses:
                    avg_debug = sum(debug_losses) / len(debug_losses)
                    min_debug = min(debug_losses)
                    max_debug = max(debug_losses)
                    summary_file.write("Debugging Phase:\n")
                    summary_file.write(f"  {'Total samples:':<{name_width-2}} {len(debug_losses):>{value_width}}\n")
                    summary_file.write(f"  {'Average Loss:':<{name_width-2}} {avg_debug:>{value_width}.4f}\n")
                    summary_file.write(f"  {'Min Loss:':<{name_width-2}} {min_debug:>{value_width}.4f}\n")
                    summary_file.write(f"  {'Max Loss:':<{name_width-2}} {max_debug:>{value_width}.4f}\n")
                    summary_file.write(f"\n")
                
                # Overall Average Loss
                all_losses = planning_losses + codegen_losses + debug_losses
                if all_losses:
                    overall_avg = sum(all_losses) / len(all_losses)
                    summary_file.write("Overall Statistics:\n")
                    summary_file.write(f"  {'Total loss samples:':<{name_width-2}} {len(all_losses):>{value_width}}\n")
                    summary_file.write(f"  {'Overall Average Loss:':<{name_width-2}} {overall_avg:>{value_width}.4f}\n")

