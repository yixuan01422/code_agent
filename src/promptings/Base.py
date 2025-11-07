from typing import List
import tiktoken
import os
import copy
import time
import torch

from models.Base import BaseModel
from datasets.Dataset import Dataset
from results.Results import Results
from utils.parse import parse_response
from time import perf_counter_ns
from constants.verboseType import *

class BaseStrategy(object):
    def __init__(
        self,
        model: BaseModel,
        data: Dataset,
        language: str,
        pass_at_k: int,
        results: Results,
        verbose: int = VERBOSE_FULL,
        model2: BaseModel = None,
        model1_path: str = None,
        model2_path: str = None,
        enable_loss_calculation: bool = False,
    ):
        self.model = model  # model1
        self.model2 = model2  # model2 (optional)
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.enable_loss_calculation = enable_loss_calculation
        
        self.data = data
        self.pass_at_k = pass_at_k
        self.results = results
        self.language = language
        self.verbose = verbose
        self.run_details = []
        
        # Statistics for dual-model mode
        self.model1_stats = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0
        }
        self.model2_stats = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0
        }
        
        # GPU device mapping for loss calculation mode
        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            # Map physical GPU to virtual index
            gpu_list = cuda_visible.split(',')
            model1_physical = os.getenv("MODEL1_GPU_PHYSICAL", gpu_list[0])
            model2_physical = os.getenv("MODEL2_GPU_PHYSICAL", gpu_list[1] if len(gpu_list) > 1 else gpu_list[0])
            try:
                self.model1_device = f"cuda:{gpu_list.index(model1_physical)}"
                self.model2_device = f"cuda:{gpu_list.index(model2_physical)}"
            except ValueError:
                self.model1_device = "cuda:0"
                self.model2_device = "cuda:1" if len(gpu_list) > 1 else "cuda:0"
        else:
            self.model1_device = "cuda:0"
            self.model2_device = "cuda:0"
    

    def append_run_details(self, run_details: dict):
        for key in run_details.keys():
            if key in self.run_details:
                self.run_details[key] += run_details[key]
            else:
                self.run_details[key] = run_details[key]


    def gpt_chat(
            self, 
            processed_input: List[dict], 
            frequency_penalty=0, 
            presence_penalty=0,
            use_model=1
        ):
        
        # Select model based on use_model parameter
        if use_model == 1:
            model = self.model
        elif use_model == 2:
            if self.model2 is None:
                raise ValueError("Model 2 is not configured but was requested")
            model = self.model2
        else:
            raise ValueError(f"Invalid use_model parameter: {use_model}. Must be 1 or 2.")
        
        response, run_details = model.prompt(
            processed_input=processed_input, 
            frequency_penalty=frequency_penalty, 
            presence_penalty=presence_penalty
        )
        self.append_run_details(run_details)
        
        # Record model-specific statistics in run_details
        model_key = f"model{use_model}_calls"
        model_prompt_key = f"model{use_model}_prompt_tokens"
        model_completion_key = f"model{use_model}_completion_tokens"
        
        if model_key not in self.run_details:
            self.run_details[model_key] = 0
            self.run_details[model_prompt_key] = 0
            self.run_details[model_completion_key] = 0
        
        self.run_details[model_key] += 1
        self.run_details[model_prompt_key] += run_details.get("prompt_tokens", 0)
        self.run_details[model_completion_key] += run_details.get("completion_tokens", 0)
        
        # Update in-memory statistics (for current session)
        stats = self.model1_stats if use_model == 1 else self.model2_stats
        stats["calls"] += 1
        if "prompt_tokens" in run_details:
            stats["prompt_tokens"] += run_details["prompt_tokens"]
        if "completion_tokens" in run_details:
            stats["completion_tokens"] += run_details["completion_tokens"]
        
        return response


    def gpt_chat_with_loss(
        self,
        processed_input: List[dict],
        generate_model: int = 1,
        loss_model: int = 2,
        calculate_reverse: bool = False,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        **kwargs
    ):
        """
        Generate text with one model and calculate loss with another.
        Uses PyTorch models directly (not vLLM) for both generation and loss calculation.
        
        Args:
            processed_input: Input messages in OpenAI format [{"role": "user", "content": "..."}]
            generate_model: Model for generation (1 or 2)
            loss_model: Model for loss calculation (1 or 2)
            calculate_reverse: If True, also swap models and calculate reverse loss
            temperature: Sampling temperature (default: from model config)
            top_p: Nucleus sampling parameter (default: from model config)
            max_tokens: Maximum tokens to generate (default: from model config)
            **kwargs: Additional arguments (ignored for PyTorch generation)
        
        Returns:
            dict: {
                'text': str,           # Generated text from generate_model
                'loss': float,         # Loss of loss_model on generated text
                'reverse_text': str,   # (if calculate_reverse) Text from loss_model
                'reverse_loss': float  # (if calculate_reverse) Loss of generate_model
            }
        """
        from utils.loss_calculator import LossCalculator
        
        # Get model paths and devices
        generate_model_path = self.model1_path if generate_model == 1 else self.model2_path
        loss_model_path = self.model1_path if loss_model == 1 else self.model2_path
        generate_device = self.model1_device if generate_model == 1 else self.model2_device
        loss_device = self.model1_device if loss_model == 1 else self.model2_device
        
        if generate_model_path is None:
            raise ValueError(f"Model {generate_model} path is not configured")
        if loss_model_path is None:
            raise ValueError(f"Model {loss_model} path is not configured")
        
        # Get generation parameters from model config (if not provided)
        if temperature is None:
            temperature = self.model.temperature if hasattr(self.model, 'temperature') else 0.0
        if top_p is None:
            top_p = self.model.top_p if hasattr(self.model, 'top_p') else 0.95
        if max_tokens is None:
            max_tokens = self.model.max_tokens if hasattr(self.model, 'max_tokens') else 4096
        
        # Step 1: Generate with generate_model (using PyTorch)
        generated_text = LossCalculator.generate(
            model_path=generate_model_path,
            messages=processed_input,
            device=generate_device,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        # Step 2: Calculate loss with loss_model
        loss = LossCalculator.calculate_loss(
            model_path=loss_model_path,
            messages=processed_input,
            target=generated_text,
            device=loss_device
        )
        
        result = {
            'text': generated_text,
            'loss': loss
        }
        
        # Step 3: (Optional) Calculate reverse loss
        if calculate_reverse:
            # Generate with loss_model
            reverse_text = LossCalculator.generate(
                model_path=loss_model_path,
                messages=processed_input,
                device=device,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            
            # Calculate loss with generate_model
            reverse_loss = LossCalculator.calculate_loss(
                model_path=generate_model_path,
                messages=processed_input,
                target=reverse_text,
                device=device
            )
            
            result['reverse_text'] = reverse_text
            result['reverse_loss'] = reverse_loss
        
        return result


    def run_single_pass(self, data_row: dict):
        pass

    def run(self, record_full_result, start_idx=None, end_idx=None):
        # self.data.data.reverse()
        
        # Support dataset subset via slicing
        if start_idx is not None or end_idx is not None:
            data_subset = self.data[start_idx:end_idx]
            if self.verbose >= VERBOSE_MINIMAL:
                print(f"Running on dataset subset: [{start_idx}:{end_idx}] ({len(data_subset)} items)")
        else:
            data_subset = self.data
        
        num_items = len(data_subset)
        num_success = 0

        for i, data_row in enumerate(data_subset):
            if self.verbose >= VERBOSE_FULL:
                print("", flush=True, end="")

            found = False
            for j in range(len(self.results)):
                if self.results[j]["task_id"] == data_row[self.data.id_key]:
                    item = copy.deepcopy(self.results[j])
                    cur_pass = len(item["source_codes"])
                    is_solved = item["is_solved"]
                    cur_imp = item["source_codes"][-1]
                    found = True
                    break
            if not found:
                item = {
                    self.data.id_key: data_row[self.data.id_key],
                    "task_id": data_row[self.data.id_key],
                    "language": self.language,
                    "source_codes": [],
                    "run_details": [],
                    "no_of_try": 0,
                }

                cur_pass = 0
                is_solved = False
                cur_imp = ""

            while cur_pass < self.pass_at_k and not is_solved:
                # initialize it for each run
                self.run_details = {}
                # for _ in range(10):
                #     try:
                response = self.run_single_pass(data_row)
                #     break
                # except Exception as e:
                #     time.sleep(5)
                #     pass

                cur_imp = parse_response(response)

                item["source_codes"].append(cur_imp)

                # Remove Full details
                if not record_full_result and "details" in self.run_details:
                    del self.run_details["details"]

                item["run_details"].append(self.run_details)
                
                item["no_of_try"] += 1

                is_solved = self.data.evaluate(
                    item=data_row,
                    cur_imp=cur_imp,
                    language=self.language
                )

                cur_pass += 1
            
            if is_solved:
                num_success += 1

            item["is_solved"] = is_solved

            self.results.get_results().insert(i, item)

            # Deleting duplicate results
            k = i + 1
            while True:
                # Termination condition
                if k >= len(self.results):
                    break
                
                # Deleting duplicate results
                if self.results[k]["task_id"] == data_row[self.data.id_key]:
                    del self.results.results[k]
                
                # Increment
                k += 1

            if self.verbose >= VERBOSE_MINIMAL:
                print(f'completed {i+1}/{num_items}, Solved: {self.results[i]["is_solved"]}, number of success = {num_success}/{i+1}, acc = {round(num_success/(i+1)*100, 2)}')
            
            if not found:
                self.results.save_results()

            if self.verbose >= VERBOSE_FULL:
                print("", flush=True, end="")
          
        
        if len(self.results) > len(self.data):
            self.results.results = self.results[:len(self.data)]
            self.results.save_results()
