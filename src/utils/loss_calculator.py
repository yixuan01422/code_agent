"""
Utility for calculating cross-entropy loss using HuggingFace models.
Models are cached in memory for efficiency.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class LossCalculator:
    """Calculate cross-entropy loss for language model outputs."""
    
    # Class-level cache for loaded models
    _model_cache = {}
    
    @classmethod
    def _get_model(cls, model_path, device='cuda'):
        """
        Load or retrieve cached model and tokenizer.
        
        Args:
            model_path (str): Path to HuggingFace model
            device (str): Device to load model on (e.g., 'cuda:0', 'cuda:1')
        
        Returns:
            tuple: (model, tokenizer)
        """
        # Cache key includes device to support same model on different GPUs
        cache_key = f"{model_path}@{device}"
        
        if cache_key not in cls._model_cache:
            print(f"📦 Loading model: {model_path} → {device}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            cls._model_cache[cache_key] = (model, tokenizer)
            print(f"✅ Model cached: {cache_key}")
        
        return cls._model_cache[cache_key]
    
    @classmethod
    def _messages_to_prompt(cls, tokenizer, messages):
        """
        Convert OpenAI format messages to prompt string using chat template.
        
        Args:
            tokenizer: HuggingFace tokenizer
            messages (list): Chat messages [{"role": "user", "content": "..."}]
        
        Returns:
            str: Formatted prompt
        """
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            # Use official chat template (for Qwen, Llama, etc.)
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        return prompt
    
    @classmethod
    def generate(cls, model_path, messages, device='cuda', temperature=0.0, top_p=0.95, max_tokens=4096):
        """
        Generate text using PyTorch model (without vLLM).
        
        Args:
            model_path (str): Path to HuggingFace model
            messages (list): Chat messages in OpenAI format [{"role": "user", "content": "..."}]
            device (str): 'cuda' or 'cpu'
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            max_tokens (int): Maximum tokens to generate
        
        Returns:
            str: Generated text
        """
        try:
            # Load model and tokenizer
            model, tokenizer = cls._get_model(model_path, device)
            
            # Convert messages to prompt
            prompt = cls._messages_to_prompt(tokenizer, messages)
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs.input_ids.to(device)
            prompt_len = input_ids.shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,  # avoid 0
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
            
            # Decode only the newly generated part
            generated_ids = outputs[0][prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            print(f"⚠️  Text generation failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    @classmethod
    def calculate_loss(cls, model_path, messages, target, device='cuda'):
        """
        Calculate cross-entropy loss for target text given messages.
        Loss is computed only on the target portion (messages are masked).
        
        Args:
            model_path (str): Path to HuggingFace model
            messages (list): Chat messages in OpenAI format [{"role": "user", "content": "..."}]
            target (str): Generated text to calculate loss on
            device (str): 'cuda' or 'cpu'
        
        Returns:
            float: Cross-entropy loss value, or None if calculation fails
        """
        try:
            # Load model and tokenizer
            model, tokenizer = cls._get_model(model_path, device)
            
            # Convert messages to prompt
            prompt = cls._messages_to_prompt(tokenizer, messages)
            
            # Tokenize prompt and full text (prompt + target)
            full_text = prompt + target
            
            prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
            
            prompt_len = prompt_tokens.input_ids.shape[1]
            full_ids = full_tokens.input_ids.to(device)
            
            if full_ids.shape[1] <= prompt_len:
                print(f"⚠️  Warning: target is empty or too short, skipping loss calculation")
                return None
            
            # Forward pass to get logits
            with torch.no_grad():
                outputs = model(input_ids=full_ids)
                logits = outputs.logits  # [1, seq_len, vocab_size]
            
            # Calculate loss only on target portion
            # For predicting token at position i, we use logits at position i-1
            # So: logits[prompt_len-1:seq_len-1] predicts tokens[prompt_len:seq_len]
            target_logits = logits[0, prompt_len-1:-1, :]  # [target_len, vocab_size]
            target_ids = full_ids[0, prompt_len:]  # [target_len]
            
            # Calculate cross-entropy loss
            loss = F.cross_entropy(target_logits, target_ids, reduction='mean')
            
            return loss.item()
            
        except Exception as e:
            print(f"⚠️  Loss calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached models and free GPU memory."""
        print(f"🧹 Clearing {len(cls._model_cache)} cached models...")
        cls._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Cache cleared")

