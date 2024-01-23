from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast, GPTNeoXTokenizerFast
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
import torch
import datasets

from tuning.data import tokenizer_data_utils
from tuning.config import configs, peft_config
from tuning.utils.config_utils import get_hf_peft_config
from tuning.utils.data_type_utils import get_torch_dtype

from tuning.aim_loader import get_aimstack_callback
from transformers.utils import logging
from dataclasses import asdict
from typing import Optional, Union

from peft import LoraConfig
import os
from transformers import TrainerCallback
from peft.utils.other import fsdp_auto_wrap_policy

def check_input_encoding(tokenizer, prompt):
    print(prompt)
    inp = tokenizer.encode(prompt, return_tensors="pt")
    print(inp)
    print(tokenizer.decode(inp[0]))
    inp = tokenizer.encode(prompt)
    print(inp)

def generate(
        model_args: configs.ModelArguments,
        data_args: configs.DataArguments,
        train_args: configs.TrainingArguments,
        peft_config: Optional[Union[peft_config.LoraConfig, peft_config.PromptTuningConfig]] = None,
   ):
    run_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    logger = logging.get_logger("sft_trainer")

    # Validate parameters
    if (not isinstance(train_args.num_train_epochs, float)) or (train_args.num_train_epochs <= 0):
        raise ValueError("num_train_epochs has to be an integer/float >= 1")
    if (not isinstance(train_args.gradient_accumulation_steps , int)) or (train_args.gradient_accumulation_steps <= 0):
        raise ValueError("gradient_accumulation_steps has to be an integer >= 1")

    # make sure to unset FSDP args when running on single gpu
    if not run_distributed:
        train_args.fsdp = ""
        train_args.fsdp_config = {'xla':False}

    task_type = "CAUSAL_LM"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=train_args.cache_dir,
        torch_dtype=get_torch_dtype(model_args.torch_dtype),
        use_flash_attention_2=model_args.use_flash_attn,
    )
    
    peft_config = get_hf_peft_config(task_type, peft_config)

    model.gradient_checkpointing_enable()

    # TODO: Move these to a config as well
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=train_args.cache_dir,
        #padding_side="right",
        padding_side="left",
        use_fast = True
    )
    
    model_max_length = min(train_args.model_max_length, tokenizer.model_max_length)
    logger.info(f"Model max length {model_max_length}")
    if train_args.model_max_length > tokenizer.model_max_length:
        logger.warning(f"model_max_length {train_args.model_max_length} exceeds tokenizer.model_max_length {tokenizer.model_max_length}, using tokenizer.model_max_length {tokenizer.model_max_length}")
   
    print(tokenizer)
    print(tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id)
    if not tokenizer.pad_token: #for batching
        #tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
    print(tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id)
    
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    prompts = [
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat are the names of some famous actors that started their careers on Broadway?\n\n### Response:",
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is some cool music from the 1920s?\n\n### Response:",
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nI like to host guests at my home from time to time, and I am gathering  recipes of different dishes and drinks to keep things interesting. I am interested in trying some Somali  dishes. Can you give me a recipe for Canjeero?\n\n### Response:"
            ]
    for prompt in prompts:
        check_input_encoding(tokenizer, prompt)

    check_input_encoding(tokenizer, '\n\n###')
    check_input_encoding(tokenizer, 'Instruction:')

    #for batch size = 1 i.e. single prompt
    #prompt = "Hello, my name is"
    #inputs = tokenizer(prompt, return_tensors="pt")

    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    print(inputs)

    device = torch.device('cuda')
    model = model.to(device)
    inputs = inputs.to(device)

    #outputs = model.generate(**inputs, eos_token_id=tokenizer.encode('### Instruction:'), pad_token_id=tokenizer.pad_token_id, max_length=2048)

    #outputs = model.generate(**inputs, eos_token_id=tokenizer.encode('\n\n###'), pad_token_id=tokenizer.pad_token_id, max_length=2048)
    #outputs = model.generate(**inputs, eos_token_id=[1, 29871, 13, 13, 2277, 29937], pad_token_id=tokenizer.pad_token_id, max_length=2048)

    #outputs = model.generate(**inputs, eos_token_id=tokenizer.encode('Instruction:'), pad_token_id=tokenizer.pad_token_id, max_length=2048)
    #outputs = model.generate(**inputs, eos_token_id=[1, 2799, 4080, 29901], pad_token_id=tokenizer.pad_token_id, max_length=100)
    outputs = model.generate(**inputs, eos_token_id=[10548, 2705, 27], pad_token_id=tokenizer.pad_token_id, max_length=100)

    #outputs = model.generate(**inputs, eos_token_id=tokenizer.encode(['.\n', '. ']), pad_token_id=tokenizer.pad_token_id, max_length=208)
    #outputs = model.generate(**inputs, eos_token_id=tokenizer.encode(['.\n', '. ']), max_length=208)
    #outputs = model.generate(**inputs, max_length=208)

    decoded_outputs = tokenizer.batch_decode(outputs)
    print(decoded_outputs)
    print()
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(decoded_outputs)

def main(**kwargs):
    parser = transformers.HfArgumentParser(dataclass_types=(configs.ModelArguments, 
                                                            configs.DataArguments,
                                                            configs.TrainingArguments,
                                                            peft_config.LoraConfig,
                                                            peft_config.PromptTuningConfig))
    parser.add_argument('--peft_method', type=str.lower, choices=['pt', 'lora', None, 'none'], default=None)
    model_args, data_args, training_args, lora_config, prompt_tuning_config, peft_method, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if peft_method.peft_method =="lora":
        tune_config=lora_config
    elif peft_method.peft_method =="pt":
        tune_config=prompt_tuning_config
    else:
        tune_config=None
    generate(model_args, data_args, training_args, tune_config)

if __name__ == "__main__":
    main()
