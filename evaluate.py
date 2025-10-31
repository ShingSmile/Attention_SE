import re
import sys
import io, os
import torch
import numpy as np
import logging
import tqdm
import fcntl
import time
import argparse
import csv
from collections import defaultdict, OrderedDict
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from prettytable import PrettyTable
import transformers
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from senllm import LlamaForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM
from senllm.modeling_llama import (
    DEFAULT_ZERO_SPECIAL_SKIP_THRESHOLD,
    DEFAULT_ZERO_SPECIAL_TARGET_THRESHOLD,
)
from senllm.token_matching import TokenSequenceMatcher
from colorama import Fore, Style
import textwrap
from scipy.stats import spearmanr
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime


if torch.cuda.is_available():
    print("We are using GPU!")
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)

COEFF = float(os.environ.get("COEFF", 1.0))

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


RESULT_CSV_PATH = "evaluation_results.csv"
CSV_BASE_COLUMNS = [
    "timestamp",
    "config",
    "mode",
    "task_set",
    "model_name",
    "prompt_method",
    "enable_attention_override",
    "head_order",
    "override_mode",
    "top_k",
    "gamma",
]
CSV_TASK_COLUMNS = [
    "STSBenchmark-dev",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
    "SICKRelatedness",
    "STS Avg.",
    "MR",
    "CR",
    "SUBJ",
    "MPQA",
    "SST2",
    "TREC",
    "MRPC",
    "Transfer Avg.",
]
CSV_ALL_COLUMNS = CSV_BASE_COLUMNS + CSV_TASK_COLUMNS


def append_results_to_csv(base_info: Dict[str, str], task_scores: Dict[str, str], csv_path: str = RESULT_CSV_PATH) -> None:
    """Append evaluation summary to csv file."""
    if not task_scores:
        return
    row = OrderedDict((column, "") for column in CSV_ALL_COLUMNS)
    for key, value in base_info.items():
        if key in row:
            row[key] = value
    for task_name, score in task_scores.items():
        if task_name in row:
            row[task_name] = score
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_ALL_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def lock_and_write_file(file_path, content):
    with open(file_path, 'a') as file:
        while True:
            try:
                # Acquire an exclusive lock (non-blocking)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Perform your write operations here
                file.write(content + '\n')
                file.flush()

            except IOError as e:
                print("File is locked by another process. Can't write.")
                time.sleep(1)
            finally:
                # Release the lock
                fcntl.flock(file, fcntl.LOCK_UN)
                break

def load_config_from_yaml(config_file="config.yaml", config_name=None):

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"warning: config file {config_file} not found, using command line parameters")
        return None
    except yaml.YAMLError as e:
        print(f"error: config file {config_file} format error: {e}")
        return None
    
    if config_name is None:
        config_name = yaml_config.get('default_config', 'llama-2-7b')
    
    if config_name not in yaml_config.get('models', {}):
        available_configs = list(yaml_config.get('models', {}).keys())
        print(f"error: config '{config_name}' not found")
        print(f"available configs: {available_configs}")
        return None
    
    # 获取指定配置
    config = yaml_config['models'][config_name].copy()
    
    # 添加GPU配置
    if 'gpu_config' in yaml_config:
        config['gpu_config'] = yaml_config['gpu_config']
    
    print(f"✓ successfully loaded config: {config_name}")
    return config


def build_configuration_summary(args) -> str:
    key_width = 30

    def fmt_bool(value: Optional[bool]) -> str:
        if value is None:
            return "-"
        return "Yes" if value else "No"

    def fmt_list(value: Optional[List]) -> str:
        if not value:
            return "-"
        if all(isinstance(item, (int, float, str)) for item in value) and len(value) <= 8:
            return ", ".join(map(str, value))
        return str(value)

    def fmt_value(value) -> str:
        if value is None or value == "":
            return "-"
        if isinstance(value, bool):
            return fmt_bool(value)
        if isinstance(value, (list, tuple)):
            return fmt_list(list(value))
        if isinstance(value, Mapping):
            if not value:
                return "-"
            items = []
            for k, v in value.items():
                if isinstance(v, bool):
                    items.append(f"{k}={'on' if v else 'off'}")
                else:
                    items.append(f"{k}={v}")
            return ", ".join(items)
        return str(value)

    def make_line(label: str, value) -> str:
        return f"{Fore.GREEN}{label.ljust(key_width)}:{Style.RESET_ALL} {fmt_value(value)}"

    model_path = args.model_name_or_path or "-"
    model_name = os.path.basename(model_path.rstrip("/")) if model_path and "/" in model_path else model_path
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    lines = [
        f"{Fore.CYAN}Configuration:{Style.RESET_ALL}",
        f"{Fore.YELLOW}{'-' * key_width}{Style.RESET_ALL}",
        make_line("Config Name", getattr(args, "config", None)),
        make_line("Config File", getattr(args, "config_file", None)),
        make_line("Backbone", model_name),
        make_line("Model Path", model_path),
        make_line("Tokenizer", args.tokenizer_name or "auto"),
        make_line("Evaluation Mode", args.mode),
        make_line("Task Set", args.task_set),
        make_line("Prompt Method", args.prompt_method),
        make_line("Batch Size", args.batch_size),
        make_line("Tensor Parallel", fmt_bool(args.tensor_parallel)),
        make_line("Plan", args.use_which_plan),
        make_line("Output Layer Index", args.output_layer),
        make_line("TP Starting Index", args.tp_starting_index),
        make_line("TP Exiting Index", args.tp_exiting_index),
        make_line("COEFF", COEFF),
    ]
    if cuda_visible_devices:
        lines.append(make_line("CUDA_VISIBLE_DEVICES", cuda_visible_devices))

    attn_cfg = args.attention_enhance or {}
    if attn_cfg:
        lines.extend([
            "",
            f"{Fore.MAGENTA}Attention Enhance:{Style.RESET_ALL}",
            f"{Fore.YELLOW}{'-' * key_width}{Style.RESET_ALL}",
            make_line("Enabled", attn_cfg.get("enabled", False)),
            make_line("Enable Override", attn_cfg.get("enable_attention_override")),
            make_line("Head Order", attn_cfg.get("head_order")),
            make_line("Override Mode", attn_cfg.get("override_mode") or attn_cfg.get("mode")),
            make_line("Score File", attn_cfg.get("score_file")),
            make_line("Top K", attn_cfg.get("top_k")),
            make_line("Gamma", attn_cfg.get("gamma")),
            make_line("Target Phrase", attn_cfg.get("target_phrase")),
            make_line("Target Token IDs", attn_cfg.get("target_token_ids")),
            make_line("Special Token IDs", attn_cfg.get("special_token_ids")),
            make_line("Analysis Samples", attn_cfg.get("analysis_samples")),
            make_line("Analysis Dir", attn_cfg.get("analysis_dir")),
            make_line("Analysis Tasks", attn_cfg.get("analysis_tasks")),
            make_line("Zero Skip Threshold", attn_cfg.get("zero_special_skip_threshold")),
            make_line("Zero Target Threshold", attn_cfg.get("zero_special_target_threshold")),
        ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None,
                        help="config name, read parameters from config.yaml")
    parser.add_argument("--config_file", type=str, default="config.yaml",
                        help="config file path")
    

    parser.add_argument("--tokenizer_name", type=str,
                        default='')
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na', 'stsb'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument('--tensor_parallel', action='store_true')
    parser.add_argument('--prompt_method', type=str, 
                        default='prompteol', choices=['prompteol', 'metaeol', 'cot', 'ke'], help="What prompt method to use.")
    parser.add_argument("--use_which_plan", type=str,
                        choices=['tp', 'vanilla'],
                        default='tp')
    parser.add_argument("--output_layer", type=int, 
                        default=-1)
    parser.add_argument("--tp_starting_index", type=int, 
                        default=1)
    parser.add_argument("--tp_exiting_index", type=int, 
                        default=99)
    parser.add_argument("--batch_size", type=int, 
                        default=16)

    args = parser.parse_args()
    args.attention_enhance = None
    
    if args.config:
        config = load_config_from_yaml(args.config_file, args.config)
        if config is None:
            print("config loading failed, exit program")
            sys.exit(1)
        
        args.model_name_or_path = config.get('model_name_or_path', args.model_name_or_path)
        args.use_which_plan = config.get('use_which_plan', args.use_which_plan)
        args.output_layer = config.get('output_layer', args.output_layer)
        args.tp_starting_index = config.get('tp_starting_index', args.tp_starting_index)
        args.tp_exiting_index = config.get('tp_exiting_index', args.tp_exiting_index)
        args.batch_size = config.get('batch_size', args.batch_size)
        args.mode = config.get('mode', args.mode)
        args.task_set = config.get('task_set', args.task_set)
        args.prompt_method = config.get('prompt_method', args.prompt_method)
        args.attention_enhance = config.get('attention_enhance', args.attention_enhance)
        
        if 'gpu_config' in config and 'cuda_visible_devices' in config['gpu_config']:
            os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_config']['cuda_visible_devices']
            print(f"✓ set GPU devices: {config['gpu_config']['cuda_visible_devices']}")
    
    if not args.model_name_or_path:
        print("error: model path not specified, please use --model_name_or_path parameter or specify it in the config file")
        sys.exit(1)
    model_name_lower = args.model_name_or_path.lower()

    if args.tensor_parallel:
        import tensor_parallel as tp
        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
    else:
        if 'llama' in model_name_lower:
            model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,
                                                        device_map='auto',
                                                        output_hidden_states=True,
                                                        trust_remote_code=True)
            model.model.plan = args.use_which_plan
            model.model.tp_starting_index = args.tp_starting_index
            model.model.tp_exiting_index = args.tp_exiting_index
            if hasattr(model.model, "summary_layer_index"):
                summary_layer_index = args.output_layer
                if summary_layer_index < 0:
                    summary_layer_index = len(model.model.layers) + summary_layer_index
                model.model.summary_layer_index = summary_layer_index
        elif 'qwen2' in model_name_lower:
            model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path,
                                                        device_map='auto',
                                                        output_hidden_states=True,
                                                        trust_remote_code=True)
            model.model.plan = args.use_which_plan
            model.model.tp_starting_index = args.tp_starting_index
            model.model.tp_exiting_index = args.tp_exiting_index
        elif 'gemma' in model_name_lower:
            model = Gemma2ForCausalLM.from_pretrained(args.model_name_or_path,
                                                        device_map='auto',
                                                        output_hidden_states=True,
                                                        trust_remote_code=True)
            model.model.plan = args.use_which_plan
            model.model.tp_starting_index = args.tp_starting_index
            model.model.tp_exiting_index = args.tp_exiting_index
        else:
            raise ValueError(f"Cannot find such {args.model_name_or_path.lower()} model!")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0  # Set the padding token. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    special_token_ids = set(tokenizer.all_special_ids or [])

    for attr in ["pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"]:
        value = getattr(tokenizer, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                if item is not None:
                    special_token_ids.add(int(item))
        else:
            special_token_ids.add(int(value))
    if args.attention_enhance:
        args.attention_enhance["special_token_ids"] = sorted(int(tid) for tid in special_token_ids)
    #     保留特殊 token
        args.attention_enhance["special_token_ids"] = []


    if args.use_which_plan == 'tp':
        placeholder_token = '<PST>'
        tokenizer.add_tokens([placeholder_token])
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        
        model.resize_token_embeddings(len(tokenizer))

        embedding_layer = model.get_input_embeddings()
        embedding_layer.weight.requires_grad_(False)
        
        num_dim = embedding_layer.weight.shape[1]
        device = embedding_layer.weight.device
        
        with torch.no_grad():
            embedding_layer.weight[placeholder_token_id] = torch.randn(num_dim, device=device)
        embedding_layer.weight.requires_grad_(True)

    if args.attention_enhance and args.attention_enhance.get("enabled", False):
        target_phrase = args.attention_enhance.get("target_phrase")
        matcher: Optional[TokenSequenceMatcher] = None
        analysis_dir = args.attention_enhance.get("analysis_dir", "attention_analysis")
        args.attention_enhance["analysis_dir"] = analysis_dir

        if target_phrase:
            phrases = target_phrase if isinstance(target_phrase, (list, tuple)) else [target_phrase]
            matcher = TokenSequenceMatcher.from_phrases(
                phrases,
                tokenizer,
                include_leading_space_variant=True,
            )
            if matcher.is_empty():
                logging.warning(
                    "[attention_enhance] target_phrase=%r 未能转换为有效的 token 序列。",
                    target_phrase,
                )
            else:
                logging.info(
                    "[attention_enhance] target_phrase=%r -> token_ids=%s",
                    target_phrase,
                    matcher.sequences,
                )
        elif "target_token_ids" in args.attention_enhance:
            existing = args.attention_enhance.get("target_token_ids", [])
            matcher = TokenSequenceMatcher(
                existing if isinstance(existing, (list, tuple)) else [existing]
            )
            if matcher.is_empty():
                matcher = None
            else:
                logging.info(
                    "[attention_enhance] 使用提供的 target_token_ids=%s",
                    matcher.sequences,
                )
        else:
            logging.warning("[attention_enhance] 未提供 target_phrase 或 target_token_ids，将仅放大最后 token。")

        if matcher and not matcher.is_empty():
            sequences = matcher.sequences
            args.attention_enhance["target_token_ids"] = sequences
            logging.info(
                "[attention_enhance] 最终用于匹配的 token 序列=%s",
                sequences,
            )
        else:
            args.attention_enhance.pop("target_token_ids", None)
            logging.warning("[attention_enhance] token 序列为空，将退化为放大最后 token。")

        args.attention_enhance.setdefault("enable_attention_override", True)
        args.attention_enhance.setdefault("head_order", "score")
        override_mode_raw = args.attention_enhance.get(
            "override_mode",
            args.attention_enhance.get("mode", "scale_max"),
        )
        override_mode = str(override_mode_raw or "").strip().lower()
        if override_mode not in {"scale_max", "zero_special"}:
            logging.warning(
                "[attention_enhance] override_mode=%r 不受支持，将回退为 scale_max。",
                override_mode_raw,
            )
            override_mode = "scale_max"
        args.attention_enhance["override_mode"] = override_mode

        if override_mode == "zero_special":
            skip_raw = args.attention_enhance.get("zero_special_skip_threshold")
            target_raw = args.attention_enhance.get("zero_special_target_threshold")
            skip_threshold = DEFAULT_ZERO_SPECIAL_SKIP_THRESHOLD
            target_threshold = DEFAULT_ZERO_SPECIAL_TARGET_THRESHOLD
            if skip_raw is not None:
                try:
                    skip_threshold = float(skip_raw)
                except (TypeError, ValueError):
                    logging.warning(
                        "[attention_enhance] zero_special_skip_threshold=%r 解析失败，使用默认值 %.3f。",
                        skip_raw,
                        DEFAULT_ZERO_SPECIAL_SKIP_THRESHOLD,
                    )
            if target_raw is not None:
                try:
                    target_threshold = float(target_raw)
                except (TypeError, ValueError):
                    logging.warning(
                        "[attention_enhance] zero_special_target_threshold=%r 解析失败，使用默认值 %.3f。",
                        target_raw,
                        DEFAULT_ZERO_SPECIAL_TARGET_THRESHOLD,
                    )
            args.attention_enhance["zero_special_skip_threshold"] = skip_threshold
            args.attention_enhance["zero_special_target_threshold"] = target_threshold

    print(build_configuration_summary(args))

    if (not args.tensor_parallel) and 'llama' in model_name_lower:
        model.model.configure_attention_enhance(args.attention_enhance)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        if args.mode == 'dev':
            args.tasks = ['STSBenchmark-dev']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'stsb':
        args.tasks = ['STSBenchmark']
    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 32}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size':args.batch_size}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    if args.prompt_method == "metaeol":
        if args.use_which_plan == 'tp':
            task_prompts = ["In this task, you're presented with a text excerpt. Your task is to categorize the excerpt into a broad category such as 'Education', 'Technology', 'Health', 'Business', 'Environment', 'Politics', or 'Culture'. These categories help in organizing content for better accessibility and targeting. For this task, this sentence : <PST> \"*sent 0*\" should be classified under one general category in one word:\"",
                            "In this task, you're given a statement and you need to determine whether it's presenting an 'Opinion' or a 'Fact'. This distinction is vital for information verification, educational purposes, and content analysis. For this task, this sentence : <PST> \"*sent 0*\" discriminates between opinion and fact in one word:\"",
                            "In this task, you're given a review from an online platform. Your task is to generate a rating for the product based on the review on a scale of 1-5, where 1 means 'extremely negative' and 5 means 'extremely positive'. For this task, this sentence : <PST> \"*sent 0*\" reflects the sentiment in one word:\"",
                            "In this task, you're reading a personal diary entry. Your task is to identify the predominant emotion expressed, such as joy, sadness, anger, fear, or love. For this task, this sentence : <PST> \"*sent 0*\" conveys the emotion in one word:\"",
                            "In this task, you're presented with two sentences. Your task is to assess whether the sentences convey the same meaning. Use 'identical', 'similar', 'different', or 'unrelated' to describe the relationship. To enhance the performance of this task, this sentence : <PST> \"*sent 0*\" means in one word:\"",
                            "In this task, you're given a sentence and a phrase. Your task is to determine if the phrase can be a contextual synonym within the given sentence. Options include 'yes', 'no', or 'partially'. To enhance the performance of this task, this sentence : <PST> \"*sent 0*\" means in one word:\"",
                            "In this task, you're examining a news article. Your task is to extract the most critical fact from the article. For this task, this sentence : <PST> \"*sent 0*\" encapsulates the key fact in one word:\"",
                            "In this task, you're reviewing a scientific abstract. Your task is to identify the main entities (e.g., proteins, diseases) and their relations (e.g., causes, treats). For this task, this sentence : <PST> \"*sent 0*\" highlights the primary entity or relation in one word:\"",
                            ]
        else:
            task_prompts = ["In this task, you're presented with a text excerpt. Your task is to categorize the excerpt into a broad category such as 'Education', 'Technology', 'Health', 'Business', 'Environment', 'Politics', or 'Culture'. These categories help in organizing content for better accessibility and targeting. For this task, this sentence : \"*sent 0*\" should be classified under one general category in one word:\"",
                            "In this task, you're given a statement and you need to determine whether it's presenting an 'Opinion' or a 'Fact'. This distinction is vital for information verification, educational purposes, and content analysis. For this task, this sentence : \"*sent 0*\" discriminates between opinion and fact in one word:\"",
                            "In this task, you're given a review from an online platform. Your task is to generate a rating for the product based on the review on a scale of 1-5, where 1 means 'extremely negative' and 5 means 'extremely positive'. For this task, this sentence : \"*sent 0*\" reflects the sentiment in one word:\"",
                            "In this task, you're reading a personal diary entry. Your task is to identify the predominant emotion expressed, such as joy, sadness, anger, fear, or love. For this task, this sentence : \"*sent 0*\" conveys the emotion in one word:\"",
                            "In this task, you're presented with two sentences. Your task is to assess whether the sentences convey the same meaning. Use 'identical', 'similar', 'different', or 'unrelated' to describe the relationship. To enhance the performance of this task, this sentence : \"*sent 0*\" means in one word:\"",
                            "In this task, you're given a sentence and a phrase. Your task is to determine if the phrase can be a contextual synonym within the given sentence. Options include 'yes', 'no', or 'partially'. To enhance the performance of this task, this sentence : \"*sent 0*\" means in one word:\"",
                            "In this task, you're examining a news article. Your task is to extract the most critical fact from the article. For this task, this sentence : \"*sent 0*\" encapsulates the key fact in one word:\"",
                            "In this task, you're reviewing a scientific abstract. Your task is to identify the main entities (e.g., proteins, diseases) and their relations (e.g., causes, treats). For this task, this sentence : \"*sent 0*\" highlights the primary entity or relation in one word:\"",
                            ]
    elif args.prompt_method == "prompteol":
        if args.use_which_plan == 'tp':
            task_prompts = ['This sentence : <PST> \"*sent 0*\" means in one word:\"']
        else:
            task_prompts = ["This sentence : \"*sent 0*\" means in one word:\""]
    elif args.prompt_method == "cot":
        if args.use_which_plan == 'tp':
            task_prompts = ['After thinking step by step , this sentence : <PST> \"*sent 0*\" means in one word:\"']
        else:
            task_prompts = ['After thinking step by step , this sentence : \"*sent 0*\" means in one word:\"']
    elif args.prompt_method == "ke":
        if args.use_which_plan == 'tp':
            task_prompts = ['The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : <PST> \"*sent 0*\" means in one word:\"']
        else:
            task_prompts = ['The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : \"*sent 0*\" means in one word:\"']

    print(task_prompts)

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        input_sentences = [' '.join(s) for s in batch]
        if max_length == 500:
            sentences = [tokenizer.decode(tokenizer.encode(s, add_special_tokens=False)[:max_length]) for s in sentences]
            max_length = 512

        new_sentences = []
        injected_texts: List[str] = []
        for i, s in enumerate(sentences):
            if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
            s = s.replace('"', '\'')
            if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
            for prompt in task_prompts:
                new_sentences.append(prompt.replace('*sent 0*', s).strip())
                injected_texts.append(s)
        sentences = new_sentences

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=max_length is not None
        )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None
        # Get raw embeddings
        if hasattr(model, "model") and hasattr(model.model, "set_attention_analysis_texts"):
            model.model.set_attention_analysis_texts(injected_texts, tokenizer=tokenizer)
        with torch.no_grad():
            raw_outputs = model(
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
                **batch,
            )
            hidden_states = raw_outputs.hidden_states
            hidden_states_layer = hidden_states[args.output_layer]
            base_hidden_tokens = hidden_states_layer[:, -1, :]

            if base_hidden_tokens.dtype == torch.bfloat16:
                base_hidden_tokens = base_hidden_tokens.float()
                hidden_states_layer = hidden_states_layer.float()

            num_prompts = len(task_prompts)
            if base_hidden_tokens.size(0) % num_prompts != 0:
                raise RuntimeError(
                    f"Batch size {base_hidden_tokens.size(0)} not divisible by number of prompts {num_prompts}."
                )
            num_samples = base_hidden_tokens.size(0) // num_prompts

            device = base_hidden_tokens.device
            dtype = base_hidden_tokens.dtype
            enhance_cfg = args.attention_enhance or {}
            analysis_enabled = bool(enhance_cfg.get("enabled", False))
            override_enabled = bool(enhance_cfg.get("enable_attention_override", True))
            attention_enabled = analysis_enabled and override_enabled

            enhanced_embeddings: Dict[int, List[torch.Tensor]] = defaultdict(list)
            base_hidden_mean = base_hidden_tokens.view(num_samples, num_prompts, -1).mean(dim=1)
            analysis_records = []
            if hasattr(model, "model") and hasattr(model.model, "pop_attention_analysis_records"):
                analysis_records = model.model.pop_attention_analysis_records()
            if attention_enabled and not analysis_records:
                logging.warning(
                    "[attention_analysis] attention enhance enabled but no analysis records returned; heads=%s",
                    getattr(model.model, "_attention_enhance_heads_by_layer", {}),
                )
            analysis_tasks_config: Dict[str, object] = {}
            run_heatmap_csv_analysis = True
            heatmap_csv_base_dir = None
            if args.attention_enhance:
                raw_tasks_cfg = args.attention_enhance.get("analysis_tasks") or {}
                if isinstance(raw_tasks_cfg, Mapping):
                    analysis_tasks_config = dict(raw_tasks_cfg)
                run_heatmap_csv_analysis = bool(analysis_tasks_config.get("enable_heatmap_csv", True))
                base_analysis_dir = args.attention_enhance.get("analysis_dir")
                if base_analysis_dir and run_heatmap_csv_analysis:
                    prompt_method_value = getattr(args, "prompt_method", None)
                    prompt_slug_chars: List[str] = []
                    if isinstance(prompt_method_value, str):
                        for ch in prompt_method_value.strip().lower():
                            if ch.isalnum():
                                prompt_slug_chars.append(ch)
                            elif ch in {" ", "-", "_"}:
                                prompt_slug_chars.append("_")
                    prompt_slug = ''.join(prompt_slug_chars).strip('_') or "prompt"
                    heatmap_csv_base_dir = os.path.join(base_analysis_dir, prompt_slug, "heatmap_csv_analysis")
                    os.makedirs(heatmap_csv_base_dir, exist_ok=True)
            if analysis_records:
                special_token_ids = set(tokenizer.all_special_ids or [])
                for attr in ["pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"]:
                    value = getattr(tokenizer, attr, None)
                    if value is not None:
                        special_token_ids.add(int(value))
    
                analysis_visual_limit = int(args.attention_enhance.get("analysis_samples", 0) or 0) if args.attention_enhance else 0
    
                def decode_token(token_id: int) -> str:
                    if token_id in special_token_ids:
                        return ""
                    if token_id is None:
                        return "<unk>"
                    token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                    return token_text if token_text else repr(token_id)
    
                records_by_sample: Dict[int, List[Dict]] = defaultdict(list)
                for record in analysis_records:
                    sample_idx = int(record.get("sample_index", -1))
                    records_by_sample[sample_idx].append(record)
    
                for sample_idx in sorted(records_by_sample.keys()):
                    sample_records = sorted(records_by_sample[sample_idx], key=lambda r: r.get("layer", 0))
                    if not sample_records:
                        continue
                    if analysis_visual_limit > 0:
                        should_visualize = any(record.get("visualize", False) for record in sample_records)
                    else:
                        should_visualize = True
                    reference = sample_records[0]
                    token_ids = reference.get("token_ids", [])
                    sample_text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
                    sample_text = sample_text.replace("\n", "\\n")
                    if should_visualize:
                        logging.info(
                            "[attention_analysis][sample %s] prompt=%r",
                            sample_idx,
                            sample_text,
                        )
                    query_positions = reference.get("query_positions", [])
                    query_tokens = [
                        decode_token(token_ids[pos]) if pos < len(token_ids) else f"<idx {pos}>"
                        for pos in query_positions
                    ]
                    if should_visualize:
                        logging.info(
                            "[attention_analysis][sample %s] query_tokens=%s",
                            sample_idx,
                            query_tokens,
                        )
    
                    query_labels = []
                    for idx, pos in enumerate(query_positions):
                        token_str = query_tokens[idx] if idx < len(query_tokens) else ""
                        if not token_str:
                            token_str = f"Q@{pos}"
                        query_labels.append(token_str)
    
                    input_text = reference.get("input_text") or ""
                    text_token_indices: List[int] = []
                    if input_text:
                        text_matcher = TokenSequenceMatcher.from_phrases(
                            [input_text],
                            tokenizer,
                            include_leading_space_variant=True,
                            prefixes=[" ", '"', ' "'],
                            suffixes=['"', '" '],
                        )
                        match_spans = text_matcher.find_matches_in_ids(token_ids)
                        if not match_spans:
                            decoded_prompt = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
                            if should_visualize:
                                logging.info(
                                    "[attention_analysis][sample %s] 未匹配输入文本 token: text=%r sequences=%s prompt_head=%r",
                                    sample_idx,
                                    input_text,
                                    text_matcher.sequences,
                                    decoded_prompt[:200],
                                )
                                logging.info(
                                    "[attention_analysis][sample %s] token_id_tail=%s",
                                    sample_idx,
                                    token_ids[max(len(token_ids)-20, 0):],
                                )
                        for start, end in match_spans:
                            text_token_indices.extend(range(start, end))
                    text_token_indices = sorted(set(text_token_indices))
                    # if not text_token_indices:
                    #     if should_visualize:
                    #         logging.warning(
                    #             "[attention_analysis][sample %s] 未找到输入文本对应的 token，跳过可视化。",
                    #             sample_idx,
                    #         )
                    #     continue
    
                    # valid_indices = [
                    #     idx
                    #     for idx in text_token_indices
                    #     if idx < len(token_ids) and token_ids[idx] not in special_token_ids
                    # ]
                    valid_indices = [
                        idx
                        for idx, tid in enumerate(token_ids)
                        if tid not in special_token_ids
                    ]
                    if not valid_indices:
                        if should_visualize:
                            logging.warning(
                                "[attention_analysis][sample %s] 输入文本 tokens 全为特殊符号，跳过可视化。",
                                sample_idx,
                            )
                            continue
    
                    def tidy_label(text: str) -> str:
                        text = text.replace("\n", " ").strip()
                        if not text:
                            return "<blank>"
                        return text[:10] + "…" if len(text) > 10 else text
    
                    index_to_label = {
                        idx: tidy_label(
                            decode_token(token_ids[idx]) if idx < len(token_ids) else f"<idx {idx}>"
                        )
                        for idx in valid_indices
                    }
                    all_indices = list(range(len(token_ids)))
                    full_index_to_label_list = [
                        tidy_label(
                            decode_token(token_ids[idx]) if idx < len(token_ids) else f"<idx {idx}>"
                        )
                        for idx in all_indices
                    ]
    
                    head_pairs = set()
                    for rec in sample_records:
                        layer_val = rec.get("layer")
                        for head_val in rec.get("heads", []):
                            head_pairs.add((layer_val, head_val))
                    head_count = len(head_pairs)
    
                    phrase_value = None
                    if args.attention_enhance:
                        phrase_value = args.attention_enhance.get("target_phrase")
                    if isinstance(phrase_value, (list, tuple)):
                        phrase_value = phrase_value[0] if phrase_value else None
                    if not phrase_value:
                        phrase_value = "target"
    
                    def slugify(text: str) -> str:
                        text = text.lower()
                        slug_chars = []
                        for ch in text:
                            if ch.isalnum():
                                slug_chars.append(ch)
                            elif ch in {" ", "-", "_"}:
                                slug_chars.append("_")
                        slug = ''.join(slug_chars).strip('_')
                        return slug or "phrase"
    
                    sample_dir = None
                    if heatmap_csv_base_dir and should_visualize:
                        phrase_slug = slugify(phrase_value)
                        sample_dir = os.path.join(
                            heatmap_csv_base_dir,
                            f"{phrase_slug}_heads{head_count}"
                        )
                        os.makedirs(sample_dir, exist_ok=True)

                    for record in sample_records:
                        layer = record.get("layer")
                        heads = record.get("heads", [])
                        query_scores_list = record.get("query_scores", [])

                        head_count = max(1, int(record.get("head_count", len(heads) or 1)))
                        if query_scores_list and token_ids:
                            for q_idx, scores_list in enumerate(query_scores_list):
                                scores_arr = np.array(scores_list, dtype=float)
                                filtered_pairs: List[Tuple[int, float]] = []
                                if valid_indices:
                                    valid_scores = scores_arr[valid_indices]
                                    order = np.argsort(valid_scores)[::-1][: min(3, len(valid_indices))]
                                    for order_idx in order:
                                        tgt_index = valid_indices[order_idx]
                                        filtered_pairs.append((tgt_index, valid_scores[order_idx]))
                                if not filtered_pairs:
                                    filtered_pairs = [
                                        (idx, scores_arr[idx])
                                        for idx in valid_indices
                                    ][: min(3, len(valid_indices))]
                                top_tokens = [
                                    (
                                        idx,
                                        decode_token(token_ids[idx]) if idx < len(token_ids) else f"<idx {idx}>",
                                        round(float(score), 6),
                                    )
                                    for idx, score in filtered_pairs
                                    if idx < len(token_ids)
                                ]
                                if not top_tokens:
                                    continue
                                if should_visualize and run_heatmap_csv_analysis:
                                    logging.info(
                                        "[attention_analysis][sample %s][layer %s heads %s][query %s] top_targets=%s",
                                        sample_idx,
                                        layer,
                                        heads,
                                        query_labels[q_idx] if q_idx < len(query_labels) else f"Q{q_idx}",
                                        top_tokens,
                                    )
                        else:
                            full_scores = record.get("full_scores", [])
                            filtered_pairs: List[Tuple[int, float]] = []
                            if full_scores and token_ids:
                                scores_arr = np.array(full_scores, dtype=float)
                                if valid_indices:
                                    valid_scores = scores_arr[valid_indices]
                                    order = np.argsort(valid_scores)[::-1][: min(3, len(valid_indices))]
                                    for order_idx in order:
                                        tgt_index = valid_indices[order_idx]
                                        filtered_pairs.append((tgt_index, valid_scores[order_idx]))
                            if not filtered_pairs:
                                top_indices = record.get("top_key_indices", [])
                                top_scores = record.get("top_scores", [])
                                filtered_pairs = [
                                    (idx, score) for idx, score in zip(top_indices, top_scores) if idx in valid_indices
                                ]
                            top_tokens = [
                                (
                                    idx,
                                    decode_token(token_ids[idx]) if idx < len(token_ids) else f"<idx {idx}>",
                                    round(float(score), 6),
                                )
                                for idx, score in filtered_pairs
                                if idx < len(token_ids)
                            ]
                            if not top_tokens:
                                continue
                            if run_heatmap_csv_analysis:
                                logging.info(
                                    "[attention_analysis][sample %s][layer %s heads %s] top_targets=%s",
                                    sample_idx,
                                    layer,
                                    heads,
                                    top_tokens,
                                )

                    target_dir = sample_dir
                    if should_visualize and target_dir:
                        layers = [int(rec.get("layer", 0)) for rec in sample_records]
                        unique_layers = sorted(set(layers))
                        if unique_layers:
                            seq_len = len(token_ids)
                            if not valid_indices:
                                valid_indices = [idx for idx, tid in enumerate(token_ids) if tid not in special_token_ids]
                                if not valid_indices:
                                    valid_indices = list(range(seq_len))
                            heatmap_rows = []
                            heatmap_labels = []
                            csv_heatmap_rows = []
                            csv_heatmap_labels = []
                            for layer in unique_layers:
                                layer_records = [rec for rec in sample_records if rec.get("layer") == layer]
                                if not layer_records:
                                    continue
                                record = layer_records[0]
                                query_scores_list = record.get("query_scores", [])
                                if query_scores_list:
                                    query_scores_array = np.array(query_scores_list, dtype=float)
                                    for q_idx, scores_arr in enumerate(query_scores_array):
                                        subset_scores = np.zeros(len(valid_indices), dtype=float)
                                        csv_scores = np.zeros(len(all_indices), dtype=float)
                                        for pos, token_index in enumerate(valid_indices):
                                            if token_index < scores_arr.shape[0]:
                                                subset_scores[pos] = scores_arr[token_index]
                                        for token_index in all_indices:
                                            if token_index < scores_arr.shape[0]:
                                                csv_scores[token_index] = scores_arr[token_index]
                                        heatmap_rows.append(subset_scores)
                                        csv_heatmap_rows.append(csv_scores)
                                        label = f"L{layer}-{query_labels[q_idx] if q_idx < len(query_labels) else f'Q{q_idx}'}"
                                        heatmap_labels.append(label)
                                        csv_heatmap_labels.append(label)
                                    total_scores = query_scores_array.sum(axis=0)
                                    num_queries_layer = max(1, query_scores_array.shape[0])
                                    total_scores = total_scores / num_queries_layer
                                    subset_scores = np.zeros(len(valid_indices), dtype=float)
                                    csv_scores = np.zeros(len(all_indices), dtype=float)
                                    for pos, token_index in enumerate(valid_indices):
                                        if token_index < total_scores.shape[0]:
                                            subset_scores[pos] = total_scores[token_index]
                                    for token_index in all_indices:
                                        if token_index < total_scores.shape[0]:
                                            csv_scores[token_index] = total_scores[token_index]
                                    heatmap_rows.append(subset_scores)
                                    heatmap_labels.append(f"L{layer}-mean")
                                    csv_heatmap_rows.append(csv_scores)
                                    csv_heatmap_labels.append(f"L{layer}-mean")
                                else:
                                    scores = record.get("full_scores", [])
                                    if scores:
                                        scores_arr = np.array(scores, dtype=float)
                                        subset_scores = np.zeros(len(valid_indices), dtype=float)
                                        csv_scores = np.zeros(len(all_indices), dtype=float)
                                        for pos, token_index in enumerate(valid_indices):
                                            if token_index < scores_arr.shape[0]:
                                                subset_scores[pos] = scores_arr[token_index]
                                        for token_index in all_indices:
                                            if token_index < scores_arr.shape[0]:
                                                csv_scores[token_index] = scores_arr[token_index]
                                        heatmap_rows.append(subset_scores)
                                        heatmap_labels.append(f"L{layer}")
                                        csv_heatmap_rows.append(csv_scores)
                                        csv_heatmap_labels.append(f"L{layer}")
                            if not heatmap_rows:
                                continue
                            heatmap = np.vstack(heatmap_rows)

                            if csv_heatmap_rows and sample_dir:
                                csv_path = os.path.join(sample_dir, f"sample_{sample_idx:03d}_attention.csv")
                                header = ["row_label"] + [
                                    f"{idx}:{label}" for idx, label in enumerate(full_index_to_label_list)
                                ]
                                with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                                    writer = csv.writer(csv_file)
                                    writer.writerow(header)
                                    for row_label, scores in zip(csv_heatmap_labels, csv_heatmap_rows):
                                        writer.writerow(
                                            [row_label]
                                            + [f"{float(val):.8f}" for val in scores]
                                        )
                                logging.info(
                                    "[attention_analysis][sample %s] heatmap values saved to %s",
                                    sample_idx,
                                    csv_path,
                                )

                            xtick_labels = [index_to_label[idx] for idx in valid_indices]
                            ytick_labels = heatmap_labels

                            fig, ax = plt.subplots(figsize=(max(6, len(valid_indices) * 0.35), max(2, len(heatmap_rows) * 0.6)))
                            im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
                            ax.set_xticks(range(len(valid_indices)))
                            ax.set_xticklabels(xtick_labels, rotation=90, fontsize=8)
                            ax.set_yticks(range(len(heatmap_rows)))
                            ax.set_yticklabels(ytick_labels, fontsize=9)
                            ax.set_xlabel("Token")
                            ax.set_ylabel("Layer")
                            ax.set_title(f"Sample {sample_idx} attention heatmap")
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            fig.tight_layout()
                            output_path = os.path.join(target_dir, f"sample_{sample_idx:03d}_attention.png")
                            fig.savefig(output_path)
                            plt.close(fig)
                            logging.info(
                                "[attention_analysis][sample %s] heatmap saved to %s",
                                sample_idx,
                                output_path,
                            )

                    aggregated_query_arrays = []
                    aggregated_head_counts = []
                    for rec in sample_records:
                        query_scores_list = rec.get("query_scores", [])
                        if query_scores_list:
                            aggregated_query_arrays.append(np.array(query_scores_list, dtype=float))
                            aggregated_head_counts.append(max(1, int(rec.get("head_count", len(rec.get("heads", [])) or 1))))
                    aggregated_matrix = None
                    if aggregated_query_arrays:
                        min_queries = min(arr.shape[0] for arr in aggregated_query_arrays)
                        min_len = min(arr.shape[1] for arr in aggregated_query_arrays)
                        if min_queries > 0 and min_len > 0:
                            stacked = np.stack(
                                [arr[:min_queries, :min_len] for arr in aggregated_query_arrays], axis=0
                            )
                            weights = np.array(aggregated_head_counts[: stacked.shape[0]], dtype=float)
                            weights = weights.reshape(-1, 1, 1)
                            weight_sum = weights.sum() if weights.size > 0 else 1.0
                            aggregated_matrix = (stacked * weights).sum(axis=0) / max(1.0, weight_sum)

                            if aggregated_matrix is not None:
                                if sample_dir:
                                    agg_mean_full = aggregated_matrix.mean(axis=0, keepdims=True)
                                    aggregated_for_csv = np.vstack([aggregated_matrix, agg_mean_full])
                                    agg_csv_header = ["row_label"] + [
                                        f"{idx}:{full_index_to_label_list[idx] if idx < len(full_index_to_label_list) else f'idx{idx}'}"
                                        for idx in range(aggregated_for_csv.shape[1])
                                    ]
                                    agg_row_labels_full = [
                                        query_labels[i] if i < len(query_labels) else f"Q{i}"
                                        for i in range(aggregated_matrix.shape[0])
                                    ] + ["mean"]
                                    agg_csv_path = os.path.join(
                                        sample_dir, f"sample_{sample_idx:03d}_attention_all_layers.csv"
                                    )
                                    with open(agg_csv_path, "w", newline="", encoding="utf-8") as csv_file:
                                        writer = csv.writer(csv_file)
                                        writer.writerow(agg_csv_header)
                                        for row_label, row_values in zip(agg_row_labels_full, aggregated_for_csv):
                                            writer.writerow(
                                                [row_label]
                                                + [f"{float(val):.8f}" for val in row_values]
                                            )
                                    logging.info(
                                        "[attention_analysis][sample %s] aggregated values saved to %s",
                                        sample_idx,
                                        agg_csv_path,
                                    )

                                filtered_valid_indices = [
                                    idx for idx in valid_indices if idx < aggregated_matrix.shape[1]
                                ]
                                if not filtered_valid_indices:
                                    filtered_valid_indices = list(
                                        range(min(aggregated_matrix.shape[1], len(valid_indices)))
                                    )
                                if not filtered_valid_indices:
                                    aggregated_matrix = None

                    summary_record = next(
                        (rec for rec in sample_records if rec.get("summary_hidden") is not None),
                        None,
                    )
                    aggregated_subset = None
                    if aggregated_matrix is not None:
                        filtered_valid_indices = [
                            idx for idx in filtered_valid_indices if idx < aggregated_matrix.shape[1]
                        ]
                        if filtered_valid_indices:
                            aggregated_subset = aggregated_matrix[:, filtered_valid_indices]
                            agg_labels = [
                                index_to_label.get(
                                    idx,
                                    tidy_label(
                                        decode_token(token_ids[idx]) if idx < len(token_ids) else f"<idx {idx}>"
                                    ),
                                )
                                for idx in filtered_valid_indices
                            ]
                            agg_row_labels = [
                                query_labels[i] if i < len(query_labels) else f"Q{i}"
                                for i in range(aggregated_subset.shape[0])
                            ] + ["mean"]
                            aggregated_mean = aggregated_subset.mean(axis=0, keepdims=True)
                            aggregated_subset = np.vstack([aggregated_subset, aggregated_mean])

                            if summary_record is not None:
                                summary_hidden = summary_record["summary_hidden"]
                                if isinstance(summary_hidden, torch.Tensor):
                                    summary_hidden_tensor = summary_hidden.to(device=device, dtype=dtype)
                                else:
                                    summary_hidden_tensor = torch.tensor(summary_hidden, dtype=dtype, device=device)
                                mean_weights_np = aggregated_subset[-1]
                                mean_weights = torch.from_numpy(mean_weights_np).to(device=device, dtype=dtype)
                                mean_weights = torch.clamp(mean_weights, min=0)
                                weight_sum = mean_weights.sum()
                                if weight_sum.item() <= 0:
                                    mean_weights = torch.ones_like(mean_weights) / mean_weights.numel()
                                else:
                                    mean_weights = mean_weights / weight_sum
                                summary_hidden_text = summary_hidden_tensor[filtered_valid_indices, :]
                                summary_vec = torch.matmul(mean_weights.unsqueeze(0), summary_hidden_text).squeeze(0)

                            if should_visualize and target_dir and aggregated_subset is not None:
                                fig2, ax2 = plt.subplots(
                                    figsize=(
                                        max(6, len(filtered_valid_indices) * 0.35),
                                        max(2, aggregated_subset.shape[0] * 0.6),
                                    )
                                )
                                im2 = ax2.imshow(aggregated_subset, aspect="auto", cmap="viridis")
                                ax2.set_xticks(range(len(filtered_valid_indices)))
                                ax2.set_xticklabels(agg_labels, rotation=90, fontsize=8)
                                ax2.set_yticks(range(aggregated_subset.shape[0]))
                                ax2.set_yticklabels(agg_row_labels, fontsize=9)
                                ax2.set_xlabel("Token")
                                ax2.set_ylabel("Query")
                                ax2.set_title(f"Sample {sample_idx} aggregated attention")
                                fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                                fig2.tight_layout()
                                agg_output_path = os.path.join(
                                    target_dir, f"sample_{sample_idx:03d}_attention_all_layers.png"
                                )
                                fig2.savefig(agg_output_path)
                                plt.close(fig2)
                                logging.info(
                                    "[attention_analysis][sample %s] aggregated heatmap saved to %s",
                                    sample_idx,
                                    agg_output_path,
                                )

                    if summary_record is not None:
                        batch_idx_local = int(summary_record.get("batch_idx", 0))
                        if 0 <= batch_idx_local < hidden_states_layer.size(0):
                            fused_vec = hidden_states_layer[batch_idx_local, -1, :].to(device=device, dtype=dtype)
                            orig_idx = batch_idx_local // num_prompts
                            if 0 <= orig_idx < num_samples:
                                enhanced_embeddings[orig_idx].append(fused_vec)

            if attention_enabled:
                for sample_idx in range(num_samples):
                    if not enhanced_embeddings.get(sample_idx):
                        fallback_vec = base_hidden_mean[sample_idx].to(device=device, dtype=dtype)
                        enhanced_embeddings[sample_idx].append(fallback_vec)
                missing = [idx for idx in range(num_samples) if not enhanced_embeddings.get(idx)]
                if missing:
                    logging.error(
                        "[attention_analysis] fused embeddings missing -> samples=%s available_keys=%s record_count=%s",
                        missing,
                        sorted(enhanced_embeddings.keys()),
                        len(analysis_records),
                    )
                    raise RuntimeError(
                        "Attention enhance enabled but fused embeddings missing for samples: {}".format(missing)
                    )
                sample_embeddings = []
                for sample_idx in range(num_samples):
                    vectors = enhanced_embeddings[sample_idx]
                    fused = torch.stack(vectors, dim=0).mean(dim=0)
                    # logging.debug(
                    #     "[attention_analysis] update sample %s fusedNorm=%.6f",
                    #     sample_idx,
                    #     float(fused.norm().item()),
                    # )
                    sample_embeddings.append(fused)
                outputs = torch.stack(sample_embeddings, dim=0)
            else:
                outputs = base_hidden_tokens.view(num_samples, num_prompts, -1).mean(dim=1)

            return outputs.to(dtype=torch.float32).cpu()

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    csv_metrics = OrderedDict()

    def collect_for_csv(task_names: List[str], scores: List[str], avg_label: Optional[str] = None) -> None:
        for name, score in zip(task_names, scores):
            column_name = name
            if name == "Avg." and avg_label:
                column_name = avg_label
            elif name == "Avg.":
                column_name = "Avg."
            csv_metrics[column_name] = score

    sts_scores: List[str] = []

    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark-dev']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)
        collect_for_csv(task_names, scores)
        sts_scores = list(scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        collect_for_csv(task_names, scores, avg_label="Transfer Avg.")


    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        collect_for_csv(task_names, scores, avg_label="STS Avg.")
        sts_scores = list(scores)
        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        collect_for_csv(task_names, scores, avg_label="Transfer Avg.")

    enhance_cfg = args.attention_enhance or {}

    skip_total = 0
    if hasattr(model, "model") and hasattr(model.model, "get_zero_special_skip_total"):
        skip_total = int(model.model.get_zero_special_skip_total())
        if skip_total:
            logging.info(
                "[attention_enhance][zero_special] total skipped boosts=%d",
                skip_total,
            )

    skip_threshold_val = enhance_cfg.get("zero_special_skip_threshold")
    target_threshold_val = enhance_cfg.get("zero_special_target_threshold")
    if skip_threshold_val is None and hasattr(model, "model"):
        skip_threshold_val = getattr(model.model, "_zero_special_skip_threshold", None)
    if target_threshold_val is None and hasattr(model, "model"):
        target_threshold_val = getattr(model.model, "_zero_special_target_threshold", None)

    skip_threshold_float: Optional[float] = None
    target_threshold_float: Optional[float] = None
    if skip_threshold_val is not None:
        try:
            skip_threshold_float = float(skip_threshold_val)
        except (TypeError, ValueError):
            skip_threshold_float = None
    if target_threshold_val is not None:
        try:
            target_threshold_float = float(target_threshold_val)
        except (TypeError, ValueError):
            target_threshold_float = None

    #
    # write results and template to file
    if args.task_set != 'transfer':
        with open('./sts-enhance-results', 'a') as f:
            model_name = args.model_name_or_path.split('/')[-1]
            skip_field = f" skip={skip_total}"
            if skip_threshold_float is not None:
                skip_field += f" skip_thr={skip_threshold_float:.3f}"
            elif skip_threshold_val is not None:
                skip_field += f" skip_thr={skip_threshold_val}"
            if target_threshold_float is not None:
                skip_field += f" boost_thr={target_threshold_float:.3f}"
            elif target_threshold_val is not None:
                skip_field += f" boost_thr={target_threshold_val}"
            f.write(
                model_name
                + ' '
                + str(COEFF)
                + ' '
                + str(args.tp_starting_index)
                + ' '
                + ' '.join([str(s) for s in sts_scores])
                + skip_field
                + '\n'
            )

    if csv_metrics:
        base_info = OrderedDict()
        base_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base_info["config"] = args.config or ""
        base_info["mode"] = args.mode
        base_info["task_set"] = args.task_set
        base_info["model_name"] = os.path.basename(args.model_name_or_path.rstrip("/"))
        base_info["prompt_method"] = args.prompt_method
        base_info["enable_attention_override"] = str(enhance_cfg.get("enable_attention_override", ""))
        base_info["head_order"] = str(enhance_cfg.get("head_order", ""))
        base_info["override_mode"] = str(enhance_cfg.get("override_mode", ""))
        base_info["top_k"] = "" if enhance_cfg.get("top_k") is None else str(enhance_cfg.get("top_k"))
        base_info["gamma"] = "" if enhance_cfg.get("gamma") is None else str(enhance_cfg.get("gamma"))
        base_info["zero_special_skip_threshold"] = (
            "" if skip_threshold_float is None else f"{skip_threshold_float:.6f}"
        )
        base_info["zero_special_target_threshold"] = (
            "" if target_threshold_float is None else f"{target_threshold_float:.6f}"
        )
        base_info["zero_special_skip"] = str(skip_total)
        append_results_to_csv(base_info, csv_metrics)
if __name__ == "__main__":
    main()
