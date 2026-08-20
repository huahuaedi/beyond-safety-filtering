import os, sys, yaml, json, pickle, argparse
from pathlib import Path
import torch
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 导入配置模块并立即设置路径（必须在导入 emu3 之前）
from config import setup_paths_early
setup_paths_early()

from transformers import AutoProcessor
from emu3.mllm import Emu3Tokenizer, Emu3Pi0, Emu3Pi0Config
from utils.train_pi0 import DataArguments # Import from training script
from datasets import Emu3DrivingVAVADataset # Import from training script

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "rl_prep"))
from differentiable_vehicle_model import (  # noqa: E402
    VehicleModelConfig,
    absolute_to_relative_se2,
)
from native_control_expert import execute_model_action  # noqa: E402
from native_control_expert import initial_state_from_relative_history  # noqa: E402

# 导入配置获取函数（用于在 main 中获取配置）
from config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Emu3Pi0 with VAVA dataset format.")
    parser.add_argument("--emu_hub", required=True, type=str, help="Path to the trained Emu3Pi0 model hub.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save inference results.")
    parser.add_argument(
        "--export_vlm_context", action="store_true",
        help="Store the pooled VLM token context required by the unified policy.",
    )
    parser.add_argument("--test_data_pkl", required=True, type=str, help="Path to the test data meta pickle file.")
    parser.add_argument("--token_yaml", type=str, default=None, help="Path to the token YAML file for naming outputs.")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Number of Pi0 inference steps.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of dataloader workers.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file.")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Deterministic flow-noise seed; combined with the dataset index per scene.",
    )
    parser.add_argument(
        "--native_control_checkpoint", type=str, default=None,
        help="Action-side checkpoint whose [8,3] output represents acceleration/curvature-rate logits.",
    )
    return parser.parse_args()


def load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        return torch.load(path, map_location="cpu")


def load_native_control_weights(model, checkpoint):
    payload = load_torch_checkpoint(checkpoint)
    state = payload.get("action_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint}: missing action_state_dict")
    modules = {
        "action_expert": model.action_expert,
        "tau_emb": model.tau_emb,
        "action_projector": model.action_projector,
        "state_projector": model.state_projector,
        "action_decoder": model.action_decoder,
    }
    missing = sorted(set(modules) - set(state))
    if missing:
        raise ValueError(f"{checkpoint}: missing native-control modules {missing}")
    for name, module in modules.items():
        module.load_state_dict(state[name], strict=True)
    metadata = payload.get("metadata", {})
    if metadata.get("representation") not in {
        "native_acceleration_curvature_rate_v1",
        "native_acceleration_curvature_rate_v2",
    }:
        raise ValueError(f"{checkpoint}: unsupported representation {metadata.get('representation')!r}")
    return metadata


def setup_distributed():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def main():
    args = parse_args()
    
    # 加载配置
    config = get_config(args.config)
    
    rank, world_size = setup_distributed()
    device = f"cuda:{rank}"

    # ========================================================================================
    # 1. Create DataArguments to EXACTLY match training configuration
    # ========================================================================================
    action_tokenizer_path = config.get_path("paths.action_tokenizer")
    if action_tokenizer_path is None:
        raise ValueError("action_tokenizer_path must be set via config file or VLA_ACTION_TOKENIZER environment variable")
    
    data_args = DataArguments(
        data_path=args.test_data_pkl,
        actions=True,
        driving=True,
        use_previous_actions=True,
        actions_format="fast",
        action_tokenizer_path=action_tokenizer_path,
        frames=config.get("data.frames", 1),
        action_frames=config.get("data.action_frames", 8),
        action_dim=config.get("data.action_dim", 3),
        cur_frame_idx=config.get("data.cur_frame_idx", 3),
        pre_action_frames=config.get("data.pre_action_frames", 3),
        video_format=None,
        use_flip=False
    )

    # ========================================================================================
    # 2. Load Tokenizer and Dataset (Identical to training)
    # ========================================================================================
    vlm_model_path_for_tokenizer = config.get_path("paths.vlm_model")
    if vlm_model_path_for_tokenizer is None:
        raise ValueError("vlm_model path must be set via config file or VLA_VLM_MODEL environment variable")
    
    tokenizer = Emu3Tokenizer.from_pretrained(
        vlm_model_path_for_tokenizer,
        model_max_length=config.get("model.model_max_length", 1400),
        padding_side=config.get("model.padding_side", "right"),
        use_fast=config.get("model.use_fast", False),
    )

    dataset = Emu3DrivingVAVADataset(data_args, tokenizer=tokenizer)
    
    # ========================================================================================
    # 3. Setup DataLoader with DistributedSampler
    # ========================================================================================
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    num_workers = args.num_workers if args.num_workers is not None else config.get("data.num_workers", 12)
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("data.batch_size", 1),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=config.get("data.pin_memory", True),
        collate_fn=getattr(dataset, 'collate_fn', None)
    )

    # ========================================================================================
    # 4. Load Model
    # ========================================================================================
    model_config = Emu3Pi0Config.from_pretrained(os.path.join(args.emu_hub, "config.json"))
    
    # 获取 torch_dtype
    torch_dtype_str = config.get("model.torch_dtype", "bfloat16")
    torch_dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = torch_dtype_map.get(torch_dtype_str, torch.bfloat16)
    
    model, loading_info = Emu3Pi0.from_pretrained(
        args.emu_hub,
        config=model_config,
        pretrain_vlm_path=vlm_model_path_for_tokenizer,
        attn_implementation=config.get("model.attn_implementation", "sdpa"),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        output_loading_info=True
    )
    if rank == 0:
        print("Missing keys:", loading_info["missing_keys"])
        print("Unexpected keys:", loading_info["unexpected_keys"])
        print("Mismatched sizes:", loading_info.get("mismatched_keys", "N/A"))
        
    model = model.to(device).eval()
    native_control_metadata = None
    native_control_config = None
    if args.native_control_checkpoint:
        native_control_metadata = load_native_control_weights(
            model, args.native_control_checkpoint
        )
        native_control_config = VehicleModelConfig(
            **native_control_metadata.get("vehicle_model", {})
        )
        model.eval()
        if rank == 0:
            print(
                f"Loaded native-control checkpoint={args.native_control_checkpoint} "
                f"representation={native_control_metadata['representation']}",
                flush=True,
            )

    # ========================================================================================
    # 5. Load external metadata (token names for saving, norm stats)
    # ========================================================================================
    token_yaml_path = args.token_yaml if args.token_yaml else config.get_path("paths.token_yaml")
    if token_yaml_path is None:
        raise ValueError("token_yaml path must be set via config file, command line argument, or VLA_TOKEN_YAML environment variable")
    
    with open(token_yaml_path, 'r') as f:
        token_list = yaml.safe_load(f)['tokens']

    norm_stats_path = config.get_path("paths.norm_stats")
    if norm_stats_path is None:
        raise ValueError("norm_stats path must be set via config file or VLA_NORM_STATS environment variable")
    
    norm_cfg = json.load(open(norm_stats_path, 'r'))
    action_low = torch.tensor(norm_cfg['norm_stats']['libero']['q01'], device=device)
    action_high = torch.tensor(norm_cfg['norm_stats']['libero']['q99'], device=device)
    
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================================================
    # 6. Main Inference Loop
    # ========================================================================================
    rank_indices = list(sampler)
    pbar = tqdm(zip(dataloader, rank_indices), total=len(rank_indices), desc=f"Rank {rank}", position=rank, disable=(rank!=0))
    
    for batch, original_idx in pbar:
        # Move all tensors in the batch to the correct device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        gt_action = batch.get("action")

        num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else config.get("inference.num_inference_steps", 10)
        
        # Make stochastic flow sampling paired across checkpoints: the same
        # evaluation seed and scene index always receive identical initial
        # noise, independent of model-loading RNG consumption.
        scene_seed = int(args.seed) * 1_000_003 + int(original_idx)
        torch.manual_seed(scene_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(scene_seed)
        with torch.no_grad():
            sampled = model.sample_actions(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pre_action=batch["pre_action"].to(torch.bfloat16),
                cmd=batch["cmd"].to(torch.bfloat16),
                num_inference_steps=num_inference_steps,
                action_frames=data_args.action_frames,
                action_dim=data_args.action_dim,
                return_vlm_context=args.export_vlm_context,
            )
            if args.export_vlm_context:
                predicted_action, vlm_context = sampled
            else:
                predicted_action = sampled

        if native_control_config is None:
            # Standard DriveVLA output: normalized relative SE(2) waypoints.
            action_denorm = 0.5 * (predicted_action.squeeze(0) + 1) * (action_high - action_low) + action_low
        else:
            # Native output: unconstrained acceleration/curvature-rate logits.
            # The initial vehicle state comes only from observed previous
            # actions, never from held-out future trajectories or NAVSIM labels.
            history = batch["pre_action"].float()
            history_denorm = 0.5 * (history + 1) * (action_high - action_low) + action_low
            initial_state = initial_state_from_relative_history(
                history_denorm, native_control_config
            )
            execution = execute_model_action(
                predicted_action.float(), initial_state, native_control_config
            )
            action_denorm = absolute_to_relative_se2(
                execution["absolute_trajectory"]
            ).squeeze(0)
        
        token_name = token_list[original_idx]
        
        output_dict = {
            "action": action_denorm.cpu().numpy().tolist()
        }
        if args.export_vlm_context:
            output_dict["vlm_context"] = vlm_context[0].cpu().tolist()
        if native_control_config is not None:
            output_dict["native_control"] = {
                "controls": execution["controls"].squeeze(0).cpu().numpy().tolist(),
                "initial_state_source": "observed_pre_action",
                "checkpoint": args.native_control_checkpoint,
            }

        if gt_action is not None:
            gt_action_denorm = 0.5 * (gt_action.squeeze(0) + 1) * (action_high - action_low) + action_low
            output_dict["action_gt_denorm"] = gt_action_denorm.cpu().numpy().tolist()
        
        output_path = os.path.join(args.output_dir, f"{token_name}.json")
        with open(output_path, "w") as f:
            json.dump(output_dict, f, indent=4)
    
    dist.barrier()
    if rank == 0:
        print(f"\nInference complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
