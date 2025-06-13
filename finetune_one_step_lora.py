import argparse
import logging
import os
from pathlib import Path
import random
from typing import Optional
from accelerate.utils import set_seed, ProjectConfiguration
import torch
import math
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
# LoRAAttnProcessor in diffusers>=0.26.0 is a placeholder class.
# We need to define the old LoRAAttnProcessor here to use it with the script.
# from diffusers.models.attention_processor import LoRAAttnProcessor, Attention
from diffusers.models.attention_processor import Attention
from diffusers.optimization import get_scheduler
from huggingface_hub import hf_hub_download
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# Code được lấy từ diffusers==0.21.4 để đảm bảo tương thích
# Nguồn: https://github.com/huggingface/diffusers/blob/v0.21.4/src/diffusers/models/attention_processor.py

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / self.rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

class LoRAAttnProcessor(nn.Module):
    r"""
    Processor for implementing LoRA attention mechanism.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, network_alpha=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.network_alpha = network_alpha

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states) + self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + self.to_v_lora(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

IMAGENET_TEMPLATES_TINY = [
    "a photo of a {}.",
    "a rendering of a {}.",
    "a cropped photo of the {}.",
    "the photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a photo of my {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a photo of one {}.",
    "a close-up photo of the {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a good photo of a {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "a photo of the large {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
]

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a one-step model with dynamic prompts.")

    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="stabilityai/sdxl-turbo",
        help="One-step model to fine-tune (e.g., 'stabilityai/sdxl-turbo', 'ByteDance/Hyper-SD')."
    )
    parser.add_argument(
        "--train_data_dir", type=str, required=True,
        help="Path to the root directory of your image dataset (containing class sub-folders)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./lora_weights/dynamic_one_step_lora",
        help="Directory to save the trained LoRA."
    )
    # Thêm num_train_epochs để tránh lỗi khi truy cập args.num_train_epochs
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=16)


    #Load from checkpoint
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are stores in states separated"
            " by folders named `checkpoint-{global_step}`. These checkpoints are only suitable for resuming training."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether to resume training from a checkpoint. Can be either a string containing the path to a"
            " checkpoint folder, or `latest` to automatically select the last available checkpoint."
        ),
    )

    args = parser.parse_args()
    return args


class DynamicPromptDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, tokenizer, tokenizer_2=None, size=512):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2  # For SDXL models
        self.size = size

        self.image_paths = []
        self.class_names = []

        # Tải đường dẫn ảnh và tên class từ các thư mục con
        self.class_folders = sorted([d for d in os.scandir(data_root) if d.is_dir()], key=lambda d: d.name)
        self.class_name_list = [d.name for d in self.class_folders]

        for class_dir in self.class_folders:
            image_files = list(Path(class_dir.path).iterdir())
            self.image_paths.extend(image_files)
            # Gán tên class (tên thư mục) cho mỗi ảnh trong thư mục đó
            self.class_names.extend([class_dir.name] * len(image_files))

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        example = {}
        image = Image.open(self.image_paths[idx])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["pixel_values"] = self.image_transforms(image)

        # Tạo prompt động
        class_name = self.class_names[idx].replace("_", " ")  # Thay thế gạch dưới bằng khoảng trắng
        prompt = random.choice(IMAGENET_TEMPLATES_TINY).format(class_name)
        example["prompt"] = prompt

        return example


def main():
    args = parse_args()

    # Thiết lập Accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=project_config,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    # Tạo thư mục output chính
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Tải mô hình One-Step ---
    logger.info(f"Loading one-step model: {args.pretrained_model_name_or_path}")
    if args.pretrained_model_name_or_path == "stabilityai/sdxl-turbo":
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        tokenizer = CLIPTokenizer.from_pretrained("stabilityai/sdxl-turbo", subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/sdxl-turbo", subfolder="tokenizer_2")
        text_encoder = CLIPTextModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="text_encoder", torch_dtype=torch.float16)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("stabilityai/sdxl-turbo", subfolder="text_encoder_2", torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="unet", torch_dtype=torch.float16)
    elif args.pretrained_model_name_or_path == "ByteDance/Hyper-SD":
        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SDXL-1step-Unet.safetensors"
        from safetensors.torch import load_file
        unet = UNet2DConditionModel.from_config(base_model, subfolder="unet")
        unet.load_state_dict(load_file(hf_hub_download(repo_name, ckpt_name)))
        vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float16)
        tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2")
        text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=torch.float16)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_model, subfolder="text_encoder_2", torch_dtype=torch.float16)
    else:
        raise ValueError("Unsupported one-step model specified.")

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # --- Cấy LoRA vào U-Net ---
    logger.info("Injecting LoRA into UNet...")
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        try:
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id_str = name[len("up_blocks."):].split(".")[0]
                block_id = int(block_id_str)
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id_str = name[len("down_blocks."):].split(".")[0]
                block_id = int(block_id_str)
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                continue
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse hidden_size for {name}. Skipping. Error: {e}")
            continue
        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.lora_rank)
    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    params_to_optimize = lora_layers.parameters()
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. `pip install bitsandbytes`")
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(params_to_optimize, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08)

    logger.info("Preparing dataset...")
    train_dataset = DynamicPromptDataset(data_root=args.train_data_dir, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution)
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        prompts = [example["prompt"] for example in examples]
        return {"pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(), "prompts": prompts}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Tính số epoch cần chạy để đạt được max_train_steps
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(lora_layers, optimizer, train_dataloader, lr_scheduler)

    global_step = 0
    first_epoch = 0
    resume_step = 0
    
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            # Cập nhật global_step từ tên thư mục checkpoint
            global_step = int(path.split("-")[1])
            
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = resume_global_step // len(train_dataloader)
            resume_step = resume_global_step % len(train_dataloader)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Resuming from step {global_step}")

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Bỏ qua các step đã train ở epoch hiện tại nếu resume
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                progress_bar.update(1) # Cập nhật progress bar cho các step bị bỏ qua
                continue

            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.full((bsz,), noise_scheduler.config.num_train_timesteps - 1, device=accelerator.device, dtype=torch.long)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                prompt_ids = tokenizer(batch["prompts"], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
                prompt_ids_2 = tokenizer_2(batch["prompts"], padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
                
                prompt_embeds_1 = text_encoder(prompt_ids, return_dict=True)
                text_encoder_2_output = text_encoder_2(prompt_ids_2, return_dict=True)
                prompt_embeds_2 = text_encoder_2_output.last_hidden_state
                encoder_hidden_states = torch.cat([prompt_embeds_1.last_hidden_state, prompt_embeds_2], dim=-1)
                pooled_prompt_embeds = text_encoder_2_output.text_embeds
                
                add_time_ids = torch.tensor([[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]], device=accelerator.device, dtype=encoder_hidden_states.dtype).repeat(bsz, 1)
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}
                
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs).sample
                loss = f.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_layers.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"loss": loss.detach().item()}, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        lora_state_dict = accelerator.get_state_dict(lora_layers)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file
        save_file(lora_state_dict, output_dir / "pytorch_lora_weights.safetensors")
        logger.info(f"LoRA weights saved successfully to {output_dir / 'pytorch_lora_weights.safetensors'}")

if __name__ == "__main__":
    main()