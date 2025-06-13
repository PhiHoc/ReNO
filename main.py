import json
import logging
import os

import blobfile as bf
import torch
from datasets import load_dataset
from pytorch_lightning import seed_everything
from tqdm import tqdm

from arguments import parse_args
from models import get_model, get_multi_apply_fn
from rewards import get_reward_losses
from training import LatentNoiseTrainer, get_optimizer


def main(args):
    seed_everything(args.seed)
    bf.makedirs(f"{args.save_dir}/logs/{args.task}")
    # Set up logging and name settings
    logger = logging.getLogger()
    settings = (
        f"{args.model}{'_' + args.prompt if args.task == 't2i-compbench' else ''}"
        f"{'_no-optim' if args.no_optim else ''}_{args.seed if args.task != 'geneval' else ''}"
        f"_lr{args.lr}_gc{args.grad_clip}_iter{args.n_iters}"
        f"_reg{args.reg_weight if args.enable_reg else '0'}"
        f"{'_pickscore' + str(args.pickscore_weighting) if args.enable_pickscore else ''}"
        f"{'_clip' + str(args.clip_weighting) if args.enable_clip else ''}"
        f"{'_hps' + str(args.hps_weighting) if args.enable_hps else ''}"
        f"{'_imagereward' + str(args.imagereward_weighting) if args.enable_imagereward else ''}"
        f"{'_aesthetic' + str(args.aesthetic_weighting) if args.enable_aesthetic else ''}"
    )
    file_stream = open(f"{args.save_dir}/logs/{args.task}/{settings}.txt", "w")
    handler = logging.StreamHandler(file_stream)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel("INFO")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logging.info(args)
    if args.device_id is not None:
        logging.info(f"Using CUDA device {args.device_id}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda")
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    # Get reward losses
    reward_losses = get_reward_losses(args, dtype, device, args.cache_dir)

    # Get model and noise trainer
    pipe = get_model(
        args.model, dtype, device, args.cache_dir, args.memsave, args.cpu_offloading,
        lora_path=args.lora_path
    )
    trainer = LatentNoiseTrainer(
        reward_losses=reward_losses,
        model=pipe,
        n_iters=args.n_iters,
        n_inference_steps=args.n_inference_steps,
        seed=args.seed,
        save_all_images=args.save_all_images,
        device=device,
        no_optim=args.no_optim,
        regularize=args.enable_reg,
        regularization_weight=args.reg_weight,
        grad_clip=args.grad_clip,
        log_metrics=args.task == "single" or not args.no_optim,
        imageselect=args.imageselect,
    )

    # Create latents
    if args.model == "flux":
        # currently only support 512x512 generation
        shape = (1, 16 * 64, 64)
    elif args.model != "pixart":
        height = pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = pipe.unet.config.sample_size * pipe.vae_scale_factor
        shape = (
            1,
            pipe.unet.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
    else:
        height = pipe.transformer.config.sample_size * pipe.vae_scale_factor
        width = pipe.transformer.config.sample_size * pipe.vae_scale_factor
        shape = (
            1,
            pipe.transformer.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
    enable_grad = not args.no_optim
    if args.enable_multi_apply:
        multi_apply_fn = get_multi_apply_fn(
            model_type=args.multi_step_model,
            seed=args.seed,
            pipe=pipe,
            cache_dir=args.cache_dir,
            device=device,
            dtype=dtype,
        )
    else:
        multi_apply_fn = None

    total_best_rewards = None
    total_init_rewards = None

    if args.task == "single":
        init_latents = torch.randn(shape, device=device, dtype=dtype)
        latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
        optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
        save_dir = f"{args.save_dir}/{args.task}/{settings}/{args.prompt[:150]}"
        os.makedirs(f"{save_dir}", exist_ok=True)
        init_image, best_image, total_init_rewards, total_best_rewards = trainer.train(
            latents, args.prompt, optimizer, save_dir, multi_apply_fn
        )
        best_image.save(f"{save_dir}/best_image.png")
        init_image.save(f"{save_dir}/init_image.png")
    elif args.task == "example-prompts":
        fo = open("assets/example_prompts.txt", "r")
        prompts = fo.readlines()
        fo.close()
        for i, prompt in tqdm(enumerate(prompts)):
            # Get new latents and optimizer
            init_latents = torch.randn(shape, device=device, dtype=dtype)
            latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
            optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)

            prompt = prompt.strip()
            name = f"{i:03d}_{prompt[:150]}.png"
            save_dir = f"{args.save_dir}/{args.task}/{settings}/{name}"
            os.makedirs(save_dir, exist_ok=True)
            init_image, best_image, init_rewards, best_rewards = trainer.train(
                latents, prompt, optimizer, save_dir, multi_apply_fn
            )
            if i == 0:
                total_best_rewards = {k: 0.0 for k in best_rewards.keys()}
                total_init_rewards = {k: 0.0 for k in best_rewards.keys()}
            for k in best_rewards.keys():
                total_best_rewards[k] += best_rewards[k]
                total_init_rewards[k] += init_rewards[k]
            best_image.save(f"{save_dir}/best_image.png")
            init_image.save(f"{save_dir}/init_image.png")
            logging.info(f"Initial rewards: {init_rewards}")
            logging.info(f"Best rewards: {best_rewards}")
        for k in total_best_rewards.keys():
            total_best_rewards[k] /= len(prompts)
            total_init_rewards[k] /= len(prompts)

        # save results to directory
        with open(f"{args.save_dir}/example-prompts/{settings}/results.txt", "w") as f:
            f.write(
                f"Mean initial all rewards: {total_init_rewards}\n"
                f"Mean best all rewards: {total_best_rewards}\n"
            )
    elif args.task == "t2i-compbench":
        prompt_list_file = f"../T2I-CompBench/examples/dataset/{args.prompt}.txt"
        fo = open(prompt_list_file, "r")
        prompts = fo.readlines()
        fo.close()
        os.makedirs(f"{args.save_dir}/{args.task}/{settings}/samples", exist_ok=True)
        for i, prompt in tqdm(enumerate(prompts)):
            # Get new latents and optimizer
            init_latents = torch.randn(shape, device=device, dtype=dtype)
            latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
            optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)

            prompt = prompt.strip()
            init_image, best_image, init_rewards, best_rewards = trainer.train(
                latents, prompt, optimizer, None, multi_apply_fn
            )
            if i == 0:
                total_best_rewards = {k: 0.0 for k in best_rewards.keys()}
                total_init_rewards = {k: 0.0 for k in best_rewards.keys()}
            for k in best_rewards.keys():
                total_best_rewards[k] += best_rewards[k]
                total_init_rewards[k] += init_rewards[k]
            name = f"{prompt}_{i:06d}.png"
            best_image.save(f"{args.save_dir}/{args.task}/{settings}/samples/{name}")
            logging.info(f"Initial rewards: {init_rewards}")
            logging.info(f"Best rewards: {best_rewards}")
        for k in total_best_rewards.keys():
            total_best_rewards[k] /= len(prompts)
            total_init_rewards[k] /= len(prompts)
    elif args.task == "parti-prompts":
        parti_dataset = load_dataset("nateraw/parti-prompts", split="train")
        total_reward_diff = 0.0
        total_best_reward = 0.0
        total_init_reward = 0.0
        total_improved_samples = 0
        for index, sample in enumerate(parti_dataset):
            # Get new latents and optimizer
            init_latents = torch.randn(shape, device=device, dtype=dtype)
            latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
            optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
            
            os.makedirs(
                f"{args.save_dir}/{args.task}/{settings}/{index}", exist_ok=True
            )
            prompt = sample["Prompt"]
            init_image, best_image, init_rewards, best_rewards = trainer.train(
                latents, prompt, optimizer, multi_apply_fn
            )
            best_image.save(
                f"{args.save_dir}/{args.task}/{settings}/{index}/best_image.png"
            )
            open(
                f"{args.save_dir}/{args.task}/{settings}/{index}/prompt.txt", "w"
            ).write(
                f"{prompt} \n Initial Rewards: {init_rewards} \n Best Rewards: {best_rewards}"
            )
            logging.info(f"Initial rewards: {init_rewards}")
            logging.info(f"Best rewards: {best_rewards}")
            initial_reward = init_rewards[args.benchmark_reward]
            best_reward = best_rewards[args.benchmark_reward]
            total_reward_diff += best_reward - initial_reward
            total_best_reward += best_reward
            total_init_reward += initial_reward
            if best_reward < initial_reward:
                total_improved_samples += 1
            if index == 0:
                total_best_rewards = {k: 0.0 for k in best_rewards.keys()}
                total_init_rewards = {k: 0.0 for k in init_rewards.keys()}
            for k in best_rewards.keys():
                total_best_rewards[k] += best_rewards[k]
                total_init_rewards[k] += init_rewards[k]
        improvement_percentage = total_improved_samples / parti_dataset.num_rows
        mean_best_reward = total_best_reward / parti_dataset.num_rows
        mean_init_reward = total_init_reward / parti_dataset.num_rows
        mean_reward_diff = total_reward_diff / parti_dataset.num_rows
        logging.info(
            f"Improvement percentage: {improvement_percentage:.4f}, "
            f"mean initial reward: {mean_init_reward:.4f}, "
            f"mean best reward: {mean_best_reward:.4f}, "
            f"mean reward diff: {mean_reward_diff:.4f}"
        )
        for k in total_best_rewards.keys():
            total_best_rewards[k] /= len(parti_dataset)
            total_init_rewards[k] /= len(parti_dataset)
        # save results
        os.makedirs(f"{args.save_dir}/parti-prompts/{settings}", exist_ok=True)
        with open(f"{args.save_dir}/parti-prompts/{settings}/results.txt", "w") as f:
            f.write(
                f"Mean improvement: {improvement_percentage:.4f}, "
                f"mean initial reward: {mean_init_reward:.4f}, "
                f"mean best reward: {mean_best_reward:.4f}, "
                f"mean reward diff: {mean_reward_diff:.4f}\n"
                f"Mean initial all rewards: {total_init_rewards}\n"
                f"Mean best all rewards: {total_best_rewards}"
            )
    # elif args.task == "geneval":
    #     prompt_list_file = "../geneval/prompts/evaluation_metadata.jsonl"
    #     with open(prompt_list_file) as fp:
    #         metadatas = [json.loads(line) for line in fp]
    #     outdir = f"{args.save_dir}/{args.task}/{settings}"
    #     for index, metadata in enumerate(metadatas):
    #         # Get new latents and optimizer
    #         init_latents = torch.randn(shape, device=device, dtype=dtype)
    #         latents = torch.nn.Parameter(init_latents, requires_grad=True)
    #         optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
    #
    #         prompt = metadata["prompt"]
    #         init_image, best_image, init_rewards, best_rewards = trainer.train(
    #             latents, prompt, optimizer, None, multi_apply_fn
    #         )
    #         logging.info(f"Initial rewards: {init_rewards}")
    #         logging.info(f"Best rewards: {best_rewards}")
    #         outpath = f"{outdir}/{index:0>5}"
    #         os.makedirs(f"{outpath}/samples", exist_ok=True)
    #         with open(f"{outpath}/metadata.jsonl", "w") as fp:
    #             json.dump(metadata, fp)
    #         best_image.save(f"{outpath}/samples/{args.seed:05}.png")
    #         if i == 0:
    #             total_best_rewards = {k: 0.0 for k in best_rewards.keys()}
    #             total_init_rewards = {k: 0.0 for k in best_rewards.keys()}
    #         for k in best_rewards.keys():
    #             total_best_rewards[k] += best_rewards[k]
    #             total_init_rewards[k] += init_rewards[k]
    #     for k in total_best_rewards.keys():
    #         total_best_rewards[k] /= len(parti_dataset)
    #         total_init_rewards[k] /= len(parti_dataset)



    elif args.task == "bear_dataset" or args.task == "turtle_dataset":

        if args.prompt_file:

            prompt_list_file = args.prompt_file

        elif args.task == "bear_dataset":

            prompt_list_file = "bears_structured.jsonl"

        else:  # turtle_dataset

            prompt_list_file = "turtles_structured.jsonl"

        logging.info(f"Loading structured prompts from {prompt_list_file}")

        try:

            with open(prompt_list_file, "r", encoding="utf-8") as f:

                prompts_data = [json.loads(line) for line in f]

        except FileNotFoundError:

            logging.error(f"FATAL: Prompt file not found at {prompt_list_file}. Please create it.")

            return  # Exit if file not found

        except json.JSONDecodeError as e:

            logging.error(f"FATAL: Error decoding JSON from {prompt_list_file}. Check file format. Error: {e}")

            return  # Exit on format error

        base_save_path = f"{args.save_dir}/{args.task}/{settings}"

        os.makedirs(base_save_path, exist_ok=True)

        num_prompts = len(prompts_data)

        total_images_generated = 0

        for i, sample in tqdm(enumerate(prompts_data), total=num_prompts):

            class_name = sample["class_name"]

            prompt = sample["prompt"]

            # Tạo thư mục con cho mỗi class

            class_save_dir = os.path.join(base_save_path, class_name.replace(" ", "_"))

            os.makedirs(class_save_dir, exist_ok=True)

            expected_filename = f"{i:04d}_000.png"
            expected_filepath = os.path.join(class_save_dir, expected_filename)

            if os.path.exists(expected_filepath):
                logging.info(f"Output for prompt {i} already exists. Skipping.")
                continue  # Bỏ qua prompt này và đi đến prompt tiếp theo

            # Vòng lặp để tạo ra số lượng ảnh mong muốn cho mỗi prompt

            for j in range(args.num_images_per_prompt):

                # Tạo latents và optimizer mới cho mỗi ảnh để đảm bảo sự khác biệt

                current_seed = args.seed + total_images_generated

                seed_everything(current_seed)

                init_latents = torch.randn(shape, device=device, dtype=dtype)

                latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)

                optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)

                init_image, best_image, init_rewards, best_rewards = trainer.train(

                    latents, prompt, optimizer, None, multi_apply_fn

                )

                # Khởi tạo dict tổng kết ở lần chạy đầu tiên

                if total_best_rewards is None:
                    total_best_rewards = {k: 0.0 for k in best_rewards.keys()}

                    total_init_rewards = {k: 0.0 for k in best_rewards.keys()}

                # Cộng dồn điểm thưởng

                for k in best_rewards.keys():
                    total_best_rewards[k] += best_rewards[k]

                    total_init_rewards[k] += init_rewards[k]

                # Đặt tên file đơn giản với chỉ số

                name = f"{i:04d}_{j:03d}.png"

                best_image.save(os.path.join(class_save_dir, name))

                logging.info(f"Saved: {os.path.join(class_save_dir, name)}")

                logging.info(f"Prompt: {prompt}")

                logging.info(f"Initial rewards: {init_rewards}")

                logging.info(f"Best rewards: {best_rewards}")

                total_images_generated += 1

        # Tính điểm thưởng trung bình

        if total_images_generated > 0:

            for k in total_best_rewards.keys():
                total_best_rewards[k] /= total_images_generated

                total_init_rewards[k] /= total_images_generated
    else:
        raise ValueError(f"Unknown task {args.task}")
    # log total rewards
    logging.info(f"Mean initial rewards: {total_init_rewards}")
    logging.info(f"Mean best rewards: {total_best_rewards}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
