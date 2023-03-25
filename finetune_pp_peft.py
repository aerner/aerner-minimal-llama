import argparse
import os
import json
import math
import tqdm.auto as tqdm

import torch

import torch
import datasets
import transformers
from finetune_pp import RepeatingLoader, DatasetDataset
import peft


def write_json(x, path):
    with open(path, "w") as f:
        f.write(json.dumps(x))


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def model_forward(model, inputs):
    h = inputs
    h = h.to(model.base_model.model.model.embed_tokens.weight.device)
    h = model.base_model.model.model.embed_tokens(h)
    for layer in model.base_model.model.model.layers:
        h = h.to(layer.input_layernorm.weight.device)
        h = layer(h)[0]
    h = h.to(model.base_model.model.model.norm.weight.device)
    h = model.base_model.model.model.norm(h)
    h = model.base_model.model.lm_head(h)
    return h

class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_model_id", type=str)

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_steps", type=int)
    parser.add_argument("--save_interval", type=int)

    parser.add_argument("--peft_mode", type=str, default="lora")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--num_virtual_tokens", type=int, default=32)
    parser.add_argument("--mapping_hidden_dim", type=int, default=1024)

    args = parser.parse_args()

    #
    #
    # Dataset
    #
    #
    print("Setup Data")
    dataset = datasets.load_from_disk(args.dataset_path)
    dataset = dataset.remove_columns(["instruction", "input", "output"])
    dataloader = RepeatingLoader(torch.utils.data.DataLoader(
        DatasetDataset(dataset),
        batch_size=args.batch_size,
        # shuffle=True
    ))

    #
    #
    # Model
    #
    #
    print("Setup Model")
    # The auto/balance balancing strategy doesn't seem to work correctly,
    # so we manually compute the mappings.
    num_layers = read_json(os.path.join(args.model_path, "config.json"))[
        "num_hidden_layers"]
    device_ids = list(range(torch.cuda.device_count()))
    device_map = {
        "model.embed_tokens": device_ids[0],
        "model.norm.weight": device_ids[-1],
        "lm_head": device_ids[-1],
    }
    allocations = [
        device_ids[i] for i in
        sorted(list(range(len(device_ids))) *
               math.ceil(num_layers / len(device_ids)))
    ]
    for layer_i, device_id in enumerate(allocations):
        device_map[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.down_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.mlp.up_proj.weight"] = device_id
        device_map[f"model.layers.{layer_i}.input_layernorm.weight"] = device_id
        device_map[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = device_id
        device_map[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = device_id

    model = transformers.LlamaForCausalLM.from_pretrained(
        args.model_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # model.lm_head = CastOutputToFloat(model.lm_head)
    model.lm_head.to(torch.float16)
    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False

    #
    #
    # Peft configuration
    #
    #
    print("Setup PEFT")
    model = peft.get_peft_model(model, peft.LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
        inference_mode=False,
    ))

    print(model)

    #
    #
    # Optimizer ready
    #
    #
    print("Setup optimizer")
    opt = torch.optim.AdamW([
        p
        for p in model.parameters()
        if p.requires_grad
    ], lr=args.learning_rate)

    # # Restart progress
    # if os.path.exists(latest_path):
    #     start = read_json(latest_path)["latest_step"]
    #     model.load_state_dict(
    #         torch.load(os.path.join(os.path.join(args.save_dir, f"model-{start + 1:06d}.p"))), strict=False)
    #     opt.load_state_dict(
    #         torch.load(os.path.join(os.path.join(args.save_dir, f"opt-{start + 1:06d}.p"))))
    # else:
    #     start = 0

    # # Save initial model
    # print("Save initial model")
    # model.save_pretrained(args.finetune_model_id)
    # torch.save(model.state_dict(), "{}.pt".format(args.finetune_model_id))
    # print("Save initial model complete")

    # Train (maybe can replace with Trainer? I think Trainer might mess up the device mappings though.)
    print("Start training")
    generator = iter(dataloader)
    for step in tqdm.trange(args.num_train_steps, initial=0):
        input_ids, labels = next(generator)
        logits = model_forward(model, input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            labels.view(-1).to(logits.device),
        )
        loss.backward()
        opt.step()

        actual_step = step + 1

        if step % 10 == 0:
            print(f"Loss={loss.item():.3f}")

        if actual_step % args.gradient_accumulation_steps == 0:
            opt.zero_grad()

        if actual_step % args.save_interval == 0:
            model.save_pretrained(args.finetune_model_id)
            torch.save(model.state_dict(), "{}.pt".format(
                args.finetune_model_id))


if __name__ == "__main__":
    main()
