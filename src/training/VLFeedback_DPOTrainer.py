"""
VLFeedback DPO Trainer for Qwen3-VL with lazy image loading.

Images are stored as file paths in the dataset and loaded on-the-fly
in the data collator, keeping dataset memory usage minimal (~50MB for 10k samples
vs ~15GB when storing pixel_values).

Overrides from TRL 0.12's DPOTrainer:
  - process_row: tokenize text only, keep image_path for lazy loading
  - data collator: load images on-the-fly, produce pixel_values/image_grid_thw
  - concatenated_inputs: duplicate image data for chosen+rejected
  - concatenated_forward: pass image_grid_thw to model
"""
from typing import Any, Dict, List, Literal, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from trl import DPOTrainer
from trl.trainer.dpo_trainer import pad


class QwenVLDPODataCollator:
    """Data collator with lazy image loading for Qwen3-VL DPO training.

    Images are loaded from disk and processed per-batch via the processor,
    so only one batch's pixel_values live in memory at a time.
    """

    def __init__(self, processor, pad_token_id=0):
        self.processor = processor
        self.pad_token_id = pad_token_id

    def _load_image(self, image_path):
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224))

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_prompt_input_ids = []
        all_pixel_values = []
        all_image_grid_thw = []

        for ex in examples:
            image = self._load_image(ex["image_path"])

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex["prompt"]},
                ]}
            ]
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            processed = self.processor(
                images=image, text=prompt_text,
                add_special_tokens=False, return_tensors="pt",
            )

            all_prompt_input_ids.append(processed["input_ids"].squeeze(0))
            all_pixel_values.append(processed["pixel_values"])
            if "image_grid_thw" in processed:
                all_image_grid_thw.append(processed["image_grid_thw"].reshape(-1, 3))

        prompt_attention_mask = [torch.ones_like(ids) for ids in all_prompt_input_ids]
        chosen_input_ids = [torch.tensor(ex["chosen_input_ids"]) for ex in examples]
        chosen_attention_mask = [torch.ones_like(ids) for ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(ex["rejected_input_ids"]) for ex in examples]
        rejected_attention_mask = [torch.ones_like(ids) for ids in rejected_input_ids]

        output = {
            "prompt_input_ids": pad(all_prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"),
            "prompt_attention_mask": pad(prompt_attention_mask, padding_value=0, padding_side="left"),
            "chosen_input_ids": pad(chosen_input_ids, padding_value=self.pad_token_id),
            "chosen_attention_mask": pad(chosen_attention_mask, padding_value=0),
            "rejected_input_ids": pad(rejected_input_ids, padding_value=self.pad_token_id),
            "rejected_attention_mask": pad(rejected_attention_mask, padding_value=0),
        }

        if all_pixel_values:
            output["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        if all_image_grid_thw:
            output["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

        return output


class VLFeedback_DPOTrainer(DPOTrainer):
    """DPOTrainer with Qwen3-VL lazy image loading and image_grid_thw support."""

    def __init__(self, processor=None, **kwargs):
        self.vl_processor = processor
        super().__init__(**kwargs)
        self.data_collator = QwenVLDPODataCollator(
            processor=self.vl_processor, pad_token_id=self.padding_value
        )
        self._patch_for_gradient_checkpointing()

    def _patch_for_gradient_checkpointing(self):
        """Fix Qwen3-VL gradient checkpointing issues:
        1. Disable checkpointing for vision blocks (they cause tensor mismatch)
        2. Make _deepstack_process non-in-place (prevents checkpoint metadata mismatch)
        3. Re-apply after every gradient_checkpointing_enable call
        """
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLVisionBlock,
                Qwen3VLTextModel,
            )
        except ImportError:
            return

        def _deepstack_process_no_inplace(self_text, hidden_states, visual_pos_masks, visual_embeds):
            visual_pos_masks = visual_pos_masks.to(hidden_states.device)
            visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
            new_hidden_states = hidden_states.clone()
            new_hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :] + visual_embeds
            return new_hidden_states

        for module in self.model.modules():
            if isinstance(module, Qwen3VLTextModel):
                import types
                module._deepstack_process = types.MethodType(_deepstack_process_no_inplace, module)

        original_fn = self.model.gradient_checkpointing_enable

        def patched_fn(**gc_kwargs):
            original_fn(**gc_kwargs)
            for module in self.model.modules():
                if isinstance(module, Qwen3VLVisionBlock):
                    module.gradient_checkpointing = False

        self.model.gradient_checkpointing_enable = patched_fn

    def log(self, logs, *args, **kwargs):
        super().log(logs)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute ref log probs FIRST (no_grad → memory freed) then policy forward."""
        metrics = {}

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)
            torch.cuda.empty_cache()

        model_output = self.concatenated_forward(model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"], model_output["rejected_logps"], ref_chosen_logps, ref_rejected_logps
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/chosen"] = model_output["chosen_logps"].detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = model_output["rejected_logps"].detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = model_output["mean_chosen_logits"].detach().cpu()
        metrics[f"{prefix}logits/rejected"] = model_output["mean_rejected_logits"].detach().cpu()

        return losses.mean(), metrics

    @staticmethod
    def process_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        """
        Text-only tokenization for Qwen3-VL DPO.

        Only tokenizes chosen/rejected text and produces a placeholder
        prompt_input_ids. The actual prompt_input_ids (with expanded image
        tokens) and pixel_values are computed in the data collator when
        the image is loaded from disk.
        """
        if "chosen_input_ids" in features:
            return features

        tokenizer = processing_class.tokenizer

        prompt_input_ids = tokenizer(
            features["prompt"], add_special_tokens=False
        )["input_ids"]

        chosen_input_ids = tokenizer(
            features["chosen"], add_special_tokens=False
        )["input_ids"]
        rejected_input_ids = tokenizer(
            features["rejected"], add_special_tokens=False
        )["input_ids"]

        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "image_path": features["image_path"],
            "prompt": features["prompt"],
        }

    def concatenated_inputs(self, batch, padding_value=0):
        """Override to also concatenate image_grid_thw."""
        output = super().concatenated_inputs(batch, padding_value=padding_value)
        if "image_grid_thw" in batch:
            output["image_grid_thw"] = torch.cat([batch["image_grid_thw"], batch["image_grid_thw"]], dim=0)
        return output

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]):
        """
        Override to pass image_grid_thw to Qwen3-VL model.
        TRL 0.12's default implementation passes pixel_values but not image_grid_thw.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "image_grid_thw" in concatenated_batch:
            model_kwargs["image_grid_thw"] = concatenated_batch["image_grid_thw"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            for i in range(attention_mask.size(0)):
                nonzero = torch.nonzero(attention_mask[i])
                if len(nonzero) > 0:
                    first_one_idx = nonzero[0].item()
                    input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
                    attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
                    loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

            empty_cols = torch.sum(attention_mask, dim=0) == 0
            if empty_cols.any():
                first_empty_col = torch.nonzero(empty_cols)[0].item()
            else:
                first_empty_col = attention_mask.size(1) + 1
            input_ids = input_ids[:, : first_empty_col - 1]
            attention_mask = attention_mask[:, : first_empty_col - 1]
            loss_mask = loss_mask[:, : first_empty_col - 1]

            if self.args.max_length is not None:
                input_ids = input_ids[:, : self.args.max_length]
                attention_mask = attention_mask[:, : self.args.max_length]
                loss_mask = loss_mask[:, : self.args.max_length]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)

            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:].clone()
            loss_mask = loss_mask[:, 1:].bool()

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        labels[~loss_mask] = 0

        chunk_size = 128
        per_token_logps_chunks = []
        logit_sum = 0.0
        logit_count = 0
        chosen_logit_sum = 0.0
        chosen_logit_count = 0
        rejected_logit_sum = 0.0
        rejected_logit_count = 0

        for i in range(0, logits.size(1), chunk_size):
            chunk_logits = logits[:, i:i+chunk_size, :]
            chunk_labels = labels[:, i:i+chunk_size]
            chunk_mask = loss_mask[:, i:i+chunk_size]

            chunk_log_probs = chunk_logits.log_softmax(-1)
            chunk_per_token_logps = torch.gather(
                chunk_log_probs, dim=2, index=chunk_labels.unsqueeze(2)
            ).squeeze(2)
            chunk_per_token_logps[~chunk_mask] = 0
            per_token_logps_chunks.append(chunk_per_token_logps)

            gathered_logits = torch.gather(
                chunk_logits, dim=2, index=chunk_labels.unsqueeze(2)
            ).squeeze(2)
            c_mask = chunk_mask[:num_examples]
            r_mask = chunk_mask[num_examples:]
            chosen_logit_sum += gathered_logits[:num_examples][c_mask].sum()
            chosen_logit_count += c_mask.sum()
            rejected_logit_sum += gathered_logits[num_examples:][r_mask].sum()
            rejected_logit_count += r_mask.sum()

            del chunk_log_probs, gathered_logits

        del logits

        per_token_logps = torch.cat(per_token_logps_chunks, dim=1)
        all_logps = per_token_logps.sum(-1)

        output = {}
        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = chosen_logit_sum / max(chosen_logit_count, 1)
        output["mean_rejected_logits"] = rejected_logit_sum / max(rejected_logit_count, 1)
        return output
