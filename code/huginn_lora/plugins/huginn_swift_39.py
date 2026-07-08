from types import MethodType

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from swift.llm.model.register import Model, ModelGroup, ModelMeta, register_model
from swift.llm.template.register import register_template
from swift.llm.template.template_meta import TemplateMeta

MODEL_DIR = '/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125'


def patch_huginn_shift_loss(model):
    """Patch Huginn forward so SFT uses standard next-token shift loss."""
    if getattr(model, "_huginn_shift_loss_patched", False):
        print("[HuginnLoader] shift-loss patch already applied")
        return model

    original_forward = model.forward

    def forward_with_shift_loss(self, *args, **kwargs):
        labels = kwargs.get("labels", None)

        if not getattr(self, "_huginn_batch_debug_printed", False):
            print("[Huginn-batch-debug] num_args =", len(args), flush=True)
            print("[Huginn-batch-debug] keys =", sorted(kwargs.keys()), flush=True)
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    print(
                        f"[Huginn-batch-debug] {k}: "
                        f"shape={tuple(v.shape)} dtype={v.dtype} device={v.device}",
                        flush=True,
                    )
                else:
                    print(
                        f"[Huginn-batch-debug] {k}: type={type(v)} value={v}",
                        flush=True,
                    )

            tok = getattr(self, "_huginn_debug_tokenizer", None)
            batch_input_ids = kwargs.get("input_ids", None)
            if tok is not None and batch_input_ids is not None:
                decoded = tok.decode(
                    batch_input_ids[0].detach().cpu().tolist(),
                    skip_special_tokens=False,
                )
                print("[Huginn-batch-debug] decoded_input_prefix =", decoded[:800], flush=True)

            print(
                f"[Huginn-batch-debug] training={self.training} "
                f"autocast={torch.is_autocast_enabled()}",
                flush=True,
            )
            self._huginn_batch_debug_printed = True

        # Generation / eval path: keep original behavior
        if labels is None:
            return original_forward(*args, **kwargs)

        # Reuse Huginn forward logits, but replace only the loss definition
        kwargs_no_labels = dict(kwargs)
        kwargs_no_labels["labels"] = None

        output = original_forward(*args, **kwargs_no_labels)
        logits = output.logits

        if logits is None:
            raise RuntimeError("Huginn forward returned logits=None; cannot recompute shifted loss.")

        if not getattr(self, "_huginn_logits_debug_printed", False):
            print(
                f"[Huginn-logits-debug] logits.shape={tuple(logits.shape)} dtype={logits.dtype}",
                flush=True,
            )
            self._huginn_logits_debug_printed = True

        labels = labels.to(logits.device)

        # Keep logits full-shaped and shift labels instead.
        shift_labels = torch.full_like(labels, -100)
        shift_labels[:, :-1] = labels[:, 1:]

        valid_mask = shift_labels.ne(-100)
        if not valid_mask.any():
            loss = logits.new_tensor(0.0)
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output.loss = loss
        if hasattr(output, "log_ppl"):
            output.log_ppl = loss.detach().clone()

        return output

    model.forward = MethodType(forward_with_shift_loss, model)
    model._huginn_shift_loss_patched = True
    print("[HuginnLoader] applied shift-loss patch for SFT")
    return model


def get_huginn_model_tokenizer(model_dir, model_info, model_kwargs, load_model=True, **kwargs):
    print(f'[HuginnLoader] get_huginn_model_tokenizer: {model_dir}')

    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    print(f'[HuginnLoader] config type = {type(config)}')
    print(f'[HuginnLoader] config.n_embd = {getattr(config, "n_embd", None)}')

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_fast=False,
    )
    print(f'[HuginnLoader] tokenizer type = {type(tokenizer)}')

    model = None
    if load_model:
        # Keep the config explicit, but still respect Swift-passed model_kwargs
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            trust_remote_code=True,
            **model_kwargs,
        )
        print(f'[HuginnLoader] model type = {type(model)}')
        model = patch_huginn_shift_loss(model)
        model._huginn_debug_tokenizer = tokenizer

    return model, tokenizer


register_model(
    ModelMeta(
        model_type='huginn_raven',
        model_groups=[
            ModelGroup([
                Model(
                    model_path=MODEL_DIR,
                    hf_model_id='huginn-0125',
                ),
            ]),
        ],
        template='huginn_text',
        get_function=get_huginn_model_tokenizer,
        architectures=['RavenForCausalLM'],
        requires=['transformers>=4.53.3'],
        tags=['huginn'],
    ),
    exist_ok=True,
)

register_template(
    TemplateMeta(
        template_type='huginn_text',
        prefix=[],
        prompt=[
            '<|begin_header|>user<|end_header|>\n\n'
            '{{QUERY}}'
            '<|end_turn|>'
            '<|begin_header|>Huginn<|end_header|>\n\n'
        ],
        chat_sep=None,
        auto_add_bos=True,
        stop_words=[['eos_token_id']],
    ),
    exist_ok=True,
)