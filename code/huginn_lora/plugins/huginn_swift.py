from types import MethodType

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model
from swift.template import TemplateMeta, register_template

MODEL_DIR = '/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125'


def patch_huginn_shift_loss(model):
    if getattr(model, '_huginn_shift_loss_patched', False):
        print('[HuginnLoader] shift-loss patch already applied')
        return model

    original_forward = model.forward

    def forward_with_shift_loss(self, *args, **kwargs):
        labels = kwargs.get('labels', None)

        if not getattr(self, '_huginn_batch_debug_printed', False):
            print('[Huginn-batch-debug] num_args =', len(args), flush=True)
            print('[Huginn-batch-debug] keys =', sorted(kwargs.keys()), flush=True)
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    print(
                        f'[Huginn-batch-debug] {k}: '
                        f'shape={tuple(v.shape)} dtype={v.dtype} device={v.device}',
                        flush=True,
                    )
                else:
                    print(
                        f'[Huginn-batch-debug] {k}: type={type(v)} value={v}',
                        flush=True,
                    )
            print(
                f'[Huginn-batch-debug] training={self.training} '
                f'autocast={torch.is_autocast_enabled()}',
                flush=True,
            )
            self._huginn_batch_debug_printed = True

        if labels is None:
            return original_forward(*args, **kwargs)

        kwargs_no_labels = dict(kwargs)
        kwargs_no_labels['labels'] = None

        output = original_forward(*args, **kwargs_no_labels)
        logits = output.logits

        if not getattr(self, '_huginn_logits_debug_printed', False):
            if logits is not None:
                print(
                    f'[Huginn-logits-debug] logits.shape={tuple(logits.shape)} dtype={logits.dtype}',
                    flush=True,
                )
            else:
                print('[Huginn-logits-debug] logits is None', flush=True)
            self._huginn_logits_debug_printed = True

        if logits is None:
            raise RuntimeError(
                'Huginn forward returned logits=None; cannot recompute shifted loss.'
            )

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous().to(logits.device)

        valid_mask = shift_labels.ne(-100)
        if not valid_mask.any():
            loss = logits.new_tensor(0.0)
        else:
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output.loss = loss
        if hasattr(output, 'log_ppl'):
            output.log_ppl = loss.detach().clone()

        return output

    model.forward = MethodType(forward_with_shift_loss, model)
    model._huginn_shift_loss_patched = True
    print('[HuginnLoader] applied shift-loss patch for SFT')
    return model


class HuginnLoader(ModelLoader):
    def get_config(self, model_dir: str):
        print(f'[HuginnLoader] get_config: {model_dir}')
        config = AutoConfig.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )
        print(f'[HuginnLoader] config type = {type(config)}')
        print(f'[HuginnLoader] config.n_embd = {getattr(config, "n_embd", None)}')
        return config

    def get_processor(self, model_dir: str, config):
        print(f'[HuginnLoader] get_processor: {model_dir}')
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_fast=False,
        )
        print(f'[HuginnLoader] tokenizer type = {type(tokenizer)}')
        return tokenizer

    def get_model(self, model_dir: str, config, processor, model_kwargs):
        print(f'[HuginnLoader] get_model: {model_dir}')

        config = AutoConfig.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )
        print(f'[HuginnLoader] reloaded config.n_embd = {getattr(config, "n_embd", None)}')

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            trust_remote_code=True,
            **model_kwargs,
        )
        print(f'[HuginnLoader] model type = {type(model)}')

        model = patch_huginn_shift_loss(model)
        return model


register_model(
    ModelMeta(
        'huginn_raven',
        [
            ModelGroup([
                Model('huginn-0125', MODEL_DIR),
            ]),
        ],
        HuginnLoader,
        template='huginn_text',
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
