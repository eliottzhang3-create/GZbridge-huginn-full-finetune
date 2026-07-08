"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from swift import get_model_processor, get_template

REPO_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/huginn_pipeline")
MODEL_DIR = "/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125"
DATA_PATH = REPO_DIR / "data/scienceqa/scienceqa_alpaca_huginn_train_sft.jsonl"

sys.path.insert(0, str(REPO_DIR))
import plugins.huginn_swift  # noqa: F401，先注册自定义 model_type 和 template

with open(DATA_PATH, "r", encoding="utf-8") as f:
    sample = json.loads(next(line for line in f if line.strip()))

# 关键：不要自己单独造 tokenizer，改成通过 SWIFT 拿 processor
_, processor = get_model_processor(
    MODEL_DIR,
    model_type="huginn_raven",
    load_model=False,
)


tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

template = get_template(
    template_type="huginn_text",
    processor=processor,
)
template.set_mode("train")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    use_fast=False,
)

template = get_template(
    "huginn_text",
    processor=tokenizer,
)
template.set_mode("train")

encoded = template.encode(sample, return_template_inputs=True)
input_ids = encoded["input_ids"]
labels = encoded["labels"]
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from swift.llm.model.register import get_model_tokenizer
from swift.llm.template.register import get_template

from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/huginn_pipeline")
MODEL_DIR = "/hpc_stor03/sjtu_home/jinwei.zhang/models/huginn-0125"
DATA_PATH = REPO_DIR / "data/scienceqa/scienceqa_alpaca_fullanswer_train_sft.jsonl"

sys.path.insert(0, str(REPO_DIR))
import plugins.huginn_swift_39  # noqa: F401
from plugins.huginn_swift_39 import patch_huginn_shift_loss

with open(DATA_PATH, "r", encoding="utf-8") as f:
    sample = json.loads(next(line for line in f if line.strip()))

model, processor = get_model_tokenizer(
    MODEL_DIR,
    load_model=True,
    model_type="huginn_raven",
)

template = get_template(
    "huginn_text",
    processor=processor,
)
template.set_mode("train")
tokenizer = processor

encoded = template.encode(sample, return_template_inputs=True)
input_ids = encoded["input_ids"]
labels = encoded["labels"]

print("========== RAW SAMPLE ==========")
print(json.dumps(sample, ensure_ascii=False, indent=2))
print("========== DECODED INPUT ==========")
print(template.safe_decode(input_ids))
print("========== DECODED LABELS ==========")
print(template.safe_decode(labels))

sup = [i for i, x in enumerate(labels) if x != -100]
print("========== SUPERVISION RANGE ==========")
print({"total_tokens": len(input_ids), "supervised_tokens": len(sup), "first_supervised": sup[0], "last_supervised": sup[-1]})

print("========== TOKEN ALIGNMENT AROUND FIRST SUPERVISED TOKEN ==========")
start = max(0, sup[0] - 8)
end = min(len(input_ids), sup[0] + 24)
for i in range(start, end):
    inp_tok = tokenizer.convert_ids_to_tokens(int(input_ids[i]))
    lab = labels[i]
    lab_tok = "<-100>" if lab == -100 else tokenizer.convert_ids_to_tokens(int(lab))
    next_tok = tokenizer.convert_ids_to_tokens(int(input_ids[i + 1])) if i + 1 < len(input_ids) else "<END>"
    print(f"{i:04d} | in={inp_tok!r:<18} | label={lab_tok!r:<18} | next_in={next_tok!r}")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    torch_dtype=dtype
).to(device)
model = patch_huginn_shift_loss(model)
model.eval()

batch_input_ids = torch.tensor([input_ids], device=device)
batch_labels = torch.tensor([labels], device=device)

with torch.no_grad():
    out = model(input_ids=batch_input_ids, labels=batch_labels)

logits = out.logits.float()
loss_noshift = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    batch_labels.view(-1),
    ignore_index=-100
)
loss_shift = F.cross_entropy(
    logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
    batch_labels[:, 1:].contiguous().view(-1),
    ignore_index=-100
)

print("========== LOSS CHECK ==========")
print({
    "model_loss": float(out.loss),
    "manual_noshift_loss": float(loss_noshift),
    "manual_shift_loss": float(loss_shift),
})
