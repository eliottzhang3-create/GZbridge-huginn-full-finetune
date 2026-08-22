# Ouro_audio 项目交接说明

更新时间：2026-08-22

本文件是 code/Ouro_audio 的详细交接入口。新的 Codex/agent 必须先阅读仓库根目录 README.md，了解本仓库是本地 Windows 到远程 HPC 的代码同步 workspace，再阅读本文件掌握当前实验边界、脚本、路径、训练配置、评测状态和已知问题。所有任务必须以远程实际代码、日志、progress JSON、checkpoint state 和最终 report 为准。

本次只修改本文件，不修改仓库根目录 README.md。

---

## 1. 项目定位和边界

根仓库工作方式：

~~~text
本地 Windows 编辑代码
  -> git commit / git push
  -> 远程 Linux git pull --ff-only
  -> vc submit 提交 GPU 任务
~~~

本仓库不是运行时数据盘，不应提交模型权重、checkpoint、大型输出、缓存、大型日志或远程临时文件。

当前 Ouro_audio 主线：

~~~text
BAT + Ouro-1.4B
BAT + Qwen3-4B Base（语言模型对照）
~~~

不要混入：

- Huginn/Whisper 主线；
- HRM-Text 音频线；
- OWL/SAGE 线；
- BAT/model.pt 中打包的 Q-Former/LLM 权重；
- 旧固定 176 token compile 路线。

OWL 曾做过数据和模型审查，但当前已经切换到 BAT。不要把 OWL Stage-I/II/III、SAGE checkpoint 或 OWL Q-Former checkpoint 用于 BAT。

---

## 2. 当前主线模型契约

### 2.1 BAT + Ouro

~~~text
AudioSet + MP3D 双耳 RIR
  -> 双耳波形
  -> Spatial-AST（FP32、冻结）[B,515,768]
  -> 随机初始化 BAT Q-Former（训练）[B,64,2048]
  -> 替换 Ouro 输入前缀的 64 个音频占位位置
  -> Ouro 24 个物理层 recurrent loop 4 次
     native backbone 冻结
     q_proj/v_proj LoRA 训练
     early-exit gate 冻结，threshold=1.0
  -> assistant response-only causal next-token loss
~~~

### 2.2 BAT + Qwen3-4B Base

~~~text
AudioSet + RIR -> 双耳波形
  -> Spatial-AST FP32 冻结 [B,515,768]
  -> 随机 Q-Former [B,64,2560]
  -> Qwen3-4B Base 36-layer Transformer
  -> q_proj/v_proj LoRA
~~~

Qwen3 native backbone、embedding、norm、lm_head 冻结，只训练 Q-Former 和 LoRA。

### 2.3 重要训练结论

当前正式 Stage-III 两条线都：

~~~text
compile                    = false
fixed sequence length     = false
dynamic batch padding      = true
max_length safety ceiling  = 512
dataloader workers         = 0
local Arrow cache          = enabled
runtime monitor            = enabled
~~~

旧的 64 audio + 112 text = 176 固定方案只属于已完成的历史 smoke/compile 测试，不是当前正式训练配置。

---

## 3. 本地、远程、环境和资源

### 3.1 路径

本地：

~~~text
C:/Users/69327/Documents/huginn-full-finetune-sync
C:/Users/69327/Documents/huginn-full-finetune-sync/code/Ouro_audio
~~~

远程：

~~~text
仓库：
/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune

Ouro_audio：
/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/Ouro_audio
~~~

远程同步：

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
git pull --ff-only
conda activate swift_ouro
python -m pip check
~~~

没有 commit/push 的本地代码不会出现在远程。禁止用 git reset --hard、git checkout -- 等破坏性命令覆盖远程用户改动。

### 3.2 模型和数据

~~~text
Ouro:
/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B

Qwen3:
/hpc_stor03/sjtu_home/jinwei.zhang/models/Qwen3-4B-Base

BAT:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA

BAT closed-end:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/closed-end

MP3D RIR:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb

Spatial-AST source:
/hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST

Spatial-AST checkpoint:
/hpc_stor03/sjtu_home/jinwei.zhang/models/BAT/SpatialAST/finetuned.pth

AudioSet（公共只读）:
/hpc_stor03/public/shared/data/raa/AudioSet

私有 manifest:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests

私有 cache:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/cache

私有输出:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro
~~~

公共 AudioSet 只有只读权限，任何输出都不能写入公共目录。

### 3.3 远程环境

~~~text
conda env       swift_ouro
Python          3.10.20
torch           2.11.0+cu128
transformers    4.54.1
ms-swift        4.4.2
peft            0.18.1
accelerate      1.13.0
timm            1.0.28
librosa         0.11.0
~~~

GPU 必须通过队列提交，不能在登录节点直接运行 GPU Python。当前正式资源：

~~~text
Ouro   : pdgpu-5090, 8 GPU, 32 CPU, 256G
Qwen3  : pdgpu-3090, 8 GPU, 32 CPU, 256G
~~~

单 GPU 任务 CPU 不超过 8、内存不超过 32G。日志通常位于：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/Ouro_audio/bat/log
~~~

---

## 4. BAT 数据和 manifest

训练源：

~~~text
closed-end/stage1-clsdoa/train.json
closed-end/stage2-single/train.json
closed-end/stage3-mixup/train.json
~~~

去重 union：

~~~text
raw QA records       = 1,665,880
unique QA records    = 872,312
unique source tuples = 872,193

A = 139,392  CLASSIFICATION
B = 139,392  DOA
C = 118,000  MIXUP_SINGLE_CLASSIFICATION
D = 118,000  MIXUP_SINGLE_DOA
E = 357,528  binary/non-binary reasoning
~~~

阶段定义：

~~~text
Stage-I   = A+B
Stage-II  = A+B+C+D
Stage-III = A+B+C+D+E
~~~

question_id 不是全局唯一键，去重必须结合 question/type/content/source tuple。

### 当前 Stage-III manifest

~~~text
manifest:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage3_ab_cde_2epoch.jsonl

report:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/stage3_ab_cde_2epoch_report.json
~~~

manifest 共 1,744,640 条，四个有序 block：

~~~text
epoch 1: A+B -> C+D+E
epoch 2: A+B -> C+D+E
~~~

当前正式脚本只跑第一个 epoch：

~~~text
872,320 / global batch 64 = 13,630 optimizer steps
~~~

生成/审计脚本：

~~~text
bat/scripts/compose_bat_stage3_ab_cde_manifest.py
bat/scripts/audit_bat_stage3_ab_cde_manifest.py
bat/run_audit_bat_stage3_ab_cde_manifest_3090.sh
~~~

训练 runtime shuffle 关闭，顺序由 manifest block 决定。

官方 eval 文件不是 JSONL，而是顶层对象中的 data 数组。正确查看方式：

~~~bash
jq '.data[32]' /path/to/eval.json
~~~

不要使用 jq '.[32]'，也不要把 sed 的格式化 JSON 物理行交给 jq。

---

## 5. 音频、Spatial-AST、Q-Former

实现：

~~~text
bat/models/spatial_ast_audio.py
bat/eval_contract.py
plugins/ouro_bat_spatial_ast_swift.py
plugins/qwen3_bat_spatial_ast_swift.py
~~~

训练 renderer 流程：

1. 从 AudioSet 读取源音频，多通道取第一通道；
2. 重采样到 32 kHz；
3. 约 -14 dBFS RMS normalization；
4. 读取双耳 RIR [2,L]；
5. RIR crop/zero-pad 到 2 秒，即 64,000 samples；
6. 每个耳道用 scipy.signal.fftconvolve；
7. 输出 crop/zero-pad 到 10 秒，即 [2,320000]；
8. 双源分别渲染后逐元素平均：

~~~python
rendered = (rendered_1 + rendered_2) / 2.0
~~~

这是双耳波形逐元素平均，不是左右耳平均，也不是未经缩放的直接相加。当前有音频 RMS normalization，但没有额外的后 RIR/后混合 LUFS normalization。

Spatial-AST 内部使用双耳 log-mel 和 IPD sin/cos 构造四通道输入，输出：

~~~text
[B,515,768] FP32
~~~

训练绕过分类头，只取 token。Spatial-AST 始终 eval、FP32、冻结。

Q-Former：

~~~text
Ouro  : [B,515,768] -> [B,64,2048]
Qwen3 : [B,515,768] -> [B,64,2560]
~~~

Q-Former 随机初始化，不加载 BAT/model.pt、OWL checkpoint 或其他预训练 Q-Former。

注意评测 renderer 的 RIR policy 与训练 renderer 需要单独核对：

~~~text
training renderer: 固定 2 秒 crop/zero-pad
eval official_bat: 维持官方 eval 原始 policy
eval checkpoint_matched: 2 秒 crop/zero-pad
~~~

在线 eval 默认是 official_bat。若要求严格复用训练 renderer，显式设置 BAT_EVAL_RIR_POLICY=checkpoint_matched，并在 report 中记录。

---

## 6. Ouro 结构、gate 和 cache

Ouro 关键配置：

~~~text
model_type       = ouro
hidden_size      = 2048
physical layers  = 24
total_ut_steps   = 4
vocab size       = 49152
~~~

24 个物理层被 recurrent loop 执行 4 次，不是 96 个独立层。

训练：

~~~text
use_cache = false
early_exit_threshold = 1.0
gate = frozen
Ouro native parameters = frozen
embeddings/norm/lm_head = frozen
~~~

LoRA：

~~~text
target = q_proj, v_proj
rank = 8
alpha = 32
dropout = 0.05
~~~

### Ouro cache 兼容层

~~~text
code/Ouro_audio/compat/ouro_cache.py
~~~

当前 Transformers 4.54.1 与远程 UniversalTransformerCache 有 API 冲突，兼容层运行时 patch，不修改远程 modeling_ouro.py。

官方 recurrent cache 按 loop/layer 分槽：

~~~text
logical slots = total_ut_steps * num_hidden_layers = 4*24 = 96
slot = current_ut * num_hidden_layers + layer_idx
~~~

因此不是只保存最后一轮的 cache。推理 use_cache=true 使用 loop-indexed cache；训练 use_cache=false。

曾尝试只 compile Ouro recurrent core，LoRA q/v 进入 core 图，lm_head 不单独 compile；但出现 nvcc/Inductor、segmentation fault 和 DDP 首次编译时间过长等问题。当前正式训练 compile 已关闭。

---

## 7. 当前正式 Stage-III 训练配置

### Ouro

~~~text
wrapper:
bat/run_train_bat_ouro_stage3_ab_cde_5090.sh

remote:
bat/scripts/train_bat_ouro_stage3_ab_cde_remote.sh
bat/scripts/train_bat_ouro_stage3_ab_cde.py

queue = pdgpu-5090
8 GPU, 32 CPU, 256G
~~~

### Qwen3

~~~text
wrapper:
bat/run_train_qwen3_bat_stage3_ab_cde_3090.sh

remote:
bat/scripts/train_qwen3_bat_stage3_ab_cde_remote.sh
bat/scripts/train_qwen3_bat_stage3_ab_cde.py

queue = pdgpu-3090
8 GPU, 32 CPU, 256G
~~~

两条线共同配置：

~~~text
manifest              = stage3_ab_cde_2epoch.jsonl
actual epoch           = first epoch only
actual steps           = 13,630
per-device batch       = 8
global batch           = 64
gradient accumulation  = 1
learning rate          = 0.002
optimizer              = AdamW
betas                  = (0.9, 0.95)
weight decay           = 0.05
scheduler              = cosine / half-cycle cosine
warmup                 = ceil(steps*0.13)
compile                = disabled
fixed sequence         = disabled
dynamic batch padding  = enabled
max_length             = 512 safety ceiling
workers                = 0
pin_memory             = false
dataset shuffle        = false
train loader shuffle   = false
save_steps             = 3000
save_total_limit       = 2
checkpoint             = full resumable
~~~

512 不是把所有样本 pad 到 512；collator 只 pad 到当前 batch 的最长自然序列。

提交模板：

~~~bash
BAT_STAGE3_AB_CDE_MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage3_ab_cde_2epoch.jsonl \
BAT_STAGE3_AB_CDE_REPORT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/stage3_ab_cde_2epoch_report.json \
BAT_STAGE3_AB_CDE_OUTPUT_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/<new-private-run> \
bash code/Ouro_audio/bat/run_train_bat_ouro_stage3_ab_cde_5090.sh
~~~

~~~bash
QWEN3_BAT_STAGE3_AB_CDE_MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage3_ab_cde_2epoch.jsonl \
QWEN3_BAT_STAGE3_AB_CDE_REPORT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/stage3_ab_cde_2epoch_report.json \
QWEN3_BAT_STAGE3_AB_CDE_OUTPUT_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/qwen3/<new-private-run> \
bash code/Ouro_audio/bat/run_train_qwen3_bat_stage3_ab_cde_3090.sh
~~~

新鲜 output 目录必须为空。resume 必须显式指定同一路线的合法 checkpoint，不能跨 curriculum/Stage-III 路线 resume。

---

## 8. 动态 padding、audio prefix 和 loss

每条样本先形成：

~~~text
64 个音频占位 ID + 自然文本 token
~~~

再由 collator 进行 batch 内 padding。旧固定 176 仅是历史 smoke。

日志中显示的 pad/eos 或 <|endoftext|> 是占位 ID 的文本化显示。真正 forward 时：

~~~text
Ouro  [B,64,2048]
Qwen3  [B,64,2560]
~~~

连续 embedding 替换这 64 个位置。

loss 是：

~~~text
logits[:, :-1] 对 labels[:, 1:]
~~~

labels 必须满足：

~~~text
system prompt       = -100
user prompt         = -100
audio prefix        = -100
batch padding       = -100
assistant response  = valid targets
~~~

当前正式脚本传入 loss_scale=default、is_binary_loss_scale=true。动态 padding/resume smoke 检查 assistant-only label span、audio/padding mask、shifted CE、finite logits/loss、训练参数更新和冻结参数不变。

---

## 9. Local Arrow cache 和稳定性

正式脚本先运行 bat/scripts/prewarm_bat_arrow_cache.py，再把：

~~~text
HF_DATASETS_CACHE = /tmp/<job-root>/datasets
MODELSCOPE_CACHE  = /tmp/<job-root>/modelscope
~~~

指向 job-local 路径，完成后才启动 torchrun。

相关代码：

~~~text
bat/cache_contract.py
bat/runtime_monitor.py
bat/scripts/prewarm_bat_arrow_cache.py
~~~

runtime monitor 记录 global step、RSS、GPU memory、文件系统、cache 和进程状态。

Local Arrow cache 主要解决云盘 JSON/Arrow metadata、mmap 和多进程 cache creation 竞争；它不等于预缓存 AudioSet、RIR 或 Spatial-AST feature。AudioSet/RIR/fftconvolve/Spatial-AST 仍在线处理。

历史 BF16 Spatial-AST feature cache 曾出现 16 路任务掉线、存储量大、coverage/index 审计复杂等问题，当前正式训练不依赖 feature cache。

---

## 10. 已知 checkpoint 和最新进度

曾用于评测的 Ouro checkpoint：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/stage3_ab_cde_localcache_0819_v2/v0-20260819-120617/checkpoint-10500
~~~

曾用于评测的 Qwen3 checkpoint：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/qwen3/stage3_ab_cde_localcache_0819_v2/v0-20260819-150938/checkpoint-10500
~~~

具体训练历史必须以 run 目录内 trainer_state.json、optimizer.pt、scheduler.pt、日志和 prewarm report 为准。README 不替代 checkpoint audit。

当前不能仅凭 README 宣称三阶段完成、Stage-III 收敛或全部 eval 完成。

---

## 11. Eval 实现和当前异常

### Eval 入口

~~~text
bat/scripts/audit_bat_eval_contract.py
bat/run_audit_bat_eval_contract_3090.sh
bat/scripts/eval_bat_ouro_online.py
bat/scripts/eval_bat_ouro_online_remote.sh
bat/run_eval_bat_ouro_online_3090.sh
bat/scripts/smoke_bat_eval_generation.py
bat/run_smoke_bat_eval_generation_3090.sh
bat/run_generate_bat_eval_samples_3090.sh
~~~

Phase-I contract audit 是 metadata-only：不 decode、不加载 RIR、不卷积、不导入 Spatial-AST、不加载模型。Phase-II/online eval 才执行真实 renderer、Spatial-AST、Q-Former 和 generation。

评测集合：

~~~text
A             stage1-clsdoa/eval-stage1-classification.json
B             stage1-clsdoa/eval-stage1-doa.json
C             stage2-single/eval-stage2-classification.json
D             stage2-single/eval-stage2-doa.json
E-direction   stage3-mixup/eval-stage3-direction.json
E-distance    stage3-mixup/eval-stage3-distance.json
E-nonbinary   stage3-mixup/eval-stage3-nonbinary.json
~~~

指标：

~~~text
A/C            Detection mAP
B/D            DoA / DP
E-direction    direction accuracy
E-distance     distance accuracy
E-nonbinary    diagnostic only
~~~

A/C 当前 model_output_embedding 模式：

1. ground truth 按 AudioSet label 构成 355 类 multi-hot；
2. 生成答案 tokenization；
3. 取模型 lm_head/output embedding token rows；
4. mean pooling + L2 normalize；
5. 355 个类别名称在同一模型 token-row 空间编码；
6. cosine similarity 作为各类分数；
7. 各类 AP 的 mean 为 Detection mAP。

需要：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/class_labels_indices_subset.csv
~~~

当前模式不依赖 OpenAI API，也不需要 audioset_class_embeds.npy。

generation：

~~~text
use_cache      = true
do_sample      = false
num_beams      = 1
max_new_tokens = 10
~~~

max_new_tokens 只限制回答阶段新 token，不包括 prompt/audio prefix。旧 200 token、beam=4 已移除。

### 最近异常

Ouro checkpoint-10500 的 B/D/E-direction/E-distance eval 中，控制台只看到 progress 1，但 progress JSON 实际显示：

~~~text
B             completed = 32
D             completed = 4
E-direction   completed = 37
E-distance    completed = 5
~~~

控制台只在第 1 条和每 100 条打印，不代表只处理 1 条。

progress 同时显示 errors=0、aborted_on_cuda_oom=false、first_cuda_oom=null、status=running，说明没有被 Python try/except 捕获的 OOM。更可能是 native SIGSEGV/SIGBUS、底层 CUDA/音频库异常或外部终止。已检查中断样本，字段和 source pair 正常，不能先认定数据坏了。

下一步应做逐记录 heartbeat：

~~~text
render-start -> render-done
generate-start -> generate-done
cleanup-start -> cleanup-done
GPU allocated/reserved
~~~

还应保留 faulthandler、作业退出码和系统日志。普通 Python 无法捕获 segmentation fault。

当前 eval 已显式释放 output_ids、generated_ids、waveform、临时 input，执行 gc.collect() 和 torch.cuda.empty_cache()，首个 Python CUDA OOM 立即停止并保留 progress；这些措施不能捕获 native crash。

---

## 12. 已通过的审计和 smoke

模型/多模态：

- Ouro native load/inference；
- Ouro ms-swift text registration；
- Qwen3 Base native load/generation；
- Qwen3 ms-swift text registration；
- Spatial-AST strict checkpoint load；
- Spatial-AST FP32/frozen；
- random Q-Former structure；
- Spatial-AST -> Q-Former -> Ouro/Qwen3 forward/backward；
- audio prefix replacement；
- Ouro 4 recurrent loop；
- LoRA target/rank/frozen backbone。

训练/resume：

- Ouro 单卡 LoRA/Q-Former smoke；
- Ouro 8 卡 DDP、preflight、checkpoint、resume；
- curriculum callback/stage marker；
- Stage-III manifest order/digest；
- Qwen3 单卡和 8 卡 DDP；
- Qwen3 DDP resume；
- shifted next-token CE；
- dynamic padding/assistant-only label/resume smoke 已加入，最新远程代码需重新确认。

性能/稳定性：

- DataLoader workers 0/2/4/8；
- local Arrow cold/warm audit；
- renderer 256 条单进程；
- renderer 8 进程小测试；
- NCCL all-reduce；
- 固定波形 Ouro DDP；
- SDPA backend；
- Ouro recurrent compile graph reuse（历史，正式训练关闭）。

---

## 13. 历史问题

1. Ouro cache API 冲突：compat/ouro_cache.py 运行时 patch。
2. early-exit gate backward 次数错误：目标是 gate 冻结、threshold=1.0、完整 4 loop。
3. Trainer CE mismatch：修复 audio/padding mask、causal shift、DDP denominator 和 Swift loss reproduction。
4. DDP RNG：兼容 rng_state_0.pth 到 rng_state_7.pth。
5. Swift checkpoint 嵌套在 v0-<timestamp>：audit 按实际层级找 checkpoint。
6. 重复 Q-Former 审计误报：按 module ownership 区分。
7. 固定 176/compile：历史 smoke，当前正式禁用。
8. SIGBUS/SIGSEGV：workers=0、local Arrow prewarm、faulthandler、runtime monitor。
9. BF16 Spatial-AST cache：掉线、存储和 coverage 风险，当前不依赖。
10. eval 长生成：统一 greedy single beam、10 new tokens、逐条释放、首个 Python OOM 停止。

---

## 14. 新任务标准清单

代码修改后：

~~~bash
git diff --check
python -m py_compile <changed_python_files>
git status --short
git add <only-intended-files>
git commit -m "<message>"
git push
~~~

远程先：

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
git pull --ff-only
~~~

正式训练前确认：

- queue/resource；
- model/plugin/manifest/report/output；
- output 是否为空；
- BAT_FIXED_SEQUENCE_LENGTH=false；
- torch_compile=false；
- max_length=512 safety ceiling；
- local Arrow/ModelScope cache；
- workers=0；
- runtime monitor；
- Spatial-AST FP32/frozen；
- Q-Former random/trainable；
- native backbone frozen；
- LoRA target/rank/alpha/dropout；
- Ouro gate frozen、4 loops；
- use_cache=false；
- assistant-only loss；
- save_steps/save_total_limit；
- checkpoint 可完整 resume。

训练结束检查：

- checkpoint 数量和 retention；
- adapter_model.safetensors；
- adapter_config.json；
- optimizer.pt；
- scheduler.pt；
- trainer_state.json；
- training_args.bin；
- 8 个 rank RNG；
- global/scheduler step；
- prewarm report；
- runtime monitor；
- resume audit；
- final eval report。

单个 status=ok 只表示对应报告覆盖范围通过，不代表训练收敛或最终评测成功。

---

## 15. 关键文件速查

~~~text
模型注册：
  plugins/ouro_bat_spatial_ast_swift.py
  plugins/qwen3_bat_spatial_ast_swift.py
  plugins/ouro_text_swift.py
  plugins/qwen3_text_swift.py

Ouro cache：
  compat/ouro_cache.py

音频/Q-Former：
  bat/models/spatial_ast_audio.py
  bat/eval_contract.py

正式训练：
  bat/run_train_bat_ouro_stage3_ab_cde_5090.sh
  bat/scripts/train_bat_ouro_stage3_ab_cde_remote.sh
  bat/scripts/train_bat_ouro_stage3_ab_cde.py
  bat/run_train_qwen3_bat_stage3_ab_cde_3090.sh
  bat/scripts/train_qwen3_bat_stage3_ab_cde_remote.sh
  bat/scripts/train_qwen3_bat_stage3_ab_cde.py

manifest/cache：
  bat/scripts/compose_bat_stage3_ab_cde_manifest.py
  bat/scripts/audit_bat_stage3_ab_cde_manifest.py
  bat/scripts/prewarm_bat_arrow_cache.py
  bat/cache_contract.py
  bat/runtime_monitor.py

smoke/resume：
  bat/scripts/smoke_bat_ouro_stage3_ab_cde_resume.py
  bat/scripts/smoke_qwen3_bat_ddp.py
  bat/run_smoke_bat_ouro_stage3_ab_cde_resume_5090.sh
  bat/run_smoke_qwen3_bat_ddp_resume_3090.sh

eval：
  bat/scripts/audit_bat_eval_contract.py
  bat/scripts/eval_bat_ouro_online.py
  bat/scripts/smoke_bat_eval_generation.py
  bat/run_eval_bat_ouro_online_3090.sh
  bat/run_smoke_bat_eval_generation_3090.sh

稳定性：
  bat/scripts/profile_ouro_bat_dataloader.py
  bat/scripts/audit_bat_renderer_processes.py
  bat/scripts/test_bat_nccl_allreduce.py
  bat/scripts/test_bat_fixed_waveform_ddp.py
~~~

---

## 16. 最短接手上下文

1. 当前主线是 BAT + Ouro-1.4B，Qwen3-4B Base 是对照；不是 OWL、不是 Huginn/Whisper。
2. Spatial-AST FP32 冻结；Q-Former 随机初始化且训练；语言模型 native backbone 冻结；只训练 q/v LoRA 和 Q-Former。
3. Ouro 24 个物理层循环 4 次；gate 冻结、threshold=1.0；训练 use_cache=false。
4. AudioSet -> 32 kHz/RMS -> RIR 2 秒 crop/pad -> 双耳 fft 卷积 -> 10 秒双耳波形 -> Spatial-AST。
5. 当前 Stage-III manifest 是 stage3_ab_cde_2epoch.jsonl，正式脚本只跑第一个 epoch，13,630 steps，per-device batch=8，global batch=64。
6. 当前正式训练关闭 compile 和固定 176，使用动态 batch padding、max_length=512 ceiling、workers=0。
7. Ouro 队列 pdgpu-5090；Qwen3 队列 pdgpu-3090；均为 8 GPU、32 CPU、256G。
8. 两条正式线都使用 job-local /tmp Arrow/ModelScope cache、runtime monitor、每 3000 steps 保存、最多 2 个 checkpoint。
9. Eval 使用 max_new_tokens=10、num_beams=1、use_cache=true；A/C 用模型 lm_head token-row embedding 做 Detection mAP。
10. 最近 B/D/E eval 进程在少量记录后消失，progress 显示没有 Python OOM；数据字段正常，下一步做逐记录 renderer/generate/native crash 诊断。
11. 新 agent 必须先 git pull --ff-only，再核对实际脚本和 report，不要复用旧命令或旧结论。
