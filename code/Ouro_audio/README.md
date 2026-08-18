# Ouro Audio 分支说明

更新时间：2026-08-17

本 README 是 Ouro_audio 分支的工作入口。新加入的 agent 应先阅读本文件，再阅读仓库根目录的 README.md，最后根据本文列出的脚本和远程路径核对实际代码与任务状态。

> 重要：本分支不负责 Huginn 主线，也不负责 OWL/SAGE 主线。当前有效工作是：在 ms-swift 中使用 Ouro-1.4B，接入 BAT 论文中的 Spatial-AST 音频编码链路和随机初始化 Q-Former，并训练 Q-Former + Ouro LoRA；Spatial-AST、Ouro 主体和 Ouro gate 均冻结。

---

## 1. 项目定位与当前状态

### 当前主线：BAT + Ouro-1.4B

目标是复现 BAT（Learning to Reason about Spatial Sounds with Large Language Models）的语言模型课程学习思路，但将 BAT 原始语言模型替换为 Ouro-1.4B：

~~~text
AudioSet 音频
  + MP3D RIR
  + 双耳渲染
      ↓
Spatial-AST（冻结，FP32）
      ↓  [B, 515, 768]
随机初始化 BAT Q-Former（训练）
      ↓  [B, 64, 2048]
替换 Ouro 输入中的 64 个音频占位 token
      ↓
Ouro-1.4B recurrent Transformer（主体冻结）
  ├─ Ouro LoRA：训练
  └─ early_exit_gate：冻结，early_exit_threshold=1.0，始终完整执行 4 次循环
      ↓
next-token prediction / causal CE
~~~

截至 2026-08-17：

- Ouro-1.4B 原生加载、文本推理和 ms-swift 文本推理已通过。
- Ouro 的自定义 Universal Transformer cache 与当前 Transformers 版本的兼容问题已通过运行时兼容层解决。
- Spatial-AST、Q-Former、音频渲染、Ouro 多模态前向/反向链路已通过审计。
- 单卡和 8 卡 DDP 的短程训练、梯度、LoRA/Q-Former 可训练性、CE 错位、checkpoint 保存和 resume 已完成多轮 smoke/preflight。
- curriculum callback、阶段边界 checkpoint、固定长度 padding/truncation 和 compile resume smoke 已通过。
- 三阶段正式 BAT curriculum 训练已经提交并运行过；正式训练的最终任务指标仍需以实际输出 checkpoint 和评测结果确认，不能仅凭启动日志宣布收敛。
- 当前正式路线使用在线音频处理；Spatial-AST BF16 特征预缓存曾做过实验，但由于存储量、16 路任务掉线和 cache 覆盖审计问题，目前不是正式训练的前置条件。

### 历史/非当前路线

- Huginn + Whisper 音频主线：由其他 agent/聊天负责，本分支不修改。
- OWL + SAGE：已进行过数据和模型审查，但由于 OWL 数据组织与论文课程定义存在较大歧义，当前 BAT 路线已替代它；不要把 OWL 的 Stage-I/II/III、SAGE 或 OWL Q-Former checkpoint 混入当前 BAT 训练。
- Ouro 纯文本 LoRA smoke：已完成，属于模型注册和 LoRA 机制的历史验证，不是当前音频训练配置。
- BAT/model.pt：当前不加载。Q-Former 按要求随机初始化，不读取 BAT 打包模型中的 Q-Former/LLM 权重。

---

## 2. 本地、远程和 GitHub 工作方式

### 本地 Windows 工作区

~~~text
C:/Users/69327/Documents/huginn-full-finetune-sync
C:/Users/69327/Documents/huginn-full-finetune-sync/code/Ouro_audio
~~~

本地通常只有核心代码和配置；Ouro 权重、Spatial-AST 权重、AudioSet、RIR、训练 manifest、缓存和训练输出在远程 Linux 服务器。不要假设本地可以直接运行完整 GPU 训练。

### 远程 Linux 工作区

~~~text
远程仓库：
/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune

远程 Ouro 分支代码：
/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/Ouro_audio
~~~

代码通过 GitHub 中转。远程执行代码前通常：

~~~bash
cd /hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune
git pull --ff-only
~~~

如果本地修改没有提交并推送到 GitHub，远程不会自动拥有修改。远程代码目录中的未提交用户改动必须保留，禁止用破坏性命令覆盖。

### GPU 提交规则

任何使用 GPU 的任务都必须提交到队列，不能在登录节点直接运行。当前常用提交方式是 vc submit，脚本中已经封装好提交参数。

已使用过的队列：

- pdgpu-3090
- pdgpu-4090
- pdgpu-5090

当前资源约束：

- 单 GPU 任务的 CPU 核数不能超过 8。
- 单 GPU 任务的内存不能超过 32G。
- 8 卡任务应明确检查脚本中的 -g8、CPU 和内存是否符合集群限制。
- 脚本文件名中的队列名不一定可靠，必须以脚本实际的 vc submit -p ... 为准。

登录节点只用于：

- git pull、查看文件和日志；
- jq、find、wc 等低成本检查；
- 不使用 GPU 的 manifest/报告检查。

GPU 任务产生的日志一般在：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/code/GZbridge-huginn-full-finetune/code/Ouro_audio/bat/log
~~~

### 远程数据、模型和输出目录

~~~text
Ouro 权重:
/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B

BAT 数据:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA

BAT closed-end QA:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/closed-end

解压后的 MP3D RIR:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb

Spatial-AST 代码:
/hpc_stor03/sjtu_home/jinwei.zhang/code/Spatial-AST

Spatial-AST 权重:
/hpc_stor03/sjtu_home/jinwei.zhang/models/BAT/SpatialAST/finetuned.pth

AudioSet（只读公共目录）:
/hpc_stor03/public/shared/data/raa/AudioSet

私有 manifest:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests

私有音频特征缓存:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/cache

私有训练输出:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro
~~~

公共 AudioSet 目录只有只读权限。任何 manifest、缓存、报告、checkpoint 或临时文件都不能写入 /hpc_stor03/public/shared/data/raa/AudioSet。

---

## 3. 环境

主要远程环境：

~~~bash
conda activate swift_ouro
~~~

已核验的关键版本：

~~~text
Python       3.10.20
torch        2.11.0+cu128
torchaudio   2.11.0+cu128
torchvision  0.26.0+cu128
transformers 4.54.1
ms-swift     4.4.2
peft         0.18.1
trl          0.18.0
accelerate   1.13.0
timm         1.0.28
librosa      0.11.0
ffmpeg       /usr/bin/ffmpeg
~~~

基本检查：

~~~bash
python -m pip check
python -c "import torch, transformers, swift, peft, timm, librosa; print(torch.__version__, transformers.__version__)"
~~~

Spatial-AST 官方代码依赖 timm、librosa 等。它们不是可以永久忽略的“兼容层问题”，必须安装在远程训练环境中；兼容层只用于处理少量 API/版本差异。

---

## 4. Ouro-1.4B 结构和加载契约

模型目录：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/models/Ouro-1.4B
~~~

使用 Hugging Face 仓库中的远程代码和本地权重：

- config.json
- configuration_ouro.py
- modeling_ouro.py
- model.safetensors
- tokenizer 文件

固定 revision：

~~~text
574fa66cb8bf5abdc979642d01cf2b79b16bfab1
~~~

当前关键配置：

~~~json
{
  "model_type": "ouro",
  "architectures": ["OuroForCausalLM"],
  "hidden_size": 2048,
  "intermediate_size": 5632,
  "num_hidden_layers": 24,
  "total_ut_steps": 4,
  "vocab_size": 49152
}
~~~

解释：

- num_hidden_layers=24 是 24 个物理 Transformer 层。
- total_ut_steps=4 表示同一组 24 层循环执行 4 次，不是 96 个独立层。
- 当前 BAT 训练固定 4 次循环。
- early_exit_threshold=1.0，gate 冻结，训练时不提前退出，完整执行四轮。
- 训练使用 use_cache=false。训练反向传播不使用推理 KV cache sharing。
- 推理时使用的 KV cache 是 Ouro 自定义 cache，不等同于普通 Llama cache。

### Cache 兼容层

当前 Transformers 4.54.1 下，Ouro 远程 UniversalTransformerCache 曾出现：

- key_cache 属性冲突；
- cache 层列表为空导致 get_mask_sizes 越界。

解决方式是运行时将 Ouro 远程模块中的 cache 类替换为：

~~~text
code/Ouro_audio/compat/ouro_cache.py
~~~

这不是修改 Hugging Face 模型目录，也不需要把 modeling_ouro.py 复制到 Git 仓库。加载模型时仍通过 trust_remote_code=true、local_files_only=true 使用远程快照；插件在运行时应用兼容 patch。

---

## 5. 当前 BAT 模型的可训练参数

当前要求如下：

| 部件 | 状态 | 说明 |
|---|---:|---|
| Spatial-AST | 冻结 | FP32 推理 |
| BAT Q-Former | 训练 | 随机初始化，约 67.79M 参数 |
| Ouro Transformer 主体 | 冻结 | 原始模型权重不更新 |
| Ouro LoRA | 训练 | q_proj/v_proj，rank=8 |
| Ouro early-exit gate | 冻结 | threshold=1.0，完整 4 次循环 |
| Ouro embeddings / norm / lm_head | 冻结 | 不训练 |

LoRA 配置：

~~~text
target_modules = ["q_proj", "v_proj"]
r = 8
lora_alpha = 32
lora_dropout = 0.05
~~~

训练参数审计必须同时检查：

1. Spatial-AST 没有 requires_grad=True；
2. Ouro 原生参数没有 requires_grad=True；
3. gate 没有 requires_grad=True；
4. 只有 Q-Former 和 LoRA 参数出现在 optimizer；
5. forward 和 backward 都出现 4 次 shared-layer/gate 调用；
6. checkpoint 只包含 Q-Former 和 LoRA 的可训练状态，不应误保存完整 Ouro 权重；
7. Q-Former 没有误加载 BAT/model.pt 或其他预训练 checkpoint。

---

## 6. 音频输入链路

主要实现：

~~~text
code/Ouro_audio/bat/models/spatial_ast_audio.py
code/Ouro_audio/plugins/ouro_bat_spatial_ast_swift.py
~~~

数据源：

~~~text
AudioSet:
  /hpc_stor03/public/shared/data/raa/AudioSet

RIR:
  /hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/mp3d_reverb
~~~

当前处理契约：

1. 从 AudioSet 读取 mono 音频；多通道输入取第一通道。
2. 重采样到 32 kHz。
3. 做近似 -14 dBFS RMS 音频归一化。
4. 读取双耳 RIR，形状为 [2, L]。
5. RIR 固定到 2 秒，即 64,000 samples：
   - 超过 2 秒：裁剪；
   - 少于 2 秒：尾部补零；
   - 目标：避免不同 RIR 长度导致渲染长度和训练性能抖动。
6. 对每个耳道用 scipy.signal.fftconvolve 做卷积。
7. 裁剪/补零渲染结果到 10 秒、32 kHz，即 [2, 320000]。
8. 双音源分别渲染后逐元素平均：
   rendered = (rendered_1 + rendered_2) / 2.0
   这是双耳波形的逐元素平均，不是左右耳平均，也不是简单音频相加。
9. Spatial-AST 从双耳波形构造内部输入：
   - 双耳 log-mel 频谱；
   - IPD 的正弦和余弦；
   - 组合为四通道输入；
   - 通过官方 Spatial-AST token 路径得到 [B, 515, 768]。
10. 绕过 Spatial-AST 分类头，只取其 token 表示。
11. Q-Former 将 [B,515,768] 映射为固定 [B,64,2048]。
12. 用这 64 个音频 embedding 替换 Ouro 输入序列开头的 64 个音频占位 token。

注意：

- 当前 RIR 统计中，超过 2 秒的有效 RIR 为 356/42,262，约 0.842%；已经加入 crop/zero-pad。
- 有音频 RMS 归一化，但目前不是后 RIR、后混合的 LUFS 归一化。
- 双音源的“平均”与论文代码/当前实现契约一致；如果更换音频混合策略，必须重新做音频审计和训练对照。
- 当前正式训练没有依赖 BF16 特征缓存。预缓存路线是可选优化，不是 correctness requirement。

---

## 7. 文本、音频 token、padding 和 loss

当前固定长度设置：

~~~text
audio prefix tokens = 64
text budget          = 112
max sequence length  = 176
~~~

逻辑是：

- 文本 token 长度超过 112：截断文本到 112；
- 文本 token 长度少于 112：补 padding；
- 64 个音频 token 始终保留；
- 总输入长度固定为 64 + 112 = 176；
- padding 的 attention_mask=0；
- 音频 prefix 和 padding 对应 label 为 -100；
- padding 不参与 loss；
- 有效文本目标按 causal next-token prediction 错位计算。

“补齐”不是额外生成音频 token：音频 token 固定 64 个，补的是文本段，使整个 Ouro 输入固定为 176。超长样本保留，但文本超过 112 的部分被截断，不能因为超长而丢弃样本。

需要注意：

- Swift 的日志会把音频占位 ID 显示成 Ouro tokenizer 的 <|endoftext|>，这是占位符 ID 的文本化显示，不代表 Q-Former 输出被转换成了词表 token。
- 真正 forward 时，64 个占位位置会被 Q-Former 的连续 [64,2048] embedding 替换。
- labels 的具体覆盖范围必须以当前 template/collator 审计为准；token_acc 是有效 label 的 teacher-forced shifted argmax accuracy，不自动等于“只统计 response 的准确率”。

---

## 8. BAT 数据和课程学习

远程数据：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/SpatialSoundQA/closed-end
~~~

训练文件：

~~~text
stage1-clsdoa/train.json
stage2-single/train.json
stage3-mixup/train.json
~~~

类型映射：

~~~text
A = CLASSIFICATION
B = DOA
C = MIXUP_SINGLE_CLASSIFICATION
D = MIXUP_SINGLE_DOA
E = binary / non-binary spatial reasoning
~~~

审计得到的累计数据量：

| 课程阶段 | 包含类型 | 记录数 |
|---|---|---:|
| Stage-I | A+B | 278,784 |
| Stage-II | A+B+C+D | 514,784 |
| Stage-III | A+B+C+D+E | 872,312 |

去重后的 union 统计：

~~~text
raw QA records       = 1,665,880
unique QA records    = 872,312
unique source tuples = 872,193

A = 139,392
B = 139,392
C = 118,000
D = 118,000
E = 357,528
~~~

question_id 不是全局唯一 ID，不能单独作为去重键；必须结合 question/type/content/source tuple 判断。当前主线课程使用的是完整 QA manifest，不应根据 question_id 单独删除样本。

### 正式三阶段 curriculum

主 manifest：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/curriculum_train.jsonl
~~~

报告：

~~~text
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/curriculum_report.json
~~~

课程配置：

- Stage-I：2 epochs，A+B；
- Stage-II：2 epochs，A+B+C+D；
- Stage-III：3 epochs，A+B+C+D+E；
- 每卡 batch size=2；
- 8 卡 DDP；
- gradient accumulation=1；
- global batch size=2x8=16；
- curriculum manifest 在运行时关闭 shuffle；
- manifest 内部已经按每个 curriculum block/epoch 做确定性 shuffle；
- Stage-III 每个 epoch 补齐到 global batch 的整数倍；
- 三个 stage 在同一个 Trainer/optimizer/scheduler 中连续训练，不在阶段边界停止进程，不重新初始化模型或优化器；
- 阶段边界只触发 checkpoint 保存和 marker 写入。

按当前计数计算的步数：

~~~text
Stage-I:
  278,784 x 2 / 16 = 34,848 steps

Stage-II:
  514,784 x 2 / 16 = 64,348 steps
  cumulative boundary = 99,196

Stage-III:
  872,320 x 3 / 16 = 163,560 steps
  cumulative boundary = 262,756

total = 262,756 optimizer steps
~~~

全局 warmup 和 half-cycle cosine scheduler 的边界、warmup 数量必须以实际训练脚本打印的最终配置和 curriculum_report.json 为准。不要把 stage 重新拆成三个独立 scheduler，也不要把论文里的 epoch_partitioning_factor=10 误当成额外的十倍训练 epoch；当前 launcher 只记录论文配置，不把它解释成额外训练倍数。

正式入口：

~~~bash
BAT_CURRICULUM_MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/curriculum_train.jsonl \
BAT_CURRICULUM_REPORT=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/curriculum_report.json \
BAT_CURRICULUM_OUTPUT_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/curriculum_stage123_compile-0816 \
BAT_MAX_SEQUENCE_LENGTH=176 \
BAT_TORCH_COMPILE=true \
bash code/Ouro_audio/bat/run_train_bat_ouro_curriculum_5090.sh
~~~

脚本名可能仍带 5090，但提交队列必须查看脚本实际内容。最近版本曾切换到 pdgpu-3090，不要仅凭文件名判断。

### Stage-III 独立实验路线

这是另一条实验路线，不能与完整 curriculum checkpoint 混用：

~~~text
manifest:
/hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/stage3_ab_cde_2epoch.jsonl

report:
/hpc_stor03/sjtu_home/jinwei.zhang/outputs/ouro/bat/stage3_ab_cde_2epoch_report.json
~~~

规则：

- 只训练 Stage-III 数据；
- 每 epoch 先打乱 A+B，再打乱 C+D+E；
- 2 epochs；
- per-device batch=8；
- 8 卡 global batch=64；
- learning rate=0.002；
- warmup 约 13%；
- 每 epoch 保存 checkpoint；
- 入口：bat/run_train_bat_ouro_stage3_ab_cde_5090.sh；
- 该路线 checkpoint 和完整三阶段 curriculum checkpoint 不可互相 resume。

---

## 9. 重要代码和入口

### 模型/插件

~~~text
plugins/ouro_text_swift.py
  Ouro 文本模型在 ms-swift 中的注册和文本推理。

plugins/ouro_bat_spatial_ast_swift.py
  当前 BAT 多模态 Ouro 注册、processor、template、Q-Former 和音频链路。

compat/ouro_cache.py
  Ouro Universal Transformer cache 的 Transformers 4.54.1 运行时兼容层。

bat/models/spatial_ast_audio.py
  AudioSet/RIR/双耳渲染/Spatial-AST token/Q-Former 输入链路。

bat/ouro_compile.py
  只对 OuroForCausalLM.model 的 recurrent Transformer core 做 torch.compile。
~~~

### curriculum 正式训练

~~~text
bat/curriculum.py
  curriculum manifest 字段、边界和确定性 shuffle 契约。

bat/configs/training.py
  BAT 训练超参数和当前训练配置。

bat/scripts/compose_bat_curriculum_manifest.py
  生成完整三阶段 curriculum manifest。

bat/scripts/train_bat_ouro_curriculum.py
  一个连续 Trainer 中完成 Stage-I/II/III；阶段边界 callback 保存 checkpoint。

bat/scripts/curriculum_checkpoint.py
  阶段 marker、checkpoint 内容和 resume 审计。

bat/run_train_bat_ouro_curriculum_5090.sh
  正式三阶段 DDP 提交入口；实际 queue 以脚本内容为准。
~~~

### Stage-III 独立训练

~~~text
bat/scripts/train_bat_ouro_stage3_ab_cde.py
bat/scripts/stage3_ab_cde_checkpoint.py
bat/run_train_bat_ouro_stage3_ab_cde_5090.sh
~~~

### manifest、审计和 smoke

~~~text
bat/scripts/build_bat_unique_manifests.py
bat/scripts/split_bat_source_manifest.py
bat/scripts/audit_bat_phase1_data.py
bat/scripts/audit_bat_qformer_contract.py
bat/scripts/audit_bat_spatial_ast_audio.py
bat/scripts/audit_bat_train_contract.py
bat/scripts/audit_bat_manifest_token_lengths.py
bat/scripts/audit_bat_stage3_manifest.py

bat/scripts/smoke_bat_ouro_lora.py
bat/scripts/smoke_bat_ouro_ddp.py
bat/scripts/smoke_bat_ouro_curriculum.py
bat/scripts/smoke_bat_ouro_stage3_ab_cde.py
bat/scripts/profile_bat_ouro_pipeline.py
~~~

启动前应先检查 shell launcher 最终调用了哪个 Python 脚本、使用哪个 manifest、哪个输出目录、哪个队列和哪组 GPU 资源。

---

## 10. compile 和性能结论

当前 compile 设计：

~~~text
Spatial-AST      eager
音频渲染         eager
Q-Former         eager
Ouro recurrent core (OuroForCausalLM.model)  torch.compile
LoRA q/v         包含在 Ouro recurrent core 的 compile 图中
Ouro lm_head     eager，不单独 compile
wrapper/template eager
~~~

原因：

- Ouro 的主要循环计算在 OuroForCausalLM.model；
- LoRA q/v 是 core 内部线性层的一部分，应与 core 一起进入图；
- lm_head 在当前测量中不是主要瓶颈，不强行扩大 compile 范围；
- Spatial-AST、Q-Former 和 wrapper 包含动态/外部处理，暂时保持 eager；
- 使用固定长度 176 和 dynamic=false，目标是在稳定 shape 下复用已编译图。

已知现象：

- 第一个 compile step 很慢是正常的编译开销，后续相同 shape 应复用图；
- 早期 full-pipeline compile 曾遇到 nvcc 权限错误和 segmentation fault，因此当前只 compile Ouro core；
- Inductor 的 repro 生成可能调用 nvcc，而远程环境中的 nvcc 权限不完整；相关副作用已在 profiling/compile 路径中关闭或规避；
- 3090 上实际 SDPA 审计得到 efficient attention kernel，不应仅依据 flash_sdp_enabled=true 就宣称使用 Flash kernel；
- Linux kernel 4.18、NCCL cleanup warning、PEFT label_names warning 目前不是训练 correctness 的主要问题，但应在报告中区分 warning 和 fatal error。

profiling 的历史结论：

- 早期约 100 秒/step 的数字包含 profiler/CUDA 同步和通信测量开销，不能直接当作真实训练速度；
- 后续 DDP 分解的典型 p50 约为：
  - data wait：约 0.0001 秒；
  - backward：约 0.60 秒；
  - backward compute：约 0.50 秒；
  - DDP communication span：约 0.10 秒；
  - step wall：约 1.0 秒。
- 在线音频 RIR/Spatial-AST 仍可能造成抖动；BF16 特征预缓存可作为后续优化，但不能把一次 profiling 的 data wait 结果简单解释成所有音频处理都没有成本。

---

## 11. 已完成的关键审计和故障修复

已完成或通过的审计包括：

- Ouro-1.4B 本地 native load/generation；
- Ouro ms-swift text inference；
- Ouro cache 兼容 patch；
- Spatial-AST strict checkpoint load；
- Spatial-AST 参数冻结和 FP32 输出；
- Q-Former 随机初始化结构审计；
- Spatial-AST -> Q-Former -> Ouro 多模态 forward/backward；
- Ouro 4 次 recurrent forward/backward；
- LoRA target/rank/可训练参数审计；
- shifted causal CE 与 Trainer CE 对齐；
- DDP 8 卡 smoke；
- 160 条 preflight；
- checkpoint 完整性；
- 两步 resume；
- curriculum marker 和 stage boundary callback；
- curriculum compile smoke/resume；
- Stage-III manifest 顺序与 digest；
- 固定长度 176、截断、padding 和 -100 label 审计；
- RIR 长度 crop/zero-pad 审计；
- Spatial-AST attention/backend profiling；
- Ouro recurrent core 的 compile 图复用 smoke。

历史问题和处理：

1. Ouro cache 的 key_cache 属性冲突：用 compat/ouro_cache.py 运行时替换 cache 实现。
2. Transformers mask/cache 层列表为空：兼容 cache 初始化并正确提供 mask size 信息。
3. Trainer CE 与手工 CE 不一致：修正 causal shift、音频 prefix/padding mask 和有效 label 统计。
4. DDP checkpoint 缺少 rng_state.pth：ms-swift/Transformers 在 DDP 下生成 rng_state_0.pth 到 rng_state_7.pth；审计逻辑已兼容按 rank 的命名。
5. curriculum checkpoint 路径嵌套在 v0-<timestamp>：审计逻辑已按实际输出目录寻找 checkpoint 和 marker。
6. 重复 Q-Former 模块导致审计误报：训练模型与冻结审计副本分离识别。
7. [B,T] 与 [B,T,H] 审计误判：现在分别检查 token 序列 shape 和 embedding shape。
8. RIR 长度不一致：已统一 2 秒 crop/zero-pad 后再卷积。
9. 16 路 BF16 feature cache 掉线/覆盖问题：已做 cache index/finite/coverage 审计；当前正式路线仍使用在线处理。
10. 正式 curriculum launcher/回调问题：修复过 shell/Python 语法、fresh output 目录拒绝、阶段 marker 保存和 resume 审计问题；修改后必须重新同步远程代码再提交。

---

## 12. 当前操作纪律和检查清单

提交任何正式训练前：

1. 本地确认修改已提交并推送 GitHub；
2. 远程 git pull --ff-only；
3. 确认远程实际脚本版本和 queue；
4. 确认 manifest/report/output 都位于私有目录；
5. 确认 BAT_MAX_SEQUENCE_LENGTH=176；
6. 确认 BAT_TORCH_COMPILE=true 时实际只 compile Ouro recurrent core；
7. 确认 use_cache=false；
8. 确认 Spatial-AST FP32、冻结；
9. 确认 Q-Former 随机初始化且训练；
10. 确认 Ouro LoRA 训练、Ouro 原生参数和 gate 冻结；
11. 确认 global batch、epoch、warmup、scheduler 与当前路线一致；
12. 确认输出目录是新的，或明确使用合法 resume；
13. 不要把 Stage-III 独立路线 checkpoint resume 到三阶段 curriculum；
14. 训练中用 tail -f 看日志，用 jq 查看报告，不要通过登录节点直接启动 GPU Python；
15. 训练结束后检查 checkpoint、trainer state、optimizer、scheduler、RNG state、curriculum marker 和 digest。

status=ok 的单个审计报告只代表该审计覆盖的范围通过，不能替代最终 checkpoint、resume、训练收敛和任务评测。

---

## 13. 给新 agent 的最短上下文

如果只需要快速接手：

1. 当前不是 Huginn，也不是 OWL；当前是 BAT + Ouro-1.4B。
2. Ouro 24 层共享 Transformer 循环 4 次；训练固定 4 次，gate 冻结。
3. Spatial-AST FP32 冻结；随机 Q-Former 和 Ouro q/v LoRA 训练。
4. 音频从只读 AudioSet 读取，经 32 kHz、RMS、2 秒 RIR crop/zero-pad、双耳 fft 卷积得到 10 秒双耳波形。
5. Spatial-AST 输出 [B,515,768]，Q-Former 输出 64 个 [2048] 音频 token。
6. 固定 Ouro 序列长度 176：64 音频 token + 112 文本 token；padding label=-100。
7. 正式三阶段 manifest 是：
   /hpc_stor03/sjtu_home/jinwei.zhang/data/BAT/manifests/curriculum_train.jsonl
8. 正式三阶段是 2/2/3 epochs，8 卡 DDP，每卡 batch 2，全局 batch 16。
9. GPU 必须提交队列；公共 AudioSet 只读；输出写私有目录。
10. 先看实际 launcher 和远程代码，再根据 report 和 checkpoint 事实判断，不要凭旧 README 或文件名推断当前状态。
