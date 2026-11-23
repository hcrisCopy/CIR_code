# CIR 条件充分表征实验

给一张参考图和一句修改要求，模型要从图库里找出目标图。这套代码先检查检索排名，再测量替换某个视觉因素后，目标图相对困难负样本的排名优势怎样变化。

## 环境

以下命令都在 Linux 服务器的 `CIR_code/` 下执行。使用 Python 3.12 和 CUDA 12.8 版 PyTorch，服务器需有兼容驱动。无需在线日志账号。[PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/)

```bash
conda create -n cir python=3.12 pip -y
conda activate cir
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --index-url https://pypi.org/simple --timeout 60 -r requirements.txt
python -m pip check
```

## 数据和模型

一次下载 CIRR、FashionIQ 的图包与标注。解包程序按划分核对 train、val 图片，写到 `../CIR_data/datasets/`。[数据包](https://huggingface.co/datasets/BUAADreamer/cir_dataset)

```bash
hf download BUAADreamer/cir_dataset --repo-type dataset --revision c4defab4386f8b90306b07d4054c653f2194d94e --include 'cirr_dataset/*' 'fashionIQ_dataset/*' --local-dir ../CIR_data/datasets
python prepare.py --dataset both
python prepare_models.py --models both
```

模型放在 `../CIR_data/checkpoints/`。脚本用 curl 下载 OpenAI ViT-B/16 和 CLIP4Cir 作者发布的五个分卷，从中提取两个数据集各自的 RN50x4 编码器与 Combiner。服务器只需上传 `CIR_code/`，后续训练和评测读取本地文件。[CLIP4Cir 权重](https://github.com/ABaldrati/CLIP4Cir/releases/tag/weights)

也可以在 Windows 上准备权重。五个分卷共 8.55 GB，脚本按 SHA-256 校验并直接解包。下载中断后重跑原命令即可续传。若已下载 ViT，只补公开基线可用 `python prepare_models.py --models release`。

## 运行顺序

**1. 验收公开模型。** 在完整验证图库上跑 RN50x4。每条查询的目标排名、Top-K、困难候选和相似度写入 `../CIR_data/results/diagnosis/`。CIRR 同时报全图库和子集 Recall，FashionIQ 分三类评测后取宏平均。

```bash
python run.py release --dataset both --splits val --gpus 8 --encode-batch 64 --top-k 10 --limit 0
```

**2. 建立 ViT-B/16 基线。** 冻结 CLIP，分别用两个数据集的 train 拟合 Combiner，在各自 val 选权重。缓存图库、文本、组合查询和第 6 个视觉块的 CLS 与 patch 均值。随后用相同正负对和相同容量的探针比较三种视觉读出。缓存位于 `../CIR_data/embeddings/`，权重、训练曲线位于 `../CIR_data/checkpoints/diagnosis/`。

```bash
python run.py baseline --dataset both --gpus 8 --layer 6 --encode-batch 64 --global-batch 256 --epochs 20 --lr 0.0003 --temperature 0.07 --limit 0
python run.py probe --dataset both --gpus 8 --layer 6 --width 64 --global-batch 256 --epochs 20 --lr 0.0003 --temperature 0.07 --limit 0
```

**3. 核查错误，做干预。** 按原模型排对、排错分别抽样。第一条命令生成 `../CIR_data/reviews/` 下的人工核查表。确认负样本和可替换区域后运行第二条，逐样本效应写入 `../CIR_data/results/interventions/`。[核查表填写说明](analysis/人工核查.md)

```bash
python run.py review --dataset both --splits train val --per-group 100 --layer 6 --limit 0
python run.py intervene --dataset both --splits train val --backbone vit_cir --gpus 8 --temperature 0.07 --layer 6 --limit 0
```

**4. 做训练对照。** 在已核查的 train 样本上比较线性层、文本门控、困难负样本、打乱监督和真实必要性监督。用固定末轮在完整 val 图库评测，并在核查过的 val 样本统计修复和新增错误。结果见 `../CIR_data/results/diagnosis/` 下的 `summary.json` 和 `paired_review.json`。

```bash
python run.py compare --dataset both --gpus 8 --layer 6 --feature patch --global-batch 256 --epochs 20 --width 64 --lr 0.0003 --temperature 0.07 --weight 1 --limit 0
```

单卡把 `--gpus 8` 改成 `1`，训练的全局 batch 保持一致。调试时可在各阶段统一设置 `--dataset cirr --limit 128`；只限制查询数，保留完整图库。中断后重跑同一命令，缓存和排名按批续跑，训练从最近完成的 epoch 继续。`--clean` 重建所选阶段的输出，人工核查表单独保存。
