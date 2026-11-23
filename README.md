# CIR 条件充分表征实验

给模型一张参考图和一句修改要求，让它从图库中找目标图。这里检验两个问题：模型会不会把本该改掉的属性带进结果？测出的属性依赖能不能改善真实检索？

## 环境

在 Linux 的 `CIR_code/` 下运行。单卡实验使用 Python 3.12、PyTorch 2.8 和 CUDA 12.8。已有 `cir` 环境可直接激活。[PyTorch 安装表](https://pytorch.org/get-started/previous-versions/)

```bash
conda create -n cir python=3.12 pip -y
conda activate cir
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --index-url https://pypi.org/simple --timeout 60 -r requirements.txt
```

## 准备数据和权重

CIRR 和 FashionIQ 放在 `../CIR_data/datasets/`。主实验只需 OpenAI 的 `../CIR_data/checkpoints/ViT-B-16.pt`。已有图像、标注和权重时，下面前三条可以跳过。最后一条准备 CIRCUS 的四个小标注文件，用来单独评估人工核验过的查询。[图包](https://huggingface.co/datasets/BUAADreamer/cir_dataset) · [CIRCUS](https://github.com/EdinburghNLP/CIRCUS)

```bash
hf download BUAADreamer/cir_dataset --repo-type dataset --revision c4defab4386f8b90306b07d4054c653f2194d94e --include 'cirr_dataset/*' 'fashionIQ_dataset/*' --local-dir ../CIR_data/datasets
python prepare.py --dataset both
python prepare_models.py --models vit
python experiments.py --dataset both --prepare-slices
```

受网络限制时，可在本地准备后复制对应文件。另行复核 CLIP4Cir 作者的 RN50x4 时才需要 `python prepare_models.py --models release`，共 8.55 GB。`--models both` 会准备两套权重。[CLIP4Cir 发布页](https://github.com/ABaldrati/CLIP4Cir/releases/tag/weights)

## 单卡运行

```bash
python experiments.py --dataset both --limit 0 --seeds 42 7 123 --products 128
python -m analysis.evidence
```

第一条分别训练两个数据集的冻结 ViT + Combiner，比较最终层与第六层的读出，再做同参考图的指令配对、参考图去色、训练监督对照和 CIRCUS 切片评测。FashionIQ 另测三种指令：保留原色、指定颜色、换成另一种颜色。图库始终使用原划分；受控改色单独做二选一。训练曲线在 `../CIR_data/checkpoints/diagnosis/`，逐查询排名在 `../CIR_data/results/paired_edits/`，第二条生成 `../CIR_data/results/nov23/evidence/` 下的汇总表和图。

只看基线或先调试，可运行 `python run.py baseline --dataset cirr --gpus 1 --limit 128 --layer 6`。全量用 `--limit 0`。缓存和训练支持断点续跑。

## 可选的人工核查

`python run.py review --dataset both --splits train val --per-group 100 --layer 6 --limit 0` 生成 `../CIR_data/reviews/` 下的核查表。填好区域和供体后，用 `python run.py intervene --dataset both --splits train val --backbone vit_cir --gpus 1 --layer 6 --limit 0` 测局部替换，再用 `python run.py compare --dataset both --gpus 1 --feature patch --layer 6 --limit 0` 比较监督方式。结果在 `../CIR_data/results/interventions/` 和 `../CIR_data/results/diagnosis/`。[填写说明](analysis/人工核查.md)
