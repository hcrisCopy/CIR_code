#!/usr/bin/env bash
# Run from CIR_code after activating the cir environment, one GPU per dataset.
set -euo pipefail
dataset=${1:?Use cirr or fashioniq}
case "$dataset" in cirr|fashioniq) ;; *) echo "Use cirr or fashioniq" >&2; exit 2 ;; esac
args=(--dataset "$dataset" --backbone vit_cir --layer 6)
python run.py baseline --dataset "$dataset" --gpus 1
python run.py probe --dataset "$dataset" --gpus 1
python -m analysis.color_views --dataset "$dataset" --backbone vit --splits train val
python -m analysis.paired_edits "${args[@]}" --split train
python -m analysis.color_operator --dataset "$dataset"
for feature in global cls patch; do
    python -m analysis.train "${args[@]}" --name "retrieval_$feature" --feature "$feature" --loss retrieval
done
for loss in retrieval hard shuffled necessity; do
    python -m analysis.train "${args[@]}" --name "color_$loss" --paired-color --feature patch --gate --loss "$loss"
done
for seed in 7 123; do
    for loss in hard shuffled necessity; do
        python -m analysis.train "${args[@]}" --name "color_${loss}_s$seed" --paired-color --feature patch --gate --loss "$loss" --seed "$seed"
    done
done
python -m analysis.paired_edits "${args[@]}" --split val --operator --heads probe_global probe_cls probe_patch color_retrieval color_hard color_shuffled color_necessity retrieval_global retrieval_cls retrieval_patch
python -m analysis.validation_slices "${args[@]}"
if [[ "$dataset" == fashioniq ]]; then
    python -m analysis.controlled_color "${args[@]}" --heads probe_global probe_patch color_hard color_shuffled color_necessity
fi
