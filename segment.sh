#!/bin/bash

SUPREM_DIR=$HOME/repo/SuPreM
INFERENCE_DIR=$SUPREM_DIR/direct_inference

dataset=FLARE
task=reconstruction_src1000_det500_turns8_photons1000
#task=denoising_sigma0.2
target_file=mmse
backbone=unet
while [ $# -gt 0 ]; do
    case "$1" in
    --dataset)
        dataset=$2
        shift 2
        ;;
    --task)
        task=$2
        shift 2
        ;;
    --target_file)
        target_file=$2
        shift 2
        ;;
    --backbone)
        backbone=$2
        shift 2
        ;;
    *)
        echo "Unknown argument: $1" >&2
        exit 1
        ;;
    esac
done

space_x=1.5
space_y=1.5
case $dataset in
TotalSegmentator)
    space_z=1.5
    ;;
FLARE)
    space_z=3.0
    ;;
*)
    echo "Unknown dataset: $dataset" >&2
    exit 1
    ;;
esac

case "$backbone" in
unet)
    model_path=$INFERENCE_DIR/pretrained_checkpoints/supervised_suprem_unet_2100.pth
    ;;
*)
    echo "Unknown backbone: $backbone" >&2
    exit 1
    ;;
esac

data_root_path=./results/$dataset/$task/prediction
save_path=./results/$dataset/$task/segmentation

torchrun --nproc_per_node=4 --rdzv_backend=c10d $INFERENCE_DIR/inference.py \
    --save_dir $save_path/$backbone \
    --checkpoint $model_path \
    --data_root_path $data_root_path \
    --target_file $target_file \
    --a_min 0 \
    --a_max 1 \
    --b_min 0 \
    --b_max 1 \
    --space_x $space_x \
    --space_y $space_y \
    --space_z $space_z \
    --store_result \
    --suprem \
    --dist
