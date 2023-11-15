# FARLAB - UrbanECG
# Developer: @mattwfranchi
# Last Edited: 11/14/2023

# This script contains static commands for the drivable area inference pipeline.

SINGLE_INFERENCE = 'python ./test.py ${CONFIG_FILE} --format-only --format-dir ${OUTPUT_DIR} [--options]'

MULTI_INFERENCE = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=12000 ./test.py $CFG_FILE \
    --format-only --format-dir ${OUTPUT_DIR} [--options] \
    --launcher pytorch'