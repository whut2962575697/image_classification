#!/usr/bin/env bash

 DATA_DIR='/usr/demo/common_data'
 SAVE_DIR='/usr/demo/common_data/minist_outputs/exp-baseline-wrn40_4-warmup10-epo400-32x32-use_nonlocal'
 python train.py --config_file='configs/baseline_train.yml' \
     SOLVER.BASE_LR '3e-4' SOLVER.WARMUP_EPOCH "10" SOLVER.MAX_EPOCHS "400" SOLVER.START_SAVE_EPOCH "300" SOLVER.EVAL_PERIOD "1" \
     SOLVER.PER_BATCH "128" \
     INPUT.SIZE_TRAIN "([32,32])" INPUT.SIZE_TEST "([32,32])" INPUT.RESIZE_TRAIN "([36,36])" INPUT.RESIZE_TEST "([36,36])" INPUT.USE_AUTOAUG "True" INPUT.USE_CUT_MIX "True" \
     MODEL.NAME "baseline" MODEL.BACKBONE "wrn40_4" MODEL.USE_NONLOCAL "True"\
     DATASETS.DATA_PATH "('${DATA_DIR}')" \
     OUTPUT_DIR "('${SAVE_DIR}')"