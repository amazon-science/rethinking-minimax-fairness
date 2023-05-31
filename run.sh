# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# For all models except MLP, include group feature
python main.py --dir 'allmodel' --dataset_id $1 --model_id -1 --steps_minimax_fair 10000 --to_categorical --warm_start_minimax_fair

# For MLP model, include group feature
python main.py --dir 'allmodel' --dataset_id $1 --model_id 1 --steps_minimax_fair 10000 --to_categorical --warm_start_minimax_fair

# For all models except MLP, exclude group feature
python main.py --dir 'allmodel' --dataset_id $1 --model_id -1 --exclude_group_feature --steps_minimax_fair 10000 --to_categorical --warm_start_minimax_fair

# For MLP model, exclude group feature
python main.py --dir 'allmodel' --dataset_id $1 --model_id 1 --exclude_group_feature --steps_minimax_fair 10000 --to_categorical --warm_start_minimax_fair