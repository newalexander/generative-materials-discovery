# Copyright 2023 Johns Hopkins University Applied Physics Laboratory

# Licensed under the Apache License, Version 2.0

ROOT_DIR=...  # point to directory where ALIGNN training data CIFS are located
ID_PROP=...  # point to where `materials-discovery/data/alignn/Ed_tern.csv` is located
OUTPUT_DIR=... # point to wherever you want to save the model

python alignn/alignn/train_folder.py \
    --root_dir $ROOT_DIR \
    --id_prop $ID_PROP \
    --structure_loc cifs \
    --config_name config.json \
    --file_format cif \
    --output_dir $OUTPUT_DIR
