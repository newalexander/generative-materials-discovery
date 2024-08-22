# Copyright 2023 Johns Hopkins University Applied Physics Laboratory

# Licensed under the Apache License, Version 2.0

MODEL_LOCATION=... # point to `materials_discovery/Ed_tern`
PGCGM_DATA=...  # point to `materials-discovery/data/pgcgm/Ed_tern.csv`
PGCGM_STRUCTURES=...  # point to `materials-discovery/data/pgcgm/merged_cifs`
PGCGM_PREDICTIONS=...  # where predictions for PGCGM data should be saved\


python alignn/pretrained.py \
  --model_name Ed_tern \
  --test_set_file $PGCGM_DATA \
  --test_set_structures $PGCGM_STRUCTURES \
  --test_set_save_file $PGCGM_PREDICTIONS \
  --batch_size 1
