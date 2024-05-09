# Copyright 2023 Johns Hopkins University Applied Physics Laboratory

# Licensed under the Apache License, Version 2.0

PGCGM_DATA=...  # point to `pgcgm/2022_12_02_12_00_31-structures.csv`
PGCGM_STRUCTURES=...  # point to `pgcgm/cifs/2022-12-02-12-00-31/`
PGCGM_PREDICTIONS=...  # where predictions for PGCGM data should be saved\


python alignn/pretrained.py \
  --model_name Ed_tern \
  --test_set_file $PGCGM_DATA \
  --test_set_structures $PGCGM_STRUCTURES \
  --test_set_save_file $PGCGM_PREDICTIONS \
  --batch_size 1