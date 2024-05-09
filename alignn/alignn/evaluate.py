#!/usr/bin/env python

"""Module to train for a folder with formatted dataset."""
import csv
import os
import sys
import time

from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson

from alignn.data import get_train_val_loaders
from alignn.train import evaluate_dgl
from alignn.config import TrainingConfig

from argparse import ArgumentParser
from joblib import Parallel, delayed
from multiprocessing import cpu_count


parser = ArgumentParser(
    description="Atomistic Line Graph Neural Network"
)
parser.add_argument(
    "--root_dir",
    default="./",
    help="Folder with id_props.csv, structure files",
)
parser.add_argument(
    '--id_prop',
    default='id_prop.csv',
    type=str,
    help='csv with material IDs and properties'
)
parser.add_argument(
    '--structure_loc',
    default='',
    type=str,
    help='location within `root_dir` where structure files are located'
)
parser.add_argument(
    "--config_name",
    default="alignn/examples/sample_data/config_example.json",
    type=str,
    help="Name of the config file",
)
parser.add_argument(
    "--classification_threshold",
    default=None,
    type=float,
    help="Floating point threshold for converting into 0/1 class"
    + ", use only for classification tasks",
)
parser.add_argument(
    '--checkpoint',
    type=str
)
parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format.",
    type=str,
)
parser.add_argument(
    "--batch_size", default=None, help="Batch size, generally 64",
    type=int,
)

parser.add_argument(
    "--epochs", default=None, help="Number of epochs, generally 300",
    type=int,
)

parser.add_argument(
    "--output_dir", default="./", help="Folder to save outputs",
    type=str,
)


def ingest_datapoint(datum, root_dir, structure_loc, file_format):
    info = {}
    file_name = datum[0]
    file_path = os.path.join(root_dir, structure_loc, file_name)
    if file_format == "poscar":
        atoms = Atoms.from_poscar(file_path)
    elif file_format == "cif":
        atoms = Atoms.from_cif(file_path)
    elif file_format == "xyz":
        # Note using 500 angstrom as box size
        atoms = Atoms.from_xyz(file_path, box_size=500)
    elif file_format == "pdb":
        # Note using 500 angstrom as box size
        # Recommended install pytraj
        # conda install -c ambermd pytraj
        atoms = Atoms.from_pdb(file_path, max_lat=500)
    else:
        raise NotImplementedError(
            "File format not implemented", file_format
        )

    info["atoms"] = atoms.to_dict()
    info["jid"] = file_name

    tmp = [float(j) for j in datum[1:]]  # float(i[1])
    if len(tmp) == 1:
        tmp = tmp[0]
    info["target"] = tmp  # float(i[1])

    return info


def evaluate_on_folder(
    root_dir="examples/sample_data",
    id_prop="id_prop.csv",
    config_name="config.json",
    structure_loc='',
    keep_data_order=False,
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    file_format="poscar",
    output_dir=None,
):
    """evaluate on a folder."""
    id_prop_dat = os.path.join(root_dir, id_prop)
    config = loadjson(config_name)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    config.keep_data_order = keep_data_order
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
    with open(id_prop_dat, "r", encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        data = [row for row in reader][:64]

    dataset = Parallel(n_jobs=cpu_count()-1, verbose=11
                       )(delayed(ingest_datapoint)(datum, root_dir, structure_loc, file_format) for datum in data)
    print('dataset:', len(dataset), dataset[0])

    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        dataset_array=dataset,
        target=config.target,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=0.,
        val_ratio=0.,
        test_ratio=1.,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=True,
        output_dir=config.output_dir,
    )
    t1 = time.time()
    evaluate_dgl(
        config,
        checkpoint=args.checkpoint,
        train_val_test_loaders=[
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ],
    )
    t2 = time.time()
    print("Time taken (s):", t2 - t1)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    args.root_dir = '/Users/newas1/data/materials_project/mp_stability/'
    args.structure_loc = 'cifs/'
    args.config_name = '/Users/newas1/PycharmProjects/alignn-proj/alignn-clone/alignn/alignn/examples/sample_data/config_example.json'
    args.id_prop = 'Ed_tern_slice.csv'
    args.file_format = 'cif'
    args.checkpoint = '/Users/newas1/data/materials_project/alignn/Ed_tern/checkpoint_300.pt'
    print(args)

    evaluate_on_folder(
        root_dir=args.root_dir,
        id_prop=args.id_prop,
        config_name=args.config_name,
        structure_loc=args.structure_loc,
        keep_data_order=True,
        classification_threshold=args.classification_threshold,
        output_dir=args.output_dir,
        batch_size=(args.batch_size),
        epochs=(args.epochs),
        file_format=(args.file_format),
    )
