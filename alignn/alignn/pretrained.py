#!/usr/bin/env python

"""Module to download and load pre-trained ALIGNN models."""
import requests
import os
import zipfile
import argparse
import tempfile
import sys

import torch
import pandas as pd

from torch.utils.data import DataLoader

from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.data import get_torch_dataset
from alignn.pretrained_models import local_models, figshare_models

from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.db.jsonutils import dumpjson, loadjson

from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count

tqdm.pandas()

parser = argparse.ArgumentParser(description="Atomistic Line Graph Neural Network Pretrained Models")
parser.add_argument("--model_name", default="Ed_tern",)
parser.add_argument("--file_format", default="cif", help="poscar/cif/xyz/pdb file format.")

parser.add_argument("--file_path", default=None,
                    # "/fast/newas1/data/materials_project/mp_stability/cifs/mp-1203324.cif",
                    help="Path to single prediction file.",)

parser.add_argument('--test_set_file', default='/fast/newas1/data/materials_project/mp_stability/Ed_tern_slice.csv',
                    help='path to CSV of test set IDs')
parser.add_argument('--test_set_structures', default='/fast/newas1/data/materials_project/mp_stability/cifs/',
                    help='path to folder containing test set structures')
parser.add_argument('--batch_size', default=256, type=int,
                    help='batch size for test set dataloader')
parser.add_argument('--n_jobs', default=32,
                    help='number of cores to use when ingesting dataset')
parser.add_argument('--test_set_save_file',
                    default='/fast/newas1/data/materials_project/Ed_tern/Ed_tern_slice_pred.csv')

parser.add_argument("--cutoff", default=8,
                    help="Distance cut-off for graph constuction, usually 8 for solids and 5 for molecules.",)
parser.add_argument("--max_neighbors", default=12,
                    help="Maximum number of nearest neighbors in the periodic atomistic graph construction.",)


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def get_figshare_models():
    """Return the figshare links for models."""
    return figshare_models


def get_local_models():
    """Return the locations of local models"""
    return local_models


def get_figshare_model(model_name="jv_formation_energy_peratom_alignn"):
    """Get ALIGNN torch models from figshare."""
    # https://figshare.com/projects/ALIGNN_models/126478

    tmp = figshare_models[model_name]
    url = tmp[0]
    output_features = tmp[1]
    if len(tmp) > 2:
        config_params = tmp[2]
    else:
        config_params = {}
    zfile = model_name + ".zip"
    path = str(os.path.join(os.path.dirname(__file__), zfile))
    if not os.path.isfile(path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True
        )
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    zp = zipfile.ZipFile(path)
    names = zp.namelist()
    chks = []
    for i in names:
        if "checkpoint_" in i and "pt" in i:
            tmp = i
            chks.append(i)
    print("Using chk file", tmp, "from ", chks)
    print("Path", os.path.abspath(path))
    # print("Loading the zipfile...", zipfile.ZipFile(path).namelist())
    data = zipfile.ZipFile(path).read(tmp)
    model = ALIGNN(
        ALIGNNConfig(
            name="alignn", output_features=output_features, **config_params
        )
    )
    new_file, filename = tempfile.mkstemp()
    with open(filename, "wb") as f:
        f.write(data)
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    model.to(device)
    model.eval()
    if os.path.exists(filename):
        os.remove(filename)
    return model


def get_local_model(model_name):
    """Get ALIGNN torch models from a local directory."""
    tmp = local_models[model_name]
    file = tmp[0]
    config_params = loadjson(tmp[2])['model']
    print(f'using model: {file} with params:')
    print(config_params)

    model = ALIGNN(ALIGNNConfig(**config_params))
    model.load_state_dict(torch.load(file, map_location=device)["model"])
    model.to(device)
    model.eval()
    return model


def get_atoms(file_path):
    if args.file_format == "poscar":
        atoms = Atoms.from_poscar(file_path)
    elif args.file_format == "cif":
        atoms = Atoms.from_cif(file_path)
    elif args.file_format == "xyz":
        atoms = Atoms.from_xyz(file_path, box_size=500)
    elif args.file_format == "pdb":
        atoms = Atoms.from_pdb(file_path, max_lat=500)
    else:
        raise NotImplementedError("File format not implemented", args.file_format)
    return os.path.basename(file_path), atoms


def get_prediction(
    model_name="jv_formation_energy_peratom_alignn",
    cutoff=8.,
    max_neighbors=12,
):
    """Get model prediction on a single structure."""
    atoms = get_atoms(args.file_path)[0]

    if model_name in figshare_models.keys():
        model = get_figshare_model(model_name)
    else:
        model = get_local_model(model_name)
    # print("Loading completed.")
    g, lg = Graph.atom_dgl_multigraph(
        atoms,
        cutoff=float(cutoff),
        max_neighbors=max_neighbors,
    )
    return (
        model([g.to(device), lg.to(device)])
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )


def get_predictions(
    model_name="jv_formation_energy_peratom_alignn",
    cutoff=8.,
    max_neighbors=12,
    neighbor_strategy="k-nearest",
    use_canonize=True,
    atom_features="cgcnn",
    line_graph=True,
    pin_memory=False,
    batch_size=1,
    print_freq=100,
):
    """Use pretrained model on a number of structures."""
    if model_name in figshare_models.keys():
        model = get_figshare_model(model_name)
    else:
        model = get_local_model(model_name)
    print('model loaded')

    # ingest test set
    test_set = pd.read_csv(args.test_set_file, header=None).rename({'id': 'file_path'}, axis=1)
    atoms = []
    for file_path in tqdm(test_set['file_path']):
        try:
            atoms.append(get_atoms(os.path.join(args.test_set_structures, file_path)))
        except TypeError or KeyError or ValueError:
            continue
    # atoms = Parallel(n_jobs=args.n_jobs, verbose=11)(
    #     get_atoms(os.path.join(args.test_set_structures, file_path)) for file_path in test_set['file_path'])

    mem = []
    for _atoms_id, _atoms in tqdm(atoms):
        info = {'atoms': _atoms.to_dict(),
                'prop': -9999.,  # placeholder only
                'jid': _atoms_id}
        mem.append(info)

    test_data = get_torch_dataset(
        dataset=mem,
        target="prop",
        neighbor_strategy=neighbor_strategy,
        atom_features=atom_features,
        use_canonize=use_canonize,
        line_graph=line_graph,
        cutoff=cutoff,
        max_neighbors=max_neighbors
    )
    print('torch dataset generated')

    collate_fn = test_data.collate_line_graph
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=1,
        pin_memory=pin_memory,
    )
    print('test loader made')

    # evaluate model on test set
    results = []
    with torch.no_grad():
        ids = test_loader.dataset.ids
        for dat, _id in zip(test_loader, ids):
            g, lg, target = dat
            pred = model([g.to(device), lg.to(device)])
            pred = pred.cpu().numpy().tolist()
            info = {'id': _id,
                    'pred': pred}
            results.append(info)
            print_freq = int(print_freq)
            if len(results) % print_freq == 0:
                print(len(results))

    if args.batch_size == 1:
        df = pd.concat([pd.DataFrame(foo, index=[0]) for foo in results]).reset_index(drop=True)
    else:
        df = pd.concat([pd.DataFrame(foo, index=[0]) for foo in results]).reset_index(drop=True)
    df = df.assign(m_id=pd.DataFrame(mem)['jid'])
    df.to_csv(args.test_set_save_file, index=False)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])

    if args.file_path is not None:
        out_data = get_prediction(
            model_name=args.model_name,
            cutoff=float(args.cutoff),
            max_neighbors=int(args.max_neighbors)
        )
    if args.test_set_file is not None and args.test_set_structures is not None:
        get_predictions(
            model_name=args.model_name,
            cutoff=float(args.cutoff),
            max_neighbors=int(args.max_neighbors),
            batch_size=args.batch_size,
        )
