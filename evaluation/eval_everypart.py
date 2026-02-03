import sys

sys.path.append(".")

import os
import argparse
import hydra
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from PartSAM.utils.torch_utils import replace_with_fused_layernorm
from safetensors.torch import load_model
import matplotlib.colors as mcolors
import pointops
import torch
import numpy as np
import trimesh
import gc
from torch.utils.data import ConcatDataset, DataLoader
from utils.ValDataset import ValDataset,collate_fn_eval
from collections import defaultdict,Counter, deque
from tqdm import tqdm
import multiprocessing as mp
from utils.infer_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="partsam", help="path to config file"
    )
    parser.add_argument("--config_dir", type=str, default="../configs")
    args, unknown_args = parser.parse_known_args()

    # ---------------------------------------------------------------------------- #
    # Load configuration
    # ---------------------------------------------------------------------------- #
    with hydra.initialize(args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config, overrides=unknown_args)
        OmegaConf.resolve(cfg)

    seed = cfg.get("seed", 83)

    # ---------------------------------------------------------------------------- #
    # Setup model
    # ---------------------------------------------------------------------------- #
    set_seed(seed)
    model = hydra.utils.instantiate(cfg.model)
    model.apply(replace_with_fused_layernorm)

    # ---------------------------------------------------------------------------- #
    # Load pre-trained model
    # ---------------------------------------------------------------------------- #
    load_model(model, cfg.eval_params.ckpt_path)

    # ---------------------------------------------------------------------------- #
    # Inference
    # ---------------------------------------------------------------------------- #
    model.eval()
    model.cuda()

    val_dataset = ValDataset(root_dir=cfg.dataset.root_dir)
    val_dataloader = DataLoader(
        val_dataset,
        **cfg.val_dataloader,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate_fn_eval,
    )
    for idx, data in enumerate(tqdm(val_dataloader, desc="Evaluating")):
        
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.cuda(non_blocking=True)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                data[k] = [t.cuda(non_blocking=True) for t in v]
        try:
            data_input = {k: (v.clone() if isinstance(v, torch.Tensor) else v.copy() if isinstance(v, list) else v) for k, v in data.items()}

        except Exception as e:
            continue

        data_input.pop("ids", None)

        coord = data_input["coords"][0]
        coord_offset = torch.tensor([coord.shape[0]]).cuda()
        fps_point_number = cfg.eval_params.fps_point_number
        new_coord_offset = torch.tensor([fps_point_number]).cuda()
        fps_idx = pointops.farthest_point_sampling(coord,coord_offset,new_coord_offset)
        prompt_labels = torch.tensor([1], dtype=torch.long).unsqueeze(0).cuda()
        masks = []
        scores = []
        batch_size = cfg.eval_params.batch_size

        with torch.no_grad():
            for batch_start in range(0, fps_idx.size(0), batch_size):
                
                batch_end = min(batch_start + batch_size, fps_idx.size(0))
                batch_indices = range(batch_start, batch_end)
                selected_indices = fps_idx[batch_start:batch_end]
                prompt_points_temp = coord[selected_indices].unsqueeze(1) 
                data_in = {
                    "coords": data_input["coords"][0].repeat(len(batch_indices), 1, 1),
                    "color": data_input["color"][0].repeat(len(batch_indices), 1, 1),
                    "normal": data_input["normal"][0].repeat(len(batch_indices), 1, 1),
                    "point_to_face": data_input["point_to_face"][0].repeat(len(batch_indices), 1),
                    "vertices": data_input["vertices"][0].repeat(len(batch_indices), 1, 1),
                    "faces": data_input["faces"][0].repeat(len(batch_indices), 1, 1),
                    "prompt_coords": prompt_points_temp,
                    "selected_indices": selected_indices,  
                    "prompt_labels": prompt_labels.expand(len(batch_indices), -1)
                }

                batch_masks, batch_scores = model.predict_masks(**data_in)
                masks.append(batch_masks.cpu())
                scores.append(batch_scores.cpu())
                del batch_masks, batch_scores
                torch.cuda.empty_cache()
                gc.collect()


        masks = torch.cat(masks, dim=0)
        scores = torch.cat(scores, dim=0)
        masks = masks.reshape(-1, masks.size(2))>0
        scores = scores.reshape(scores.size(0)*scores.size(1), -1)
        iou_thes = cfg.eval_params.iou_threshold
        top_indices = (scores>iou_thes).squeeze()
        nms_thes = cfg.eval_params.nms_threshold
        masks = masks[top_indices]
        scores = scores[top_indices]

        model.labels = None
        model.pc_embeddings = None

        print(  f"Number of masks after Thersholding: {masks.shape[0]}")

        nms_indices = nms(masks, scores, threshold=nms_thes)
        print(  f"Number of masks after NMS: {len(nms_indices)}")
        filtered_masks = masks[nms_indices]
        
        sorted_masks = sort_masks_by_area(filtered_masks)
        labels = torch.full((sorted_masks.size(1),), -1)
        for i in range(len(filtered_masks)):
            labels[sorted_masks[i]] = i

        mesh = trimesh.Trimesh(vertices=(data_input["vertices"][0].squeeze(0)).cpu().numpy(), faces=(data_input["faces"][0].squeeze(0)).cpu().numpy())
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
        face_index = data_input["point_to_face"][0].cpu().numpy()
        num_faces = len(mesh.faces)
        num_labels = max(labels) + 1
        votes = np.zeros((num_faces, num_labels), dtype=np.int32)
        np.add.at(votes, (face_index, labels), 1)
        max_votes_labels = np.argmax(votes, axis=1)
        max_votes_labels[np.all(votes == 0, axis=1)] = -1
        valid_mask = max_votes_labels != -1
        face_centroids = mesh.triangles_center
        coord = torch.tensor(face_centroids).cuda().contiguous().float()
        valid_coord = coord[valid_mask]
        valid_offset = torch.tensor(valid_coord.shape[0]).cuda()
        invalid_coord = coord[~valid_mask]
        invalid_offset = torch.tensor(invalid_coord.shape[0]).cuda()
        indices, distances = pointops.knn_query(1, valid_coord, valid_offset, invalid_coord, invalid_offset)
        indices = indices[:, 0].cpu().numpy()
        mesh_group = max_votes_labels.copy()
        mesh_group[~valid_mask] = mesh_group[valid_mask][indices]
        mesh = post_processing(mesh_group, mesh, cfg.eval_params)

        id = data['ids'][0]
        mesh_save_path = os.path.join(f"results", f"{id}.ply")
        print(f"Saving mesh to {mesh_save_path}")
        mesh.export(mesh_save_path)



if __name__ == "__main__":
    main()
