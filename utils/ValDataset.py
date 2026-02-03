import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import trimesh
import pdb
import json
from utils.aug import *
from utils.point import sample_surface 

class ValDataset(Dataset):
    def __init__(self, root_dir, num_points=100000,seed=666):
        self.seed = seed
        self.mesh_dir = root_dir 
        self.num_points = num_points
        self.files = []
        self.categories = []
        for dirpath, dirnames, filenames in os.walk(self.mesh_dir):
            for filename in filenames:
                if filename.lower().endswith('.glb') or filename.lower().endswith('.ply') or filename.lower().endswith('.obj'):
                    self.files.append(os.path.join(dirpath, filename)) 

    def __len__(self):
        return len(self.files)

    def _sample_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        points, point_to_face,colors = sample_surface(mesh,count = self.num_points, sample_color=True,seed = self.seed)
        # points, point_to_face = trimesh.sample.sample_surface(mesh,count = self.num_points, seed = self.seed)
        if colors is None:
            colors = np.full((points.shape[0], 3), 192, dtype=np.uint8)
        colors = colors[:,:3]

        if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
            normals = mesh.face_normals[point_to_face]
        else:
            normals = np.ones((points.shape[0], 3), dtype=np.float32)

        return points, colors, normals, point_to_face

    
    def __getitem__(self, idx):
        mesh_dir = self.files[idx]
        if  mesh_dir.lower().endswith('.glb'):
            mesh = trimesh.load(mesh_dir, force='mesh', file_type='glb')
        elif mesh_dir.lower().endswith('.ply'):
            mesh = trimesh.load(mesh_dir, force='mesh', file_type='ply')
        else:
            mesh = trimesh.load(mesh_dir, force='mesh', file_type='obj')
        mesh_id = os.path.splitext(os.path.basename(mesh_dir))[0]

        vertices = mesh.vertices
        bbmin, bbmax = vertices.min(0), vertices.max(0)
        center, scale = (bbmin + bbmax)*0.5, 2.0 * 0.9 / (bbmax - bbmin).max()
        mesh.vertices = (vertices - center) * scale

        points, colors, normals, point_to_face = self._sample_points(mesh)
            
        return {
            'coords': points,  
            'normal': normals,
            'color': colors,
            'point_to_face': torch.from_numpy(point_to_face),
            'vertices': mesh.vertices,
            'faces': torch.from_numpy(mesh.faces),
            'mesh_id':mesh_id
        }
    
def prep_points_train(xyz,color, normal, vertices, eval=False):
    data_dict = {"coord": xyz,"color":color,"normal": normal, "vertices": vertices}
    data_dict = CenterShift(apply_z=True)(data_dict)
    if not eval:
        data_dict = RandomRotate(angle=[-0.5, 0.5],axis='z',center=[0,0,0],p=0.7)(data_dict)
        data_dict = RandomRotate(angle=[-0.5, 0.5],axis='x',p=0.7)(data_dict)
        data_dict = RandomRotate(angle=[-0.5, 0.5],axis='y',p=0.7)(data_dict)
        data_dict = RandomFlip(p=0.5)(data_dict)
        data_dict = ChromaticAutoContrast(p=0.2,blend_factor=None)(data_dict)
        data_dict = ChromaticTranslation(p=0.6, ratio=0.05)(data_dict)
        data_dict = ChromaticJitter(p=0.6, std=0.05)(data_dict)
        data_dict = CenterShift(apply_z=True)(data_dict)
    data_dict = NormalizeMy()(data_dict)
    if not eval:
        data_dict = RandomScale(scale=[0.95, 1.05])(data_dict)
    data_dict = NormalizeColor()(data_dict)
    data_dict = ToTensor()(data_dict)
    
    return data_dict

def collate_fn_eval(batch):
    all_coords = []
    all_normals = []
    all_colors = []
    all_point_to_face = []
    all_vertices = []
    all_faces = []
    all_ids = []
    for item in batch:
        coords = item['coords']  # [N,3]
        color = item['color']
        normal = item['normal']  # [N,3]
        vertices = item['vertices']  # [N,3]
        N = coords.shape[0]
        id = item['mesh_id']
        for m in range(1):
            data_tmp = prep_points_train(coords,color,normal,vertices,eval=True)
            all_coords.append(data_tmp["coord"].unsqueeze(0)) 
            all_colors.append(data_tmp["color"].unsqueeze(0))
            all_normals.append(data_tmp["normal"].unsqueeze(0))

            all_point_to_face.append(item['point_to_face'].unsqueeze(0))
            all_vertices.append(data_tmp['vertices'].unsqueeze(0))
            all_faces.append(item['faces'].unsqueeze(0))
            all_ids.append(id)

            
    batch_coords = torch.cat(all_coords, dim=0).squeeze(1)  # [b,1,N,3]
    batch_normals = torch.cat(all_normals, dim=0).squeeze(1)  # [b,1,N,3]
    batch_colors = torch.cat(all_colors, dim=0).squeeze(1)  # [b,1,N,3]
    batch_point_to_face = torch.cat(all_point_to_face, dim=0)  #
    batch_vertices = all_vertices  # [b,1,N,3]
    batch_faces = all_faces
    batch_ids = all_ids

    return {
        'coords': batch_coords,
        'normal': batch_normals,
        'color': batch_colors,
        'point_to_face': batch_point_to_face,
        'vertices': batch_vertices,
        'faces': batch_faces,
        'ids': batch_ids
    }


