"""Bake a downloaded CC0 glTF rock into a small, dependency-free radial asset.

Development only: requires trimesh, scipy and rtree. The renderer loads the NPZ
with NumPy; it never downloads models or imports these development dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


def bake(source, target):
    mesh=trimesh.load(source,force='mesh')
    # glTF Y-up -> image Y-down, with the long natural axis across the palm.
    turn=np.array([[0,0,1],[0,-1,0],[1,0,0.]])
    vertices=mesh.vertices@turn.T
    center=(vertices.min(0)+vertices.max(0))/2
    scale=np.linalg.norm(vertices-center,axis=1).max()
    vertices=(vertices-center)/scale
    normals=mesh.vertex_normals@turn.T
    geom=trimesh.Trimesh(vertices=vertices,faces=mesh.faces,process=False)
    texture=np.asarray(mesh.visual.material.baseColorTexture.convert('RGB'))
    tex_uv=mesh.visual.uv
    rows,cols=256,512
    latitude=np.linspace(-np.pi/2+.0001,np.pi/2-.0001,rows)
    longitude=np.linspace(-np.pi,np.pi,cols,endpoint=False)
    lat,lon=np.meshgrid(latitude,longitude,indexing='ij')
    directions=np.stack([np.cos(lat)*np.cos(lon),np.sin(lat),np.cos(lat)*np.sin(lon)],-1).reshape(-1,3)
    radius=np.zeros(len(directions),np.float32)
    albedo=np.zeros((len(directions),3),np.uint8)
    normal=np.zeros((len(directions),3),np.float32)
    for start in range(0,len(directions),512):
        d=directions[start:start+512]
        locations,ray,face=geom.ray.intersects_location(np.zeros_like(d),d,multiple_hits=False)
        bary=trimesh.triangles.points_to_barycentric(vertices[mesh.faces[face]],locations)
        uv=(tex_uv[mesh.faces[face]]*bary[...,None]).sum(1)
        n=(normals[mesh.faces[face]]*bary[...,None]).sum(1)
        n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8)
        radius[start+ray]=np.linalg.norm(locations,axis=1)
        albedo[start+ray]=cv2.remap(texture,(uv[:,0]*(texture.shape[1]-1)).astype(np.float32)[None],
            ((1-uv[:,1])*(texture.shape[0]-1)).astype(np.float32)[None],cv2.INTER_LINEAR)[0]
        normal[start+ray]=n
    missing=radius==0
    if missing.mean()>.025:
        raise ValueError(f'Scan has too many radial holes: {missing.mean():.1%}')
    if missing.any():
        from scipy.spatial import cKDTree
        nearest=cKDTree(directions[~missing]).query(directions[missing])[1]
        for values in (radius,albedo,normal):values[missing]=values[~missing][nearest]
    sphere=trimesh.creation.icosphere(subdivisions=2)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,radius=radius.reshape(rows,cols),albedo=albedo.reshape(rows,cols,3),
        normal=normal.reshape(rows,cols,3),directions=sphere.vertices.astype(np.float32),faces=sphere.faces)
    metadata={'asset':'Rock 07','author':'Jenelle van Heerden','source':'https://polyhaven.com/a/rock_07',
        'license':'CC0-1.0','license_url':'https://polyhaven.com/license',
        'adaptation':'Radial resampling of the photographed model and its UV diffuse map; normalized and reoriented.',
        'original_gltf_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'baked_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'radial_holes_filled':int(missing.sum()),'radial_samples':len(radius),'surface_triangles':len(sphere.faces)}
    target.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('target',type=Path)
    args=p.parse_args();bake(args.source,args.target)
