"""CUDA timing on the calling host of an input-only visibility prototype.

Compares one real map by default; --verify-all checks all eight maps and D4.
FP32 error tolerance 0.001 m, solely an implementation check, not mesh accuracy.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from analyze_tx_xyz import read_json
from experiment_provenance import fingerprint
from probe_terrain_visibility import clearance


def clearance_cuda(height, tx, resolution, tx_agl, rx_agl, chunk=4096):
    h,w = height.shape
    rows,cols = np.indices(height.shape,dtype=np.float64)
    ends = np.stack((rows.ravel(),cols.ravel()),axis=1)
    distance = np.linalg.norm(ends-np.array(tx),axis=1)*resolution
    intervals = 2*np.maximum(1,np.ceil(distance/resolution).astype(int))
    with torch.no_grad():
        surface = torch.from_numpy(height).to(device="cuda",dtype=torch.float32)[None,None]
        endpoints = torch.from_numpy(ends).to(device="cuda",dtype=torch.float32)
        steps = torch.from_numpy(intervals).to(device="cuda")
        z_tx = surface[0,0,tx[0],tx[1]]+tx_agl
        output = []
        for start in range(0,h*w,chunk):
            end = endpoints[start:start+chunk]
            n = steps[start:start+chunk]
            k = torch.arange(int(n.max())+1,device="cuda")[None,:]
            alpha = (k/n[:,None]).clamp_max(1)
            y = tx[0]+alpha*(end[:,0,None]-tx[0])
            x = tx[1]+alpha*(end[:,1,None]-tx[1])
            grid = torch.stack((2*x/(w-1)-1,2*y/(h-1)-1),dim=-1)[None]
            terrain = F.grid_sample(surface,grid,mode="bilinear",padding_mode="border",align_corners=True)[0,0]
            z_rx = surface[0,0,end[:,0].long(),end[:,1].long()]+rx_agl
            ray = z_tx+alpha*(z_rx[:,None]-z_tx)
            delta = (ray-terrain).masked_fill(k>n[:,None],float("inf"))
            output.append(delta.min(1).values)
        return torch.cat(output).reshape(h,w)


def analyze(source,metadata,out,repeats,verify_all=False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback timing reported as GPU")
    out.mkdir(parents=True,exist_ok=False)
    meta = read_json(metadata)
    lo,hi = meta["heightmap_range_m"]
    resolution = meta["resolution_m_per_px"]
    tx_agl,rx_agl = (meta["simulation"][key] for key in ("tx_height_m_agl","rx_height_m_agl"))
    paths = sorted((source/"tx_xyz_v1_seed0/seed0_zeros/boundary").glob("*.npz"))
    assert len(paths) == 8
    rows = []
    for index,path in enumerate(paths):
        with np.load(path) as cache:
            inputs = cache["input"].copy()
        height = inputs[0].astype(np.float64)*(hi-lo)+lo
        tx_map = inputs[1]
        assert ((tx_map==0)|(tx_map==1)).all() and tx_map.sum()==1
        tx = tuple(np.argwhere(tx_map>0)[0])
        result = clearance_cuda(height,tx,resolution,tx_agl,rx_agl)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = clearance_cuda(height,tx,resolution,tx_agl,rx_agl)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
        row = {"sample":path.stem,"seconds_raw":times,"seconds_mean":float(np.mean(times))}
        if index==0 or verify_all:
            reference = clearance(height,tx,resolution,tx_agl,rx_agl)
            prediction = result.cpu().numpy()
            absolute = np.abs(reference-prediction)
            assert absolute.max()<0.001
            row.update({"fp64_reference_max_abs_error_m":float(absolute.max()),
                        "visibility_disagreements":int(((reference>1e-4)!=(prediction>1e-4)).sum())})
            if verify_all:
                marker=np.zeros_like(height)
                marker[tx]=1
                transforms=[]
                for turns in range(4):
                    for flip in (False,True):
                        transformed=np.rot90(height,turns)
                        transformed_marker=np.rot90(marker,turns)
                        expected=np.rot90(prediction,turns)
                        if flip:
                            transformed=np.fliplr(transformed)
                            transformed_marker=np.fliplr(transformed_marker)
                            expected=np.fliplr(expected)
                        transformed_tx=tuple(np.argwhere(transformed_marker>0)[0])
                        actual=clearance_cuda(transformed.copy(),transformed_tx,resolution,tx_agl,rx_agl).cpu().numpy()
                        error=float(np.max(np.abs(actual-expected)))
                        assert error<0.001
                        transforms.append({"rot90":turns,"flip_lr":flip,"max_abs_error_m":error})
                row["d4_checks"]=transforms
        rows.append(row)
        print(f"CUDA {path.stem}: {row['seconds_mean']:.4f} s",flush=True)
    audit = {"maps":rows,"mean_seconds_per_map":float(np.mean([row["seconds_mean"] for row in rows])),
             "verify_all":verify_all,
             "gpu":torch.cuda.get_device_name(0),"torch":str(torch.__version__),
             "scope":"CPU coordinate setup, host-to-GPU input transfer, FP32 ray samples/grid_sample/min and synchronize, batch1, warmup excluded. File IO, CPU output transfer, feature asinh scaling, training and model forward excluded. No AMP.",
             "source_fingerprints":[fingerprint(metadata)]+[fingerprint(path) for path in paths],
             "scripts":[fingerprint(Path(__file__)),fingerprint(Path(__file__).with_name("probe_terrain_visibility.py"))],
             "limits":"Timing on the calling host identified by gpu/torch, not full-dataset throughput or challenge runtime compliance. Default compares one map; verify_all compares eight maps and their D4 transforms. These are numerical implementation checks against a bilinear/sample reference, not original ray-tracing geometry validation."}
    (out/"cuda_benchmark.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:audit[key] for key in ("mean_seconds_per_map","gpu","scope","limits")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--metadata",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--verify-all",action="store_true",help="Compare all eight maps to FP64 and verify GPU D4 equivariance")
    args=parser.parse_args()
    if args.repeats<1:
        parser.error("--repeats must be positive")
    analyze(args.source,args.metadata,args.out,args.repeats,args.verify_all)
