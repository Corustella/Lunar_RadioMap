"""FP64 CUDA implementation of the deterministic D/S reference geometry."""
import cmath
import math
import os
import sys
import time
import numpy as np
from numba import config

# The official NVIDIA binding resolves API versions explicitly. Numba 0.61's
# ctypes binding selects an incompatible cuCtxGetDevice_v2 on CUDA 13 drivers.
if "numba.cuda" in sys.modules and not config.CUDA_USE_NVIDIA_BINDING:
    raise RuntimeError("Set NUMBA_CUDA_USE_NVIDIA_BINDING=1 before importing numba.cuda")
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
config.reload_config()
if not config.CUDA_USE_NVIDIA_BINDING:
    raise RuntimeError("CUDA geometry requires cuda-python; install requirements-mechanisms.txt")
from numba import cuda, float64, int32


@cuda.jit(device=True)
def terrain_z(h, x, y):
    r = min(max(int(math.floor(y)), 0), h.shape[0]-2)
    c = min(max(int(math.floor(x)), 0), h.shape[1]-2)
    u, v = x-c, y-r
    if v <= u:
        return h[r,c]+u*(h[r,c+1]-h[r,c])+v*(h[r+1,c+1]-h[r,c+1])
    return h[r,c]+u*(h[r+1,c+1]-h[r+1,c])+v*(h[r+1,c]-h[r,c])


@cuda.jit(device=True)
def crossing(start, delta):
    if delta > 1e-15:
        return (math.floor(start+1e-12)+1-start)/delta, 1/delta
    if delta < -1e-15:
        return (math.ceil(start-1e-12)-1-start)/delta, -1/delta
    return math.inf, math.inf


@cuda.jit(device=True)
def visible(h, a, b, spacing, epsilon):
    x, y = a[0]/spacing, a[1]/spacing
    dx, dy, dz = (b[0]-a[0])/spacing, (b[1]-a[1])/spacing, b[2]-a[2]
    if a[2] < terrain_z(h,x,y)-epsilon or b[2] < terrain_z(h,x+dx,y+dy)-epsilon:
        return False
    tx, sx = crossing(x,dx)
    ty, sy = crossing(y,dy)
    td, sd = crossing(x-y,dx-dy)
    while True:
        t = min(tx,ty,td)
        if t >= 1-1e-10:
            break
        if t > 1e-10 and a[2]+t*dz-terrain_z(h,x+t*dx,y+t*dy) <= epsilon:
            return False
        if tx <= t+1e-12:
            tx += sx
        if ty <= t+1e-12:
            ty += sy
        if td <= t+1e-12:
            td += sd
    return a[2]+.5*dz-terrain_z(h,x+.5*dx,y+.5*dy) > epsilon


@cuda.jit
def visibility_kernel(h, source, receivers, spacing, epsilon, out):
    i = cuda.grid(1)
    if i < len(receivers):
        out[i] = visible(h,source,receivers[i],spacing,epsilon)


@cuda.jit(device=True)
def dot3(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


@cuda.jit(device=True)
def contains_xy(q, face, spacing, epsilon):
    u, v = (q[0]-face[0,0])/spacing, (q[1]-face[0,1])/spacing
    tol = epsilon/spacing
    if u < -tol or v < -tol or u > 1+tol or v > 1+tol:
        return False
    if face[1,1] == face[0,1]:
        return v <= u+tol
    return v >= u-tol


@cuda.jit(device=True)
def shared_edge_owner(q, fid, faces, normals, rows, cols, spacing, epsilon):
    c, r = int(math.floor(q[0]/spacing)), int(math.floor(q[1]/spacing))
    for rr in range(max(0,r-1), min(rows-2,r)+1):
        for cc in range(max(0,c-1), min(cols-2,c)+1):
            for kind in range(2):
                other = 2*(rr*(cols-1)+cc)+kind
                if other >= fid:
                    continue
                difference = 0.
                for axis in range(3):
                    difference = max(difference,abs(normals[fid,axis]-normals[other,axis]))
                if difference > 1e-12:
                    continue
                plane = 0.
                for axis in range(3):
                    plane += normals[fid,axis]*(faces[other,0,axis]-faces[fid,0,axis])
                if abs(plane) <= epsilon and contains_xy(q,faces[other],spacing,epsilon):
                    return False
    return True


@cuda.jit(device=True)
def reflected_power(ki, normal, eta, thickness, wavelength, scattering):
    cosine = min(1.,abs(dot3(ki,normal)))
    root = cmath.sqrt(eta-1+cosine*cosine)
    r_te = (cosine-root)/(cosine+root)
    r_tm = (eta*cosine-root)/(eta*cosine+root)
    exp2q = cmath.exp(-4j*math.pi*thickness/wavelength*root)
    r_te = r_te*(1-exp2q)/(1-r_te*r_te*exp2q)
    r_tm = r_tm*(1-exp2q)/(1-r_tm*r_tm*exp2q)
    sx = ki[1]*normal[2]-ki[2]*normal[1]
    sy = ki[2]*normal[0]-ki[0]*normal[2]
    sz = ki[0]*normal[1]-ki[1]*normal[0]
    norm2 = sx*sx+sy*sy+sz*sz
    if norm2 < 1e-24:
        te_weight = 1.
    else:
        horizontal = math.hypot(ki[0],ki[1])
        if horizontal < 1e-12:
            weight = ki[2]*sx
        else:
            weight = ki[0]*ki[2]/horizontal*sx+ki[1]*ki[2]/horizontal*sy-horizontal*sz
        te_weight = min(1.,weight*weight/norm2)
    return (1-scattering*scattering)*(abs(r_te)**2*te_weight+abs(r_tm)**2*(1-te_weight))


@cuda.jit(device=True)
def one_path(height, source, target, fid, faces, normals, mirror, spacing,
             epsilon, eta, thickness, wavelength, scattering):
    a, n = faces[fid,0], normals[fid]
    denom, numerator = 0., 0.
    for axis in range(3):
        denom += n[axis]*(target[axis]-mirror[axis])
        numerator += n[axis]*(a[axis]-mirror[axis])
    if abs(denom) < epsilon:
        return 0.,0.
    t = numerator/denom
    if t <= 0 or t >= 1:
        return 0.,0.
    q = cuda.local.array(3,float64)
    for axis in range(3):
        q[axis] = mirror[axis]+t*(target[axis]-mirror[axis])
    if not contains_xy(q,faces[fid],spacing,epsilon):
        return 0.,0.
    if not shared_edge_owner(q,fid,faces,normals,height.shape[0],height.shape[1],spacing,epsilon):
        return 0.,0.
    if not visible(height,source,q,spacing,epsilon) or not visible(height,q,target,spacing,epsilon):
        return 0.,0.
    leg = cuda.local.array(3,float64)
    d1, d2, d = 0.,0.,0.
    for axis in range(3):
        leg[axis] = q[axis]-source[axis]
        d1 += leg[axis]*leg[axis]
        d2 += (target[axis]-q[axis])**2
        d += (target[axis]-source[axis])**2
    d1, d2, d = math.sqrt(d1),math.sqrt(d2),math.sqrt(d)
    if d1 <= epsilon or d2 <= epsilon or d <= epsilon:
        return 0.,0.
    length = d1+d2
    for axis in range(3):
        leg[axis] /= d1
    gain = (wavelength/(4*math.pi*length))**2*reflected_power(
        leg,n,eta,thickness,wavelength,scattering)
    return gain,gain*max(length/d-1,0.)


@cuda.jit
def reflection_kernel(height, source, points, faces, normals, lower, upper,
                      offsets, leaf, spacing, epsilon, eta, thickness,
                      wavelength, scattering, counts, pair_offsets, pairs, write):
    f = cuda.grid(1)
    if f >= len(faces):
        return
    signed = 0.
    for axis in range(3):
        signed += normals[f,axis]*(source[axis]-faces[f,0,axis])
    if abs(signed) <= epsilon:
        return
    mirror = cuda.local.array(3,float64)
    inward = cuda.local.array((4,3),float64)
    origin = cuda.local.array((4,3),float64)
    u, v = cuda.local.array(3,float64),cuda.local.array(3,float64)
    for axis in range(3):
        mirror[axis] = source[axis]-2*signed*normals[f,axis]
    for edge in range(3):
        for axis in range(3):
            u[axis] = faces[f,edge,axis]-mirror[axis]
            v[axis] = faces[f,(edge+1)%3,axis]-mirror[axis]
        inward[edge,0] = u[1]*v[2]-u[2]*v[1]
        inward[edge,1] = u[2]*v[0]-u[0]*v[2]
        inward[edge,2] = u[0]*v[1]-u[1]*v[0]
        orientation, norm = 0.,0.
        for axis in range(3):
            orientation += inward[edge,axis]*(faces[f,(edge+2)%3,axis]-mirror[axis])
            norm += inward[edge,axis]**2
        sign = 1. if orientation >= 0 else -1.
        for axis in range(3):
            inward[edge,axis] *= sign/math.sqrt(norm)
            origin[edge,axis] = mirror[axis]
    for axis in range(3):
        inward[3,axis] = normals[f,axis] if signed > 0 else -normals[f,axis]
        origin[3,axis] = faces[f,0,axis]
    stack = cuda.local.array(64,int32)
    stack[0], size = 0,1
    tests, candidates, valid = 0,0,0
    while size > 0:
        size -= 1
        node = stack[size]
        if lower[node,0] > upper[node,0]:
            continue
        tests += 1
        rejected = False
        for plane in range(4):
            far = 0.
            for axis in range(3):
                bound = upper[node,axis] if inward[plane,axis] >= 0 else lower[node,axis]
                far += inward[plane,axis]*(bound-origin[plane,axis])
            if far < -epsilon:
                rejected = True
                break
        if rejected:
            continue
        if node < leaf:
            for child in range(4):
                stack[size] = 4*node+1+child
                size += 1
        else:
            cell = node-leaf
            for p in range(offsets[cell],offsets[cell+1]):
                if write:
                    pair = pair_offsets[f]+candidates
                    pairs[pair,0],pairs[pair,1] = f,p
                candidates += 1
    counts[f,0],counts[f,1],counts[f,2] = tests,candidates,valid


@cuda.jit
def path_kernel(height, source, points, faces, normals, spacing, epsilon,
                eta, thickness, wavelength, scattering, pairs, accum, counts):
    i = cuda.grid(1)
    if i >= len(pairs):
        return
    f,p = pairs[i,0],pairs[i,1]
    signed = 0.
    mirror = cuda.local.array(3,float64)
    for axis in range(3):
        signed += normals[f,axis]*(source[axis]-faces[f,0,axis])
    for axis in range(3):
        mirror[axis] = source[axis]-2*signed*normals[f,axis]
    g,u = one_path(height,source,points[p],f,faces,normals,mirror,
                   spacing,epsilon,eta,thickness,wavelength,scattering)
    if g > 0:
        cuda.atomic.add(accum,(p,0),g)
        cuda.atomic.add(accum,(p,1),u)
        cuda.atomic.add(accum,(p,2),1.)
        cuda.atomic.add(counts,(f,2),1)


def visibility(h,source,receivers,spacing,epsilon):
    out = cuda.device_array(len(receivers),np.float64)
    visibility_kernel[(len(receivers)+127)//128,128](
        cuda.to_device(h),cuda.to_device(source),cuda.to_device(receivers),spacing,epsilon,out)
    return out.copy_to_host()


def reflected(height,source,points,faces,normals,lower,upper,offsets,leaf,
              spacing,epsilon,eta,thickness,wavelength,scattering,brute=False):
    if brute:
        raise ValueError('Use CPU brute-force reference')
    start = time.perf_counter()
    accum = cuda.to_device(np.zeros((len(points),3),np.float64))
    counts = cuda.to_device(np.zeros((len(faces),3),np.int64))
    arrays = [cuda.to_device(v) for v in (height,source,points,faces,normals,lower,upper,offsets)]
    pair_offsets = cuda.device_array(len(faces)+1,np.int64)
    pairs = cuda.device_array((1,2),np.int32)
    cuda.synchronize()
    upload_end = time.perf_counter()
    reflection_kernel[(len(faces)+63)//64,64](*arrays,leaf,spacing,epsilon,eta,
        thickness,wavelength,scattering,counts,pair_offsets,pairs,False)
    host_counts = counts.copy_to_host()
    count_end = time.perf_counter()
    starts = np.concatenate(([0],np.cumsum(host_counts[:,1])))
    if starts[-1] > 2**28:
        raise MemoryError('Reflection candidates exceed 2 GiB; no paths were truncated')
    pair_offsets = cuda.to_device(starts)
    fill_end = count_end
    path_end = count_end
    if starts[-1]:
        pairs = cuda.device_array((int(starts[-1]),2),np.int32)
        reflection_kernel[(len(faces)+63)//64,64](*arrays,leaf,spacing,epsilon,eta,
            thickness,wavelength,scattering,counts,pair_offsets,pairs,True)
        cuda.synchronize()
        fill_end = time.perf_counter()
        path_kernel[(len(pairs)+127)//128,128](*arrays[:5],spacing,epsilon,eta,
            thickness,wavelength,scattering,pairs,accum,counts)
        cuda.synchronize()
        path_end = time.perf_counter()
    host_accum,host_counts = accum.copy_to_host(),counts.copy_to_host().sum(0)
    end = time.perf_counter()
    diagnostics = {
        "cuda_upload_ms": (upload_end-start)*1000,
        "cone_count_and_download_ms": (count_end-upload_end)*1000,
        "cone_fill_ms": (fill_end-count_end)*1000,
        "path_visibility_power_ms": (path_end-fill_end)*1000,
        "cuda_download_ms": (end-path_end)*1000,
        "candidate_tree_passes": 2 if starts[-1] else 1,
        "device_array_bytes": int(sum(v.nbytes for v in arrays)+accum.nbytes+counts.nbytes+
                                  pair_offsets.nbytes+pairs.nbytes),
        "memory_scope": "Live reflection arrays only; excludes CUDA/JIT context, pending frees and torch",
    }
    return host_accum,host_counts,diagnostics
