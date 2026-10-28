"""Deterministic heightfield visibility and single-bounce image-source inputs.

Implementation of the reviewed D/S feature proposal. All geometry is FP64. There is
no angular window, nearest-face truncation, ray shooting, or label/mask input.
"""

import math
import time

import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads, get_thread_id


@njit(inline="always", cache=True)
def dot3(a, b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


@njit(cache=True)
def terrain_z(h, x, y):
    r = min(max(int(math.floor(y)), 0), h.shape[0] - 2)
    c = min(max(int(math.floor(x)), 0), h.shape[1] - 2)
    u, v = x - c, y - r
    if v <= u:
        return h[r, c] + u*(h[r, c+1]-h[r, c]) + v*(h[r+1, c+1]-h[r, c+1])
    return h[r, c] + u*(h[r+1, c+1]-h[r+1, c]) + v*(h[r+1, c]-h[r, c])


@njit(cache=True)
def crossing(start, delta):
    if delta > 1e-15:
        return (math.floor(start + 1e-12) + 1 - start)/delta, 1/delta
    if delta < -1e-15:
        return (math.ceil(start - 1e-12) - 1 - start)/delta, -1/delta
    return math.inf, math.inf


@njit(cache=True)
def visible(h, a, b, spacing=1.0, epsilon=1e-7):
    """Exact piecewise-linear clearance at x, y and x-y grid crossings.

    Surface endpoints are allowed (reflection vertex). Interior contacts and
    below-surface endpoints are blocked. No fixed-step sampling is used.
    """
    x, y = a[0]/spacing, a[1]/spacing
    dx, dy, dz = (b[0]-a[0])/spacing, (b[1]-a[1])/spacing, b[2]-a[2]
    if a[2] < terrain_z(h,x,y)-epsilon or b[2] < terrain_z(h,x+dx,y+dy)-epsilon:
        return False
    tx, sx = crossing(x, dx)
    ty, sy = crossing(y, dy)
    td, sd = crossing(x-y, dx-dy)
    while True:
        t = min(tx, ty, td)
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
    # Midpoint also detects a ray lying in a triangle's plane with no crossing.
    return a[2]+.5*dz-terrain_z(h,x+.5*dx,y+.5*dy) > epsilon


@njit(parallel=True, cache=True)
def visibility(h, source, receivers, spacing, epsilon):
    out = np.empty(len(receivers), np.float64)
    for i in prange(len(receivers)):
        out[i] = visible(h, source, receivers[i], spacing, epsilon)
    return out


@njit(cache=True)
def mesh(height, spacing):
    rows, cols = height.shape
    faces = np.empty((2*(rows-1)*(cols-1),3,3),np.float64)
    normals = np.empty((len(faces),3),np.float64)
    for r in range(rows-1):
        for c in range(cols-1):
            for kind in range(2):
                f = 2*(r*(cols-1)+c)+kind
                for v in range(3):
                    rr = r+int(v==2 or (v==1 and kind==1))
                    cc = c+int(v==1 or (v==2 and kind==0))
                    faces[f,v,0],faces[f,v,1],faces[f,v,2] = cc*spacing,rr*spacing,height[rr,cc]
                ux,uy,uz = faces[f,1,0]-faces[f,0,0],faces[f,1,1]-faces[f,0,1],faces[f,1,2]-faces[f,0,2]
                vx,vy,vz = faces[f,2,0]-faces[f,0,0],faces[f,2,1]-faces[f,0,1],faces[f,2,2]-faces[f,0,2]
                nx,ny,nz = uy*vz-uz*vy,uz*vx-ux*vz,ux*vy-uy*vx
                norm = math.sqrt(nx*nx+ny*ny+nz*nz)
                normals[f,0],normals[f,1],normals[f,2] = nx/norm,ny/norm,nz/norm
    return faces, normals


@njit(cache=True)
def receiver_tree(points, shape, spacing):
    """Complete Morton-ordered quadtree with tight 3D point bounds."""
    depth = int(math.ceil(math.log2(max(shape)-1)))
    side = 2**depth
    morton = np.zeros(len(points), np.int64)
    count = np.zeros(side*side,np.int64)
    for p in range(len(points)):
        x = min(max(int(math.floor(points[p,0]/spacing)),0),shape[1]-2)
        y = min(max(int(math.floor(points[p,1]/spacing)),0),shape[0]-2)
        for bit in range(depth):
            morton[p] |= ((x >> bit) & 1) << (2*bit)
            morton[p] |= ((y >> bit) & 1) << (2*bit+1)
        count[morton[p]] += 1
    offsets = np.zeros(side*side+1,np.int64)
    for i in range(side*side):
        offsets[i+1] = offsets[i]+count[i]
    cursor = offsets[:-1].copy()
    order = np.empty(len(points),np.int64)
    for p in range(len(points)):
        cell = morton[p]
        order[cursor[cell]] = p
        cursor[cell] += 1
    leaf = (side*side-1)//3
    lower = np.full((leaf+side*side,3), np.inf)
    upper = np.full_like(lower, -np.inf)
    for p in range(len(points)):
        for axis in range(3):
            lower[leaf+morton[p],axis] = min(lower[leaf+morton[p],axis],points[p,axis])
            upper[leaf+morton[p],axis] = max(upper[leaf+morton[p],axis],points[p,axis])
    for level in range(depth-1,-1,-1):
        first, n = (4**level-1)//3, 4**level
        child = 4*first+1
        for i in range(n):
            for axis in range(3):
                for k in range(4):
                    lower[first+i,axis] = min(lower[first+i,axis],lower[child+4*i+k,axis])
                    upper[first+i,axis] = max(upper[first+i,axis],upper[child+4*i+k,axis])
    return points[order], order, lower, upper, offsets, leaf


@njit(cache=True)
def receiver_points(faces,weights,rx_height):
    points = np.zeros((len(faces)*len(weights),3),np.float64)
    for f in range(len(faces)):
        for q in range(len(weights)):
            p = f*len(weights)+q
            for axis in range(3):
                for v in range(3):
                    points[p,axis] += weights[q,v]*faces[f,v,axis]
            points[p,2] += rx_height
    return points


def subtriangle_weights(subdivisions):
    """Centroids of n*n equal-area subtriangles in barycentric coordinates."""
    result = []
    for i in range(subdivisions):
        for j in range(subdivisions-i):
            for shift in ([1/3,2/3] if i+j < subdivisions-1 else [1/3]):
                u,v = (i+shift)/subdivisions,(j+shift)/subdivisions
                result.append([1-u-v,u,v])
    return np.asarray(result,np.float64)


@njit(cache=True)
def reflected_power(ki, normal, eta, thickness, wavelength, scattering):
    """Total isotropic-probe power for the actual V-polarized iso TX field.

    TE/TM are orthogonal, so the squared Jones norm reduces to the weighted
    sum below. Slab equations match Sionna RT 1.1.0 electromagnetics.py 33:106.
    No Monte Carlo selection-probability compensation is applied.
    """
    cosine = min(1.0, abs(dot3(ki,normal)))
    root = np.sqrt(eta-1+cosine*cosine)
    r_te = (cosine-root)/(cosine+root)
    r_tm = (eta*cosine-root)/(eta*cosine+root)
    exp2q = np.exp(-4j*math.pi*thickness/wavelength*root)
    r_te = r_te*(1-exp2q)/(1-r_te*r_te*exp2q)
    r_tm = r_tm*(1-exp2q)/(1-r_tm*r_tm*exp2q)
    transverse = np.cross(ki,normal)
    norm = np.sqrt(dot3(transverse,transverse))
    if norm < 1e-12:
        te_weight = 1.0
    else:
        horizontal = math.hypot(ki[0],ki[1])
        if horizontal < 1e-12:
            e_v = np.array([ki[2],0.,0.])
        else:
            e_v = np.array([ki[0]*ki[2]/horizontal,ki[1]*ki[2]/horizontal,-horizontal])
        te_weight = min(1.0,(dot3(e_v,transverse)/norm)**2)
    return (1-scattering*scattering)*(abs(r_te)**2*te_weight+abs(r_tm)**2*(1-te_weight))


@njit(cache=True)
def contains_xy(q, face, spacing, epsilon):
    u, v = (q[0]-face[0,0])/spacing, (q[1]-face[0,1])/spacing
    tol = epsilon/spacing
    if u < -tol or v < -tol or u > 1+tol or v > 1+tol:
        return False
    if face[1,1] == face[0,1]:
        return v <= u+tol
    return v >= u-tol


@njit(cache=True)
def shared_edge_owner(q, fid, faces, normals, shape, spacing, epsilon):
    """Deduplicate only identical paths on coplanar adjacent triangles."""
    c, r = int(math.floor(q[0]/spacing)), int(math.floor(q[1]/spacing))
    for rr in range(max(0,r-1),min(shape[0]-2,r)+1):
        for cc in range(max(0,c-1),min(shape[1]-2,c)+1):
            for kind in range(2):
                other = 2*(rr*(shape[1]-1)+cc)+kind
                if other >= fid:
                    continue
                if np.max(np.abs(normals[fid]-normals[other])) > 1e-12:
                    continue
                if abs(dot3(normals[fid],faces[other,0]-faces[fid,0])) > epsilon:
                    continue
                if contains_xy(q,faces[other],spacing,epsilon):
                    return False
    return True


@njit(cache=True)
def one_path(height, source, target, fid, faces, normals, mirror, spacing,
             epsilon, eta, thickness, wavelength, scattering):
    a, n = faces[fid,0], normals[fid]
    denom = dot3(n,target-mirror)
    if abs(denom) < epsilon:
        return 0.,0.
    t = dot3(n,a-mirror)/denom
    if t <= 0 or t >= 1:
        return 0.,0.
    q = mirror+t*(target-mirror)
    if not contains_xy(q,faces[fid],spacing,epsilon):
        return 0.,0.
    if not shared_edge_owner(q,fid,faces,normals,height.shape,spacing,epsilon):
        return 0.,0.
    if not visible(height,source,q,spacing,epsilon) or not visible(height,q,target,spacing,epsilon):
        return 0.,0.
    leg = q-source
    d1 = np.sqrt(dot3(leg,leg))
    d2 = np.sqrt(dot3(target-q,target-q))
    d = np.sqrt(dot3(target-source,target-source))
    length = d1+d2
    if d1 <= epsilon or d2 <= epsilon or d <= epsilon:
        return 0.,0.
    gain = (wavelength/(4*math.pi*length))**2*reflected_power(
        leg/d1,n,eta,thickness,wavelength,scattering)
    return gain, gain*max(length/d-1,0.)


@njit(parallel=True, cache=True)
def reflected(height, source, points, faces, normals, lower, upper, offsets, leaf,
              spacing, epsilon, eta, thickness, wavelength, scattering, brute=False):
    accum = np.zeros((get_num_threads(),len(points),3), np.float64)
    counts = np.zeros((len(faces),3),np.int64)
    for f in prange(len(faces)):
        n = normals[f]
        signed = dot3(n,source-faces[f,0])
        if abs(signed) <= epsilon:
            continue  # exactly grazing, zero solid-angle source-coplanar face
        mirror = source-2*signed*n
        inward = np.empty((4,3),np.float64)
        origin = np.empty((4,3),np.float64)
        for edge in range(3):
            m = np.cross(faces[f,edge]-mirror, faces[f,(edge+1)%3]-mirror)
            if dot3(m,faces[f,(edge+2)%3]-mirror) < 0:
                m = -m
            inward[edge] = m/np.sqrt(dot3(m,m))
            origin[edge] = mirror
        inward[3] = n if signed > 0 else -n
        origin[3] = faces[f,0]
        stack = np.empty(64,np.int64)
        stack[0], size = 0, 1
        thread = get_thread_id()
        while size > 0:
            size -= 1
            node = stack[size]
            if lower[node,0] > upper[node,0]:
                continue
            counts[f,0] += 1
            rejected = False
            if not brute:
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
                    counts[f,1] += 1
                    g,u = one_path(height,source,points[p],f,faces,normals,mirror,
                                   spacing,epsilon,eta,thickness,wavelength,scattering)
                    if g > 0:
                        counts[f,2] += 1
                        accum[thread,p,0] += g
                        accum[thread,p,1] += u
                        accum[thread,p,2] += 1
    return accum.sum(axis=0),counts.sum(axis=0)


def project(cells, border, border_mask, shape):
    c = cells.reshape(shape[0]-1,shape[1]-1,2)
    out = np.empty(shape,np.float64)
    out[1:-1,1:-1] = .5*(c[1:,:-1,0]+c[:-1,1:,1])
    out[border_mask] = border
    return out


def compute(height, tx, *, direct=True, specular=True, threads=8, spacing=1.,
            tx_height=3., rx_height=1., specular_quadrature=1, brute=False,
            backend="cpu", device_index=0, frequency=5775000000., permittivity=3.411,
            conductivity=.009324, thickness=10., scattering=.632, direct_refinement=8):
    start = time.perf_counter()
    height = np.ascontiguousarray(height,dtype=np.float64)
    if height.ndim != 2 or min(height.shape)<3 or max(height.shape)>1024 or not np.isfinite(height).all():
        raise ValueError("Expected a finite 3..1024 height grid")
    if not (0<=tx[0]<height.shape[0] and 0<=tx[1]<height.shape[1]):
        raise ValueError("TX outside grid")
    if (spacing<=0 or tx_height<=0 or rx_height<=0 or threads<1 or
            specular_quadrature not in (1,3) or backend not in ("cpu","cuda") or
            direct_refinement not in (0,8) or
            frequency<=0 or permittivity<=0 or conductivity<0 or thickness<=0 or
            not 0<=scattering<=1):
        raise ValueError("Invalid geometry/quadrature")
    visible_query, reflection_query = visibility, reflected
    if backend == "cuda":
        from mechanism_geometry_cuda import cuda, visibility as visible_query, reflected as reflection_query
        cuda.select_device(device_index)
    old_threads = get_num_threads()
    set_num_threads(min(threads,old_threads))
    try:
        faces,normals = mesh(height,spacing)
        source = np.array([tx[1]*spacing,tx[0]*spacing,height[tx]+tx_height])
        border_mask = np.ones(height.shape,bool)
        border_mask[1:-1,1:-1] = False
        rows,cols = np.where(border_mask)
        border = np.stack((cols*spacing,rows*spacing,height[border_mask]+rx_height),axis=1)
        bary = np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
        times = {"mesh_ms":(time.perf_counter()-start)*1000}
        output = np.zeros((4,*height.shape),np.float32)
        refined_faces = 0
        t = time.perf_counter()
        if direct:
            receivers = receiver_points(faces,bary,rx_height)
            v = visible_query(height,source,np.concatenate((receivers,border)),spacing,1e-7)
            fraction = v[:len(receivers)].reshape(-1,3).mean(1)
            partial = np.flatnonzero((fraction>0)&(fraction<1))
            if direct_refinement and len(partial):
                weights = subtriangle_weights(direct_refinement)
                refined = receiver_points(faces[partial],weights,rx_height)
                refined_v = visible_query(height,source,refined,spacing,1e-7)
                fraction[partial] = refined_v.reshape(-1,len(weights)).mean(1)
                refined_faces = len(partial)
            output[0] = project(fraction,v[len(receivers):],border_mask,height.shape)
        times['direct_ms'] = (time.perf_counter()-t)*1000
        counts = [0,0,0]
        diagnostics = None
        if specular:
            t = time.perf_counter()
            weights = np.full((1,3),1/3) if specular_quadrature == 1 else bary
            receivers = receiver_points(faces,weights,rx_height)
            points = np.concatenate((receivers,border))
            sorted_points,order,lower,upper,offsets,leaf = receiver_tree(points,height.shape,spacing)
            times['tree_ms'] = (time.perf_counter()-t)*1000
            t = time.perf_counter()
            wavelength = 299792458/frequency
            eta = complex(permittivity,-conductivity/(2*math.pi*frequency*8.8541878128e-12))
            result = reflection_query(height,source,sorted_points,faces,normals,lower,upper,offsets,leaf,
                                      spacing,1e-7,eta,thickness,wavelength,scattering,brute)
            acc,counts = result[:2]
            if backend == "cuda":
                diagnostics = result[2]
            times['specular_ms'] = (time.perf_counter()-t)*1000
            t = time.perf_counter()
            values = np.empty_like(acc)
            values[order] = acc
            values[:,2] = values[:,2] > 0
            free = (wavelength/(4*math.pi))**2/np.square(points-source).sum(1)
            values = np.column_stack((values,free))
            projected = []
            for channel in range(4):
                cells = values[:len(receivers),channel].reshape(-1,specular_quadrature).mean(1)
                projected.append(project(cells,values[len(receivers):,channel],border_mask,height.shape))
            gain,excess,exists,free = projected
            output[1] = exists
            output[2] = np.log1p(gain/free)
            output[3] = np.log1p(np.divide(excess,gain,out=np.zeros_like(excess),where=gain>0))
            times['projection_ms'] = (time.perf_counter()-t)*1000
        if not np.isfinite(output).all() or np.any(output<0) or np.any(output[:2]>1):
            raise FloatingPointError("Invalid physical features")
        coplanar = int(np.count_nonzero(np.abs(np.sum(normals*(source-faces[:,0]),axis=1))<=1e-7))
        times['total_ms'] = (time.perf_counter()-start)*1000
        return output,{"times":times,"tree_tests":int(counts[0]),"point_candidates":int(counts[1]),
                       "valid_paths":int(counts[2]),"threads":get_num_threads(),"truncated":False,
                       "backend":backend,"specular_quadrature":specular_quadrature,
                       "direct_refined_faces":refined_faces,
                       "source_coplanar_faces":coplanar,"reflection_diagnostics":diagnostics}
    finally:
        set_num_threads(old_threads)
