import numpy as np
import torch

def readVariable(fid, name, M, N):
    # rewind
    fid.seek(0, 0)
    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break
    # return if variable identifier not found
    if success == 0:
        return None
    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert (len(line) == M * N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)
    return mat

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array([0.1, 0.1, 0.1, 1.])
    hwf = c2w[3:, :]
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # import ipdb; ipdb.set_trace()
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 0))
    return render_poses

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def loadCalibrationCameraToPose(filename):
    # open file
    fid = open(filename, 'r')
    # read variables
    Tr = {}
    cameras = ['image_00', 'image_01', 'image_02', 'image_03']
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))
    # close file
    fid.close()
    return Tr

def to_cuda(batch, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if isinstance(batch, tuple) or isinstance(batch, list):
        batch = [to_cuda(b, device) for b in batch]
    elif isinstance(batch, dict):
        batch_ = {}
        for key in batch:
            if key == 'meta':
                batch_[key] = batch[key]
            else:
                batch_[key] = to_cuda(batch[key], device)
        batch = batch_
    else:
        batch = batch.to(device)
    return batch

def build_rays(ixt, c2w, H, W):
    """Compute rays of each picture

    Args:
        ixt: intrinsic matrix of the camera
        c2w: pose of the picture
        H: height of the picture
        W: width of the picture

    Returns:
        rays_o: origins of the rays, [H*w, 3]
        rays_d: directions of the rays, [H*w, 3]
    
    """
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
    XYZ = XYZ @ np.linalg.inv(ixt[:3, :3]).T
    XYZ = XYZ @ c2w[:3, :3].T
    rays_d = XYZ.reshape(-1, 3)
    rays_o = c2w[:3, 3]
    return np.concatenate((rays_o[None].repeat(len(rays_d), 0), rays_d), axis=-1) 