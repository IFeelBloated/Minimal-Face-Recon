from PIL import Image
import numpy as np
from scipy.io import loadmat
import torch
from torchvision.utils import save_image
from ReconModels.Model import ReconNetWrapper
from Renderer.ParametricFaceModel import ParametricFaceModel
from Renderer.MeshRenderer import MeshRenderer

def load_lm3d(file_path):
    Lm3D = loadmat(file_path)
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D

def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    img = img.resize((w, h), resample=Image.BICUBIC)
    img = img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.BICUBIC)
        mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return img, lm, mask

def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    t, s = POS(lm5p.transpose(), lm3D.transpose())
    s = rescale_factor/s

    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    trans_params = np.array([w0, h0, s, t[0], t[1]])

    return trans_params, img_new, lm_new, mask_new

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm

lm3d_path = 'PretrainedReconCheckpoints/similarity_Lm3D_all.mat'
recon_path = 'PretrainedReconCheckpoints/epoch_20.pth'
resnet_path = 'PretrainedReconCheckpoints/resnet50-0676ba61.pth'
bfm_path = 'PretrainedReconCheckpoints/BFM_model_front.mat'

camera_distance=10. 
focal=1015. 
center=112.
rasterize_size = int(2 * center)
fov = 2 * np.arctan(center / focal) * 180 / np.pi
znear = 5.0
zfar=15.0

Deep3DRecon = ReconNetWrapper('resnet50', use_last_fc=False, init_path=resnet_path)
Deep3DRecon.load_state_dict(torch.load(recon_path, map_location='cpu')['net_recon'])
Deep3DRecon = Deep3DRecon.cuda()
Deep3DRecon.eval()

FaceModel = ParametricFaceModel(bfm_path, camera_distance=camera_distance, focal=focal, center=center)
FaceModel = FaceModel.cuda()
FaceModel.eval()

Renderer = MeshRenderer(fov, znear, zfar, rasterize_size, use_opengl=False)
Renderer = Renderer.cuda()
Renderer.eval()


##################################


test_img_path = 'test_images/000007.jpg'
test_lm_path = 'test_images/detections/000007.txt'

img, lm = read_data(test_img_path, test_lm_path, load_lm3d(lm3d_path))
img = img.cuda()

m = Deep3DRecon(img)
face_vertex, face_texture, face_color, landmark = FaceModel(m)
mask, depth, image = Renderer(face_vertex, FaceModel.face_buf, feat=face_color)

overlay = image * mask + (1 - mask) * img
save_image(torch.cat([img[0], image[0], overlay[0]], dim=2), 'b.png')