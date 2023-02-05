import torch
from torchvision.utils import save_image
from ReconModels.Model import ReconNetWrapper
from Renderer.FaceRenderer import FaceRenderer
from FaceAlignTools import create_data_reader

lm3d_path = 'PretrainedReconCheckpoints/similarity_Lm3D_all.mat'
recon_path = 'PretrainedReconCheckpoints/epoch_20.pth'
bfm_path = 'PretrainedReconCheckpoints/BFM_model_front.mat'

Reader = create_data_reader(lm3d_path, resolution=512)

Deep3DRecon = ReconNetWrapper('resnet50', use_last_fc=False)
Deep3DRecon.load_state_dict(torch.load(recon_path, map_location='cpu')['net_recon'])
Deep3DRecon = Deep3DRecon.cuda()
Deep3DRecon.eval()

Renderer = FaceRenderer(bfm_path, resolution=512)
Renderer = Renderer.cuda()
Renderer.eval()


##################################


test_img_path = 'test_images/000002.jpg'
test_lm_path = 'test_images/detections/000002.txt'

img, lm = Reader(test_img_path, test_lm_path)
img = img.cuda()

m = Deep3DRecon(img)
mask, depth, image = Renderer(m)

overlay = image * mask + (1 - mask) * img
save_image(torch.cat([img[0], image[0], overlay[0]], dim=2), 'a.png')