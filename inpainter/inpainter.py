import os
from tqdm import tqdm

# image loading
from PIL import Image

import numpy as np
from argparse import Namespace
# pytorch app
import torch

# stylegan2-pytorch model

# torchvision model zoo - vgg16
import torchvision

# transform input image into tensors
from torchvision import transforms, utils

from stylegan2_model import Generator
from ffhq_align import *
from landmark_detector import *
from mobile_face_net import *
from psp.psp import pSp
LANDMARK_INDICES_PATH = os.path.join(os.path.dirname(__file__), 'kpt_ind.npy')

class UnNormalize(object):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: UnNormalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0.0, 1.0)


class FANNormalize(object):

    def __call__(self, tensor):
        return tensor / 255.


class Inpainter():
    def __init__(self, stylegan2_ckpt=os.path.join(os.path.dirname(__file__), 'checkpoint/stylegan2-ffhq-config-f.pt'), 
                        face_recognition_ckpt = os.path.join(os.path.dirname(__file__), 'checkpoint/model_mobilefacenet.pth'), 
                        landmark_detector_ckpt=os.path.join(os.path.dirname(__file__), 'checkpoint/DAFAN-4.pth.tar'),
                        regressor_ckpt = os.path.join(os.path.dirname(__file__), 'checkpoint/psp_ffhq_encode.pt')):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #### HYPERPARAMETERS
        ### TRAINING
        self.lambda_photometric = 1
        self.lambda_identity = 1
        self.lambda_landmark = 1e-1
        self.lambda_perceptual = 1
        self.lr = 2e-2
        self.iterations = 150

        ### MODELS
        self.size = 1024 # input image size
        self.regressor_size = 256 # psp encoder input size
        self.landmark_size = 256 # landmark detector input size
        self.perceptual_size = 224 # vgg input size
        self.face_recognition_size = 112 # mobilefacenet input size
        self.num_layers = 18

        print('loading stylegan2 model')
        g_ema = Generator(self.size, 512, 8)
        g_ema.load_state_dict(torch.load(stylegan2_ckpt)["g_ema"], strict=False)
        g_ema.eval()
        self.g_ema = g_ema.to(self.device)
        
        # torchvision vgg16
        print('loading perceptual')
        self.load_perceptual()

        # https://github.com/TreB1eN/InsightFace_Pytorch
        print('loading face recognition')
        self.load_face_recognition(face_recognition_ckpt)

        # DAFAN
        print('loading landmark detector')
        self.to_fan = transforms.Compose(
            [
                UnNormalize(),
                # FANNormalize(),
            ]
        )
        self.load_landmark_detector(landmark_detector_ckpt)
        self.landmark_indices = torch.from_numpy(np.load(LANDMARK_INDICES_PATH)).to(self.device).type(torch.LongTensor)

        print('loading regressor')
        self.load_regressor(regressor_ckpt)

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
            ]
        )

        self.to_pil = transforms.Compose(
            [
                UnNormalize(),## Visualize Extracted Facial Landmarks from dlib
                #torch.clamp(0.0,1.0),
                transforms.ToPILImage(),
            ]
        )

        self.perceptual_resize = torch.nn.Upsample(size = (self.perceptual_size, self.perceptual_size), mode='bilinear')
        self.perceptual_feature_resize = torch.nn.AdaptiveAvgPool2d((self.perceptual_size//8, self.perceptual_size//8))
        self.face_recognition_resize = torch.nn.Upsample(size = (self.face_recognition_size, self.face_recognition_size), mode='bilinear')
        self.landmark_resize = torch.nn.Upsample(size = (self.landmark_size, self.landmark_size), mode='bilinear')
        self.regressor_resize = self.landmark_resize


    def __call__(self, i0, image, mask, dense_landmarks):
        dense_landmarks = (dense_landmarks+1) * self.size/2
        dense_landmarks[:, 1] = self.size-dense_landmarks[:, 1]
        landmarks = dense_landmarks[self.landmark_indices]

        image, mask, dense_landmarks = align_image_and_landmarks(image, mask, dense_landmarks, landmarks, self.size)
        landmarks = dense_landmarks[self.landmark_indices]
        i0 = self.transform(i0).unsqueeze(0).to(self.device)
        image = self.transform(image).unsqueeze(0).to(self.device)
        mask = self.mask_transform(mask).unsqueeze(0).to(self.device)
        landmarks = torch.from_numpy(landmarks).to(self.device)
        # use RGB format for mask to match image easily
        # just to be sure

        print('initializing wp')
        # w_ini is mean latent for face
        if self.regressor is not None:
            wp = self.regressor(self.regressor_resize(image))
            wp = wp.clone().detach()
        # a regressor needs to be trained
        else:
            print('no regressor - using mean face as initiailzation')
            n=4096
            samples = 256
            w = []
            for _ in range(n//samples):
                sample_z = torch.randn(samples, 512, device=self.device)
                w.append(self.g_ema.style(sample_z))
            w = torch.cat(w, dim=0)
            mean_w = w.mean(dim=0)
            wp = mean_w.reshape(1, 1, 512).repeat(1, self.num_layers, 1).detach().clone()

        print('optimizing wp')
        wp = self.optimize(image, mask, landmarks, wp)

        result_image, _  = self.g_ema(wp, input_is_wp=True, randomize_noise=False)
        result_image = self.to_pil(result_image.squeeze(0).cpu())


        dense_landmarks[:, 1] = self.size-dense_landmarks[:, 1]
        dense_landmarks = (dense_landmarks/(self.size/2)) - 1

        return result_image, dense_landmarks
        

    def optimize(self, i0, image, mask, landmarks, wp):
        wp_split = [wp[:, :9].clone().detach(), wp[:, 9:self.num_layers-1].clone().detach(), wp[:, self.num_layers-1:].clone().detach()]

        wp_split[0].requires_grad=True
        wp_split[1].requires_grad=False
        wp_split[2].requires_grad=True
        optimizer = torch.optim.Adam([wp_split[0], wp_split[2]], lr=self.lr)


        # optimizer = torch.optim.Adam([wp], lr=self.lr)
        pbar = range(self.iterations)
        pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

        photometric_mask = mask.clone().detach()
        photometric_mask = photometric_mask.repeat(1, 3, 1, 1)

        perceptual_mask = self.perceptual_feature_resize(mask.clone().detach())
        perceptual_mask = perceptual_mask.repeat(1, 512, 1, 1)

        for idx in pbar:
            fake_image, _ = self.g_ema(torch.cat(wp_split, dim=1), input_is_wp=True, randomize_noise=False)
            pm_loss = self.photometric_loss(image, fake_image, photometric_mask)
            id_loss = self.identity_loss(i0, fake_image)
            lm_loss = self.landmark_loss(landmarks, fake_image)
            pc_loss = self.perceptual_loss(image, fake_image, perceptual_mask)

            loss = pm_loss * self.lambda_photometric +\
                    id_loss * self.lambda_identity +\
                    pc_loss * self.lambda_perceptual +\
                    lm_loss * self.lambda_landmark

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                # 'optimizing: loss: {:2f}; photometric: {:2f}; identity: {:2f}; perceptual: {:2f}'.format(loss, pm_loss, id_loss, pc_loss)
                'optimizing: photometric: {:2f}; identity: {:2f}; landmark: {:2f}; perceptual: {:2f}'.format(pm_loss, id_loss, lm_loss, pc_loss)
            )
        
        return torch.cat(wp_split, dim=1)


    def photometric_loss(self, image, fake_image, mask):
        #print(torch.max(image))
        #print(torch.max(fake_image))
        loss = torch.mean(torch.log(torch.cosh(mask * (image - fake_image))))
        return loss.squeeze()


    def identity_loss(self, i0, fake_image):
        resized_i0 = self.face_recognition_resize(i0)
        resized_fake_image = self.face_recognition_resize(fake_image)
        #NO MASK
        #resized_image = self.face_recognition_resize(image)
        #resized_fake_image = self.face_recognition_resize(fake_image)
        FI_0 = self.face_recognition(resized_i0)
        FG_i = self.face_recognition(resized_fake_image)
        loss = 1 - (torch.matmul(FI_0, FG_i.T)) / (torch.norm(FI_0) * torch.norm(FG_i))
        return loss.squeeze()

    
    def perceptual_loss(self, image, fake_image, mask):
        resized_image = self.perceptual_resize(image)
        resized_fake_image = self.perceptual_resize(fake_image) 
        loss = torch.mean(torch.log(torch.cosh(mask * (self.perceptual(resized_image) - self.perceptual(resized_fake_image)))))
        # resized_image = self.perceptual_resize(image)
        # resized_fake_image = self.perceptual_resize(fake_image)
        # loss = torch.mean(torch.log(torch.cosh(mask * (self.perceptual(resized_image) - self.perceptual(resized_fake_image)))))
        return loss.squeeze()


    def landmark_loss(self, landmarks, fake_image):
        resized_fake_image = self.landmark_resize(fake_image)
        detected_landmarks = self.landmark_detector(resized_fake_image).squeeze(0)
        # landmark shapes are 68 x 2
        landmarks = (landmarks/(self.size/2))-1
        detected_landmarks = (detected_landmarks/32)-1
        loss = torch.mean(torch.norm(detected_landmarks - landmarks, dim=1))
        return loss.squeeze()

    def load_regressor(self, ckpt):
        opts = Namespace()        
        opts.input_nc = 3
        opts.device = self.device
        opts.checkpoint_path = ckpt
        opts.encoder_type='GradualStyleEncoder'
        opts.start_from_latent_avg = True
        net = pSp(opts)
        net.to(self.device)
        net.eval()
        self.regressor = net

    def load_perceptual(self):
        # "We empirically choose 9th layer of a VGG-16 network that is pretrained as an ImageNet classifier"
        perceptual = torch.nn.Sequential(
                        *list(torchvision.models.vgg16(pretrained=True).children())[0][:20]
                    ).to(self.device)
        perceptual.eval()
        self.perceptual = perceptual


    def load_face_recognition(self, ckpt):
        self.face_recognition = MobileFaceNet(512)
        self.face_recognition.load_state_dict(torch.load(ckpt))
        self.face_recognition.to(self.device)
        self.face_recognition.eval()


    def load_landmark_detector(self, ckpt):
        fan = FAN(4)
        fan_checkpoint = torch.load(ckpt)
        #fan_weights = fan_checkpoint['state_dict']
        fan_weights = {
                k.replace('module.', ''): v for k,
                v in fan_checkpoint['state_dict'].items()}
        fan.load_state_dict(fan_weights)
        fan.to(self.device)
        fan.eval()
        self.landmark_detector = LandmarkDetector(fan, self.to_fan, self.device)

"""
if __name__ == '__main__':
    print('it\'s a module file.')
    inpainter = Inpainter()
    output_img, _ = inpainter(image=Image.open('img00000048_rendered_view_3.png').convert('RGB'),
                            mask=Image.open('img00000048_rendered_mask_view_3.png').convert('L'),
                            dense_landmarks=np.load('img00000048_S_view_3.npy'))
    output_img.save('inpainted_front.png')
"""
