import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.transforms.functional import five_crop, resize, normalize
import open_clip

from torchvision.transforms import Normalize, Resize, InterpolationMode, CenterCrop, Compose


class FrozenOpenCLIPImg(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for image
    """
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda",
                 freeze=True):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device('cpu'),
            pretrained=version
        )
        del model.token_embedding
        del model.transformer
        del model.positional_embedding
        del model.text_projection
        del model.logit_scale
        del model.ln_final
        self.model = model
        self.preprocess = Compose(
            [
                Resize(size=224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size=(224, 224)),
                Normalize(mean=open_clip.OPENAI_DATASET_MEAN, std=open_clip.OPENAI_DATASET_STD)
            ]
        )

        self.device = device
        if freeze:
            self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, img):
        image = self.preprocess(img)
        image_features = self.model.encode_image(image)
        image_features = torch.div(image_features, image_features.norm(dim=-1, keepdim=True))
        return image_features

    def encode(self, img):
        return self(img)

    def encode_five(self, img):
        img_patch = five_crop(img, size=[384, ])
        img_patch = torch.stack(
            [
                resize(img, [384, ]),
                img_patch[0],
                img_patch[1],
                img_patch[2],
                img_patch[3],
            ],
            dim=0
        ).permute(1, 0, 2, 3, 4)

        save_image(
            tensor=[
                torch.div(img_patch[0, 0], 255.0),
                torch.div(img_patch[0, 1], 255.0),
                torch.div(img_patch[0, 2], 255.0),
                torch.div(img_patch[0, 3], 255.0),
                torch.div(img_patch[0, 4], 255.0),
            ],
            fp="tttt.png"
        )

        # print(img_patch.shape)

        img_patch = img_patch.reshape(-1, 3, 384, 384)

        img_patch = resize(img_patch, size=[224, 224])

        img_patch = normalize(
            tensor=img_patch,
            mean=open_clip.OPENAI_DATASET_MEAN,
            std=open_clip.OPENAI_DATASET_STD
        )
        return self(img_patch).view(-1, 5, 1024)


if __name__ == '__main__':
    image0 = torch.tensor(np.array(Image.open("test_imgs/bear.jpg").convert("RGB"))).float()
    image1 = torch.tensor(np.array(Image.open("test_imgs/hamster.jpg").convert("RGB"))).float()
    print('shape of image0:', image0.shape)
    print('shape of image1:', image1.shape)

    image_batch = (torch.stack([image0, image1], dim=0)).permute(0, 3, 1, 2)
    print('shape of image_batch:', image_batch.shape)

    test_clip_img = FrozenOpenCLIPImg()

    img_batch_enc = test_clip_img.encode(image_batch)
    print('CLIP encoded image_batch:', img_batch_enc)
    print('shape:', img_batch_enc.shape)

    img_batch_enc_patch = test_clip_img.encode_five(image_batch)
    print('CLIP encoded image_batch, patch version:', img_batch_enc_patch)
    print('shape:', img_batch_enc_patch.shape)
