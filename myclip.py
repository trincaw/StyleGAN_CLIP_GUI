import clip
import sys
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
from einops import rearrange

sys.path.append('./CLIP/clip')


def norm1(prompt):
    "Normalize to the unit sphere."
    return prompt / prompt.square().sum(dim=-1, keepdim=True).sqrt()


class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow *
                       (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety +
                           size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


class myclip(object):
    def __init__(self, device, make_cutouts=MakeCutouts(224, 32, 0.5)):
        self.device = device
        clip_model = "ViT-B/32"
        self.make_cutouts = make_cutouts
        self.model, _ = clip.load(clip_model)
        self.model = self.model.requires_grad_(False)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def embed_text(self, prompt):
        "Normalized clip text embedding."
        return norm1(self.model.encode_text(clip.tokenize(prompt).to(self.device)).float())

    def embed_cutout(self, image):
        "Normalized clip image embedding."
        return norm1(self.model.encode_image(self.normalize(image)))

    def embed_image(self, image):
        n = image.shape[0]
        cutouts = self.make_cutouts(image)
        embeds = self.embed_cutout(cutouts)
        embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
        return embeds
