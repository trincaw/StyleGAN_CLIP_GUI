import os
from einops import rearrange
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import pickle
import clip
import sys
sys.path.append('./CLIP')
# sys.path.append('./stylegan3')


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


class CLIP(object):
    def __init__(self, make_cutouts=MakeCutouts(224, 32, 0.5)):
        clip_model = "ViT-B/32"
        self.make_cutouts = make_cutouts
        self.model, _ = clip.load(clip_model)
        self.model = self.model.requires_grad_(False)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def embed_text(self, prompt):
        "Normalized clip text embedding."
        return norm1(self.model.encode_text(clip.tokenize(prompt).to(device)).float())

    def embed_cutout(self, image):
        "Normalized clip image embedding."
        return norm1(self.model.encode_image(self.normalize(image)))

    def embed_image(self, image):
        n = image.shape[0]
        cutouts = self.make_cutouts(image)
        embeds = clip_model.embed_cutout(cutouts)
        embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
        return embeds


class StyleGAN3(object):
    def __init__(self, network_path):
        self.device = torch.device('cuda:0')
        self.G, self.w_stds = self.load_network(network_path)

    def load_network(self, network_path):
        with open(network_path, 'rb') as fp:
            G = pickle.load(fp)['G_ema'].to(device)

            zs = torch.randn([10000, G.mapping.z_dim], device=device)
            w_stds = G.mapping(zs, None).std(0)
            return G, w_stds


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def prompts_dist_loss(x, targets, loss):
    if len(targets) == 1:  # Keeps consitent results vs previous method for single objective guidance
        return loss(x, targets[0])
    distances = [loss(x, target) for target in targets]
    return torch.stack(distances, dim=-1).sum(dim=-1)


class ImageGenerator(object):
    def __init__(self, clip_model, stylegan_model, seed=-1, output_path="output"):
        self.output_path = output_path
        self.clip_model = clip_model
        self.stylegan_model = stylegan_model
        self.seed = seed if seed != -1 else np.random.randint(0, 999999999)

    def get_targets(self, texts):
        texts = [frase.strip() for frase in texts.split("|") if frase]
        targets = [self.clip_model.embed_text(text) for text in texts]
        return targets

    def run(self, texts, steps, seed, render_video=True, save_every=2):
        torch.manual_seed(seed)
        print(f"Seed: {seed}")

        targets = self.get_targets(texts)

        q = self.init_optimization(targets)
        q_ema = q
        opt = torch.optim.AdamW([q], lr=0.03, betas=(0.0, 0.999))

        self.output_path = os.path.join(self.output_path, f'{seed}')
        images = [0 for _ in range(steps)]
        for i in range(steps):
            opt.zero_grad()
            w = q * self.stylegan_model.w_stds
            image = self.stylegan_model.G.synthesis(
                w + self.stylegan_model.G.mapping.w_avg, noise_mode='const')
            embed = self.clip_model.embed_image(image.add(1).div(2))
            loss = prompts_dist_loss(
                embed, targets, spherical_dist_loss).mean(0)
            loss.backward()
            opt.step()

            q_ema = self.update_q_ema(q_ema, q)
            image = self.stylegan_model.G.synthesis(
                q_ema * self.stylegan_model.w_stds + self.stylegan_model.G.mapping.w_avg, noise_mode='const')
            if render_video:
                images[i] = image
            if i % 10 == 0:
                print(f"Image {i}/{steps} | Current loss: {loss}")
            if i % save_every == 0:
                self.save_image(image, seed, i, self.output_path)
        if render_video:
            self.save_video(images, self.output_path)

    def init_optimization(self, targets):
        with torch.no_grad():
            qs = []
            losses = []
            for _ in range(8):
                q = (self.stylegan_model.G.mapping(torch.randn([4, self.stylegan_model.G.mapping.z_dim], device=device),
                     None, truncation_psi=0.7) - self.stylegan_model.G.mapping.w_avg) / self.stylegan_model.w_stds
                images = self.stylegan_model.G.synthesis(
                    q * self.stylegan_model.w_stds + self.stylegan_model.G.mapping.w_avg)
                embeds = self.clip_model.embed_image(images.add(1).div(2))
                loss = prompts_dist_loss(
                    embeds, targets, spherical_dist_loss).mean(0)
                i = torch.argmin(loss)
                qs.append(q[i])
                losses.append(loss[i])
            qs = torch.stack(qs)
            losses = torch.stack(losses)
            i = torch.argmin(losses)
            q = qs[i].unsqueeze(0).requires_grad_()
        return q

    def update_q_ema(self, q_ema, q):
        q_ema = q_ema * 0.9 + q * 0.1
        return q_ema

    def save_image(self, image, seed, i, output_folder):
        pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0, 1))
        file_path = os.path.join(output_folder, f'{i:04}.jpg')
        os.makedirs(output_folder, exist_ok=True)
        pil_image.save(file_path)
        return pil_image

    def save_video(self, images, output_path, fps=3):
        # Convert each image in the array to PIL Image format
        pil_images = [TF.to_pil_image(img[0].add(
            1).div(2).clamp(0, 1)) for img in images]

        # Convert PIL Images to NumPy arrays
        frame_array = [np.array(img) for img in pil_images]

        # Determine the video's frame width and height
        frame_height, frame_width, _ = frame_array[0].shape

        # Create the video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(os.path.join(
            output_path, "output_video.mp4"), fourcc, fps, (frame_width, frame_height))

        # Write each frame to the video
        for frame in frame_array:
            writer.write(frame)

        # Release the writer to finalize the video
        writer.release()
        print("Video saved successfully!")


device = torch.device('cuda:0')
print('Using device:', device, file=sys.stderr)

clip_model = CLIP()
network_path = os.path.join(os.getcwd(), "network-snapshot-012000.pkl")
stylegan_model = StyleGAN3(network_path)
image_generator = ImageGenerator(clip_model, stylegan_model)
image_generator.run(texts="a brown jacket", steps=50,
                    seed=14, render_video=False, save_every=2)
