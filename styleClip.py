import os
import threading
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from threading import *


class styleClip(object):

    def __init__(self, device, controller, clip_model, stylegan_model, seed=-1, output_path="output"):
        self.device = device
        self.output_path = output_path
        self.clip_model = clip_model
        self.controller = controller
        self.stylegan_model = stylegan_model
        self.seed = seed if seed != -1 else np.random.randint(0, 999999999)

    def set_output_dir(self, output_path):
        self.output_path = output_path

    def spherical_dist_loss(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    def prompts_dist_loss(self, x, targets, loss):
        if len(targets) == 1:  # Keeps consitent results vs previous method for single objective guidance
            return loss(x, targets[0])
        distances = [loss(x, target) for target in targets]
        return torch.stack(distances, dim=-1).sum(dim=-1)

    def get_targets(self, texts):
        texts = [frase.strip() for frase in texts.split("|") if frase]
        targets = [self.clip_model.embed_text(text) for text in texts]
        return targets

    def work_run(self, texts="a brown jacket", steps=10, seed=14, render_video=False, save_every=2):
        t1 = threading.Thread(target=self.run, args=(
            texts, steps, seed, render_video, save_every))
        t1.start()

    def run(self, texts, steps, seed, render_video, save_every):

        torch.manual_seed(seed)
        print(
            f"Running styleclip for seed: {seed}, texts: {texts}, steps: {steps}")

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
            loss = self.prompts_dist_loss(
                embed, targets, self.spherical_dist_loss).mean(0)
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
                file_path = self.save_image(image, seed, i, self.output_path)
                self.controller.update_image(file_path)
        if render_video:
            self.save_video(images, self.output_path)

    def init_optimization(self, targets):
        with torch.no_grad():
            qs = []
            losses = []
            for _ in range(8):
                q = (self.stylegan_model.G.mapping(torch.randn([4, self.stylegan_model.G.mapping.z_dim], device=self.device),
                     None, truncation_psi=0.7) - self.stylegan_model.G.mapping.w_avg) / self.stylegan_model.w_stds
                images = self.stylegan_model.G.synthesis(
                    q * self.stylegan_model.w_stds + self.stylegan_model.G.mapping.w_avg)
                embeds = self.clip_model.embed_image(images.add(1).div(2))
                loss = self.prompts_dist_loss(
                    embeds, targets, self.spherical_dist_loss).mean(0)
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
        return file_path

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
