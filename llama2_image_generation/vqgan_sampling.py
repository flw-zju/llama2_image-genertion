import torch
from tqdm import tqdm
from model.vq_vae import VQModel
from model.vqgan_llama2 import Llama2
import argparse
import os
from torchvision.utils import save_image


parser = argparse.ArgumentParser()
parser.add_argument('--vqgan_path', type=str,
                    default="")
parser.add_argument('--llama_path', type=str,
                    default="")
parser.add_argument('--image_saving_path', type=str,
                    default="")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--top_k', type=int, default=5)
args = parser.parse_args()


if __name__ == "__main__":
    if not os.path.exists(args.image_saving_path):
        os.makedirs(args.image_saving_path)
        print(f"folder is created: {args.image_saving_path}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vae_model = VQModel().to(device)
    vae_model.load_state_dict(torch.load(args.vqgan_path, weights_only=True))
    llama_model = Llama2(mode="eval").to(device)
    llama_model.load_state_dict(torch.load(args.llama_path, weights_only=True))

    idx = llama_model.sample(args.batch_size, device, args.top_k)
    z_q = (vae_model.quantize.embedding(idx)
           .transpose(1, 2).reshape(
            idx.shape[0], -1, 16, 16))
    gens = vae_model.decode(z_q)
    gens = (gens + 1) / 2
    gens = torch.clamp(gens, 0, 1)

    zjr = 0
    for j in range(args.batch_size):
        rec = gens[j, ...].unsqueeze(0)
        image_name = str(zjr) + ".png"
        save_image(rec, os.path.join(args.image_saving_path, image_name), normalize=False)
        zjr += 1
    print("The number of generated images is: ", zjr)
