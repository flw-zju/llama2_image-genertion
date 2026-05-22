import torch
from tqdm import tqdm
from model.rf_llama2 import RectifiedFlow
import argparse
import os
from torchvision.utils import save_image


parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str,
                    default="")
parser.add_argument('--vae_path', type=str,
                    default="")
parser.add_argument('--image_saving_path', type=str,
                    default="")
parser.add_argument('--sample_step', type=int,
                    default=1000)
parser.add_argument('--num_sample', type=int,
                    default=256)
parser.add_argument('--batch_size', type=int,
                    default=8)
args = parser.parse_args()


if __name__ == "__main__":
    if not os.path.exists(args.image_saving_path):
        os.makedirs(args.image_saving_path)
        print(f"folder is created: {args.image_saving_path}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sde = RectifiedFlow(device=device,
                        vae_path=args.vae_path
                        ).to(device)
    sde.dit.load_state_dict(torch.load(args.model_path, weights_only=True))

    iter = (args.num_sample + args.batch_size - 1) // args.batch_size
    dt = 1. / args.sample_step
    zjr = 0

    with torch.no_grad():

        for _ in tqdm(range(iter)):
            z = torch.randn((args.batch_size, 4, 32, 32), device=device)
            for i in range(args.sample_step):
                t = torch.ones(z.shape[0]).to(device)
                t = t * i / args.sample_step
                t = t * (1 - 1e-3) + 1e-3
                pred = sde.dit(z, t * 999)
                z = z + pred * dt

            z = z / 0.2039
            z = sde.vae.decode(z)
            z = (z + 1) * 0.5
            z = torch.clamp(z, 0, 1)
            for j in range(z.shape[0]):
                rec = z[j,...].unsqueeze(0)
                image_name = str(zjr) + ".png"
                save_image(rec, os.path.join(args.image_saving_path, image_name), normalize=False)
                zjr += 1
    print("The number of generated images is: ", zjr)
