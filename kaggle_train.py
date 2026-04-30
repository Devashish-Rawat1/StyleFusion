import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image

from utils import *
from models import *

# =========================
# CONFIG (EDIT THIS ONLY)
# =========================
DATASET_PATH = "/kaggle/input/your-dataset-name"

CONTENT_DIR = f"{DATASET_PATH}/content_data"
STYLE_DIR   = f"{DATASET_PATH}/style_data"
VGG_PATH    = f"{DATASET_PATH}/vgg_normalised.pth"

SAVE_DIR = Path("/kaggle/working/experiment1")

# Phase 1
PHASE1_EPOCHS = 160
PHASE1_STYLE_WEIGHT = 5
PHASE1_SIZE = 256

# Phase 2
PHASE2_EPOCHS = 100
PHASE2_STYLE_WEIGHT = 10
PHASE2_SIZE = 512

BATCH_SIZE = 4   # use 2 if OOM
LR = 1e-4
LR_DECAY = 5e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_phase(epochs, style_weight, final_size, resume=False,
                decoder_path=None, optimizer_path=None, start_epoch=0):

    print(f"\nStarting Phase → epochs={epochs}, style_weight={style_weight}, size={final_size}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    content_transform = get_transform(512, True, final_size)
    style_transform   = get_transform(512, True, final_size)

    content_dataset = ImageFolderDataset(CONTENT_DIR, content_transform)
    style_dataset   = ImageFolderDataset(STYLE_DIR, style_transform)

    content_loader = DataLoader(content_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, drop_last=True, num_workers=2)

    style_loader = DataLoader(style_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=2)

    encoder = VGGEncoder(VGG_PATH).to(DEVICE)
    decoder = Decoder().to(DEVICE)

    optimizer = optim.Adam(decoder.parameters(), lr=LR)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + LR_DECAY * epoch)
    )

    if resume:
        print("Resuming from checkpoint...")
        decoder.load_state_dict(torch.load(decoder_path))
        optimizer.load_state_dict(torch.load(optimizer_path))

    mse_loss = torch.nn.MSELoss()
    encoder.eval()

    for epoch in range(start_epoch, epochs):

        progress = tqdm(zip(content_loader, style_loader),
                        total=min(len(content_loader), len(style_loader)))

        for content_batch, style_batch in progress:

            content_batch = content_batch.to(DEVICE)
            style_batch   = style_batch.to(DEVICE)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])
            g = decoder(t)

            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t)

            loss_s = 0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)

            loss_s *= style_weight
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_description(f"Loss: {loss.item():.4f}")

        scheduler.step()

        # Save checkpoint
        torch.save(decoder.state_dict(), SAVE_DIR / f"decoder_{epoch+1}.pth")
        torch.save(optimizer.state_dict(), SAVE_DIR / f"optimizer_{epoch+1}.pth")

        with torch.no_grad():
            output = torch.cat([content_batch, style_batch, g], dim=0)
            save_image(output, SAVE_DIR / f"output_{epoch+1}.png", nrow=BATCH_SIZE)


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    # ===== Phase 1 =====
    train_phase(
        epochs=PHASE1_EPOCHS,
        style_weight=PHASE1_STYLE_WEIGHT,
        final_size=PHASE1_SIZE
    )

    # ===== Phase 2 (resume) =====
    train_phase(
        epochs=PHASE2_EPOCHS,
        style_weight=PHASE2_STYLE_WEIGHT,
        final_size=PHASE2_SIZE,
        resume=True,
        decoder_path=SAVE_DIR / f"decoder_{PHASE1_EPOCHS}.pth",
        optimizer_path=SAVE_DIR / f"optimizer_{PHASE1_EPOCHS}.pth"
    )