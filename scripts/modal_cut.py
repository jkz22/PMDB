"""Train CUT/FastCUT (Park et al., ECCV 2020; github.com/taesungp/contrastive-unpaired-translation) on Modal:
domain A = strong-session PMDB BSE tiles (hybrid-corrected), domain B = all other labelled sites; then translate
the full strong-site BSE images A->B. Experimental route; see docs/harmonisation_shift.md."""
import modal, os, sys

app = modal.App("pmdb-cut")
vol = modal.Volume.from_name("pmdb-cut", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.10")
         .apt_install("git", "libgl1", "libglib2.0-0")
         .pip_install("torch==2.2.2", "torchvision==0.17.2", "numpy<2", "pillow", "dominate", "visdom", "packaging", "GPUtil", "scipy", "tifffile")
         .run_commands("git clone --depth 1 https://github.com/taesungp/contrastive-unpaired-translation /cut"))

@app.function(gpu="T4", timeout=4 * 3600, image=image, volumes={"/data": vol})
def train(n_epochs: int = 60, n_decay: int = 20, mode: str = "FastCUT"):
    import numpy as np, subprocess
    from PIL import Image
    z = np.load("/data/patches.npz")
    for dom in ("A", "B"):
        d = f"/cut/datasets/pmdb/train{dom}"; os.makedirs(d, exist_ok=True)
        for i, t in enumerate(z[dom]):
            Image.fromarray(np.repeat(t[..., None], 3, axis=2)).save(f"{d}/{i:05d}.png")  # CUT's loader is RGB
    for dom in ("A", "B"):
        os.makedirs(f"/cut/datasets/pmdb/test{dom}", exist_ok=True)
    cmd = ["python", "train.py", "--dataroot", "./datasets/pmdb", "--name", "pmdb_cut", "--CUT_mode", mode,
           "--load_size", "256", "--crop_size", "256", "--preprocess", "none",
           "--n_epochs", str(n_epochs), "--n_epochs_decay", str(n_decay), "--display_id", "0", "--no_html",
           "--batch_size", "4", "--num_threads", "4", "--save_epoch_freq", "2", "--checkpoints_dir", "/data/checkpoints"]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd="/cut", capture_output=False)
    vol.commit()
    return r.returncode

@app.function(gpu="T4", timeout=3600, image=image, volumes={"/data": vol})
def translate(epoch: str = "latest"):
    import numpy as np, torch, sys
    sys.path.insert(0, "/cut"); os.chdir("/cut")
    from models import networks
    G = networks.define_G(3, 3, 64, "resnet_9blocks", "instance", False, "xavier", 0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[0], opt=None)
    sd = torch.load(f"/data/checkpoints/pmdb_cut/{epoch}_net_G.pth", map_location="cuda")
    G.load_state_dict(sd); G.eval()
    z = np.load("/data/strong_full.npz"); out = {}
    with torch.no_grad():
        for k in z.files:
            img = z[k].astype(np.float32) / 127.5 - 1
            H, W = img.shape; h, w = H - H % 4, W - W % 4
            res = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
            T = 512
            for y in list(range(0, h - T + 1, T - 64)) + [h - T]:
                for x in list(range(0, w - T + 1, T - 64)) + [w - T]:
                    t = torch.from_numpy(img[y:y + T, x:x + T])[None, None].repeat(1, 3, 1, 1).cuda()
                    o = G(t)[0].mean(0).cpu().numpy()
                    res[y:y + T, x:x + T] += o; cnt[y:y + T, x:x + T] += 1
            res = np.where(cnt > 0, res / np.maximum(cnt, 1), img)
            out[k] = np.clip((res + 1) * 127.5, 0, 255).astype(np.uint8)
            print(k, "done", flush=True)
    np.savez_compressed(f"/data/translated_{epoch}.npz", **out)
    vol.commit()
    return list(out)

@app.local_entrypoint()
def main(stage: str = "train", epochs: int = 60, decay: int = 20):
    if stage == "upload":
        with vol.batch_upload(force=True) as b:
            b.put_file("patches.npz", "/patches.npz"); b.put_file("strong_full.npz", "/strong_full.npz")
        print("uploaded")
    elif stage == "train":
        rc = train.remote(n_epochs=epochs, n_decay=decay)
        print("train rc", rc)
        if rc == 0:
            print(translate.remote("latest"))
    else:
        print(translate.remote(stage))
