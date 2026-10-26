"""ロボット無しで、学習済みポリシーが「指示した楽器の方へ振るか」を確かめる。
データセットの各エピソードの先頭フレーム（構え）を入力し、予測した行動チャンクの中で
shoulder_pan が構えから最も大きく動いた量（−=左=ドラム、+=右=ボウル）と、肘の最大の下がり幅を表示する。
使い方:
  python scripts/offline_lang_check.py smolvla outputs/train/smolvla_strike_gpu/last
  python scripts/offline_lang_check.py act outputs/train/act_strike_drum_gpu/last
  --dataset local/so101-strike-mix --episodes 0,1,3,5,2,4,6,8   # 交互録画データの構えで確認
"""
import argparse, os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

TASKS = ["Strike the drum", "Strike the bowl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["smolvla", "act"])
    ap.add_argument("ckpt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dataset", default="local/so101-strike")
    ap.add_argument("--episodes", default="0,3,6,9,11,14,17,20")
    a = ap.parse_args()

    if a.kind == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as P
    else:
        from lerobot.policies.act.modeling_act import ACTPolicy as P
    policy = P.from_pretrained(a.ckpt).to(a.device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=a.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": a.device}})
    ds = LeRobotDataset(a.dataset)
    starts = ds.meta.episodes["dataset_from_index"]
    tasks = TASKS if a.kind == "smolvla" else [None]
    print(f"{'ep':>3} {'録画の楽器':16s} " + "  ".join(f"[{t or 'ACT'}] pan/肘" for t in tasks))
    with torch.inference_mode():
        for e in map(int, a.episodes.split(",")):
            it = ds[int(starts[e])]
            s0 = it["observation.state"]
            cells = []
            for t in tasks:
                obs = {k: v for k, v in it.items() if k.startswith("observation.")}
                obs["task"] = t or it["task"]
                policy.reset()
                chunk = post(policy.predict_action_chunk(pre(obs)))[0].cpu()     # (T, 6) 角度[deg]
                d = chunk - s0
                pan = d[:, 0][d[:, 0].abs().argmax()].item()
                cells.append(f"{pan:+6.1f}° /{d[:, 2].max().item():+5.1f}°")
            print(f"{e:3d} {it['task']:16s} " + "   ".join(cells))


if __name__ == "__main__":
    main()
