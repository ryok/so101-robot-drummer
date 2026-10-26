"""SmolVLA の推論時間を測る（ロボット不要。ランダム画像と言語指示で1チャンクを生成する時間）。
使い方:
  PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/smolvla_latency.py --device mps
  python scripts/smolvla_latency.py --device mps --num-steps 5     # 拡散の反復回数を減らした場合
判断基準: 1チャンク（50ステップ＝30fpsで約1.7秒分の動き）の生成が、拍の前倒し時間内に収まるか。
"""
import argparse, os, statistics, time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lerobot/smolvla_base")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--num-steps", type=int, default=None, help="拡散の反復回数（既定はconfigの値）")
    a = ap.parse_args()

    t = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(a.model)
    if a.num_steps:
        policy.config.num_steps = a.num_steps
    policy.to(a.device).eval()
    print(f"読み込み: {time.perf_counter() - t:.1f}s  device={a.device}  num_steps={policy.config.num_steps}  "
          f"chunk_size={policy.config.chunk_size}")
    print("入力:", {k: (v.type.name, tuple(v.shape)) for k, v in policy.config.input_features.items()})

    pre, _ = make_pre_post_processors(policy.config, pretrained_path=a.model,
                                      preprocessor_overrides={"device_processor": {"device": a.device}})
    obs = {}
    for k, f in policy.config.input_features.items():
        obs[k] = torch.rand(f.shape) if f.type.name == "VISUAL" else torch.zeros(f.shape)
    obs["task"] = "strike the cup"
    batch = pre(obs)

    times = []
    with torch.inference_mode():
        for i in range(a.runs + 2):                      # 最初の2回は準備運転として捨てる
            policy.reset()
            sync(a.device)
            t = time.perf_counter()
            chunk = policy.predict_action_chunk(batch)
            sync(a.device)
            if i >= 2:
                times.append(time.perf_counter() - t)
    ms = [x * 1000 for x in times]
    span = policy.config.chunk_size / 30
    print(f"\n1チャンク生成: 平均{statistics.mean(ms):.0f}ms ±{statistics.stdev(ms):.0f}ms "
          f"(最小{min(ms):.0f} / 最大{max(ms):.0f}, n={len(ms)})  出力形状={tuple(chunk.shape)}")
    print(f"1チャンクの動きの長さ: {chunk.shape[1]}ステップ ≈ {span:.2f}s（30fps）")
    print(f"→ 推論が動きの長さより短い＝先読みで間に合う: {'はい' if statistics.mean(ms) / 1000 < span else 'いいえ'}")


if __name__ == "__main__":
    main()
