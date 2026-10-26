"""学習済みポリシーで「構え → 指示した楽器を1回叩く → 構えに戻る」を繰り返し、成績を記録する。
ACT（楽器ごとに別モデル）と SmolVLA（1モデル＋指示文）を同じ手順で比べるための評価スクリプト。

使い方（自分のターミナルで。follower が動く）:
  python scripts/strike_policy.py smolvla --ckpt outputs/train/smolvla_mix_gpu/last --trials 10
  python scripts/strike_policy.py act --ckpt-drum outputs/train/act_mix_drum_gpu/last \
      --ckpt-bowl outputs/train/act_mix_bowl_gpu/last --trials 10
  --offline: ロボット無しでデータセットの構えフレームを入力し、推論〜判定〜記録の流れだけ検証する

流れ: leader で構えを教えて Enter → 各試行で「構えに戻す → 1秒待つ → 推論 → 実行(--duration 秒)」。
      指示は drum / bowl を同数ずつランダム順。試行のあと、実際に鳴った楽器を d / b / n で入力する
      （Enter だけなら自動判定 = 肘が一番下がった瞬間の pan の向き を採用）。
記録: outputs/eval/<タグ>.csv（試行ごとの指示、推論時間、肘の最下点までの時間、自動判定、人の判定）と
      <タグ>_traj.npz（試行ごとの構え・命令値・実際の角度の時系列）。
"""
import argparse, csv, os, random, time
from datetime import datetime

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors

from strike_spike import LEADER_PORT, PORT, follow_until_enter

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TASK = {"drum": "Strike the drum", "bowl": "Strike the bowl"}
FPS = 30


def load(kind, ckpt, device, num_steps=None):
    if kind == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as P
    else:
        from lerobot.policies.act.modeling_act import ACTPolicy as P
    policy = P.from_pretrained(ckpt)
    if num_steps:
        policy.config.num_steps = num_steps
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def to_policy_obs(raw, task):
    """robot.get_observation() の生データ → データセットと同じ形（state 6次元、画像 CHW float 0〜1）。"""
    obs = {"observation.state": torch.tensor([raw[f"{j}.pos"] for j in JOINTS], dtype=torch.float32)}
    for cam in ("front", "wrist"):
        obs[f"observation.images.{cam}"] = torch.from_numpy(np.ascontiguousarray(raw[cam])).permute(2, 0, 1).float() / 255
    obs["task"] = task
    return obs


class OfflineArm:
    """ロボット無し検証用: データセットの構えフレームを観測として返し、命令をそのまま現在角にする。"""
    def __init__(self, dataset):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.ds = LeRobotDataset(dataset)
        self.starts = self.ds.meta.episodes["dataset_from_index"]
        self.it, self.pos = None, None

    def reset_to(self, ep):
        self.it = self.ds[int(self.starts[ep % len(self.starts)])]
        self.pos = {f"{j}.pos": float(v) for j, v in zip(JOINTS, self.it["observation.state"])}

    def get_observation(self):
        raw = dict(self.pos)
        for cam in ("front", "wrist"):
            raw[cam] = (self.it[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return raw

    def send_action(self, act):
        self.pos.update(act)


def run_trial(arm, policy, pre, post, task, duration, device, hover=None, elbow_gain=1.0):
    """1試行: 推論 → チャンクを30fpsで実行（尽きたら観測を取り直して再推論）。軌跡と推論時間を返す。
    elbow_gain: 肘の「構えより下げる量」だけを倍率で強める（1.0=ポリシーの出力そのまま）。"""
    policy.reset()
    traj, infer_ms = [], []
    t_start = time.perf_counter()
    queue = []
    while time.perf_counter() - t_start < duration:
        if not queue:
            raw = arm.get_observation()
            t0 = time.perf_counter()
            with torch.inference_mode():
                chunk = post(policy.predict_action_chunk(pre(to_policy_obs(raw, task))))[0].cpu().numpy()
            if device == "mps":
                torch.mps.synchronize()
            infer_ms.append((time.perf_counter() - t0) * 1000)
            queue = list(chunk[: policy.config.n_action_steps])
        tick = time.perf_counter()
        a = queue.pop(0).copy()
        if elbow_gain != 1.0 and hover is not None and a[2] > hover[2]:
            a[2] = hover[2] + (a[2] - hover[2]) * elbow_gain
        arm.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, a)})
        pos = arm.get_observation() if isinstance(arm, OfflineArm) else None
        if pos is None:
            pos = {f"{k}.pos": v for k, v in arm.bus.sync_read("Present_Position").items()}
        traj.append((tick - t_start, [pos[f"{j}.pos"] for j in JOINTS], [float(v) for v in a]))
        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - tick)))
    return traj, infer_ms


def judge(traj, hover):
    """肘が最も下がった瞬間（+方向）の pan の向きで、どちらへ振ったかを自動判定する。"""
    S = np.array([p for _, p, _ in traj])
    i = int(np.argmax(S[:, 2]))
    pan, swing = S[i, 0] - hover[0], S[i, 2] - hover[2]
    side = "n" if swing < 8 else ("d" if pan < 0 else "b")
    return side, pan, swing, traj[i][0] * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["smolvla", "act"])
    ap.add_argument("--ckpt", help="SmolVLA のチェックポイント")
    ap.add_argument("--ckpt-drum", help="ACT ドラム用")
    ap.add_argument("--ckpt-bowl", help="ACT ボウル用")
    ap.add_argument("--trials", type=int, default=10, help="試行数（drum/bowl 半々）")
    ap.add_argument("--duration", type=float, default=3.0, help="1試行で動かす秒数")
    ap.add_argument("--elbow-gain", type=float, default=1.0,
                    help="肘の振り下ろし量の倍率（構えより下がる分だけ。1.0=そのまま）")
    ap.add_argument("--num-steps", type=int, default=None, help="SmolVLA の拡散反復回数（既定10）")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--front", type=int, default=1, help="俯瞰カメラの番号")
    ap.add_argument("--wrist", type=int, default=0, help="手元カメラの番号")
    ap.add_argument("--tag", default=None, help="記録ファイル名（既定: 種類_日時）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offline", action="store_true", help="ロボット無しでデータセットの構えを入力にして流れだけ検証")
    ap.add_argument("--dataset", default="local/so101-strike-mix", help="--offline で使う構えのデータセット")
    a = ap.parse_args()

    if a.kind == "smolvla":
        model = load("smolvla", a.ckpt, a.device, a.num_steps)
        models = {"drum": model, "bowl": model}
    else:
        models = {"drum": load("act", a.ckpt_drum, a.device), "bowl": load("act", a.ckpt_bowl, a.device)}

    order = ["drum", "bowl"] * (a.trials // 2) + ["drum"] * (a.trials % 2)
    random.Random(a.seed).shuffle(order)
    tag = a.tag or f"{a.kind}{'_offline' if a.offline else ''}_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs("outputs/eval", exist_ok=True)
    path = f"outputs/eval/{tag}.csv"

    robot = leader = None
    if a.offline:
        arm = OfflineArm(a.dataset)
    else:
        from lerobot.cameras.configs import Cv2Backends
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        cam = lambda i: OpenCVCameraConfig(index_or_path=i, fps=30, width=640, height=480,
                                           backend=Cv2Backends.AVFOUNDATION)
        robot = SO101Follower(SO101FollowerConfig(port=PORT, id="follower_01", num_read_retries=5,
                                                  max_relative_target=25.0,   # 1ステップの急な飛びを抑える安全弁
                                                  cameras={"front": cam(a.front), "wrist": cam(a.wrist)}))
        robot.connect()
        leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader_01"))
        leader.connect()
        arm = robot

    rows, trajs = [], []
    try:
        if robot:
            follow_until_enter(robot, leader, "leaderで『録画のときと同じ構え（両楽器の中間・上）』に合わせて Enter > ")
            hover = robot.bus.sync_read("Present_Position")
            hover_act = {f"{k}.pos": v for k, v in hover.items()}
        else:
            arm.reset_to(0)
        # 準備運転: 初回の推論はMPSの初期化で約1秒余計にかかるので、試行の前に1回ずつ空打ちしておく
        for inst, (policy, pre, post) in models.items():
            with torch.inference_mode():
                policy.reset()
                policy.predict_action_chunk(pre(to_policy_obs(arm.get_observation(), TASK[inst])))
        for n, inst in enumerate(order, 1):
            if robot:
                robot.send_action(hover_act)
                time.sleep(1.0)
            else:
                arm.reset_to(n)
                hover_act = dict(arm.pos)
            h = [hover_act[f"{j}.pos"] for j in JOINTS]
            policy, pre, post = models[inst]
            traj, infer_ms = run_trial(arm, policy, pre, post, TASK[inst], a.duration, a.device,
                                       hover=h, elbow_gain=a.elbow_gain)
            side, pan, swing, t_peak = judge(traj, h)
            auto_ok = side == inst[0]
            print(f"[{n}/{len(order)}] 指示={inst:4s} 自動判定={ {'d': 'drum', 'b': 'bowl', 'n': '振らず'}[side] }"
                  f"（pan{pan:+.0f}° 肘{swing:+.0f}°、最下点 {t_peak:.0f}ms）"
                  f" 推論 {'/'.join(f'{x:.0f}' for x in infer_ms)}ms  {'○' if auto_ok else '×'}")
            human = ""
            if robot:
                human = input("   実際に鳴った楽器 d=ドラム b=ボウル n=鳴らず（Enter=自動判定を採用）> ").strip().lower()[:1]
            result = human or side
            trajs.append(dict(instruction=inst, hover=np.array(h), t=np.array([x[0] for x in traj]),
                              actual=np.array([x[1] for x in traj]), command=np.array([x[2] for x in traj])))
            cmd = trajs[-1]["command"]
            print(f"   命令の最大: pan{cmd[:, 0].max() - h[0]:+.0f}/{cmd[:, 0].min() - h[0]:+.0f}°  "
                  f"肘{cmd[:, 2].max() - h[2]:+.0f}°（実際の肘 {swing:+.0f}°）")
            rows.append(dict(trial=n, policy=a.kind, instruction=inst, infer_ms_first=round(infer_ms[0]),
                             n_infer=len(infer_ms), peak_ms=round(t_peak), pan_deg=round(pan, 1),
                             elbow_deg=round(swing, 1), auto=side, human=human, result=result,
                             correct=int(result == inst[0])))
        if robot:
            robot.send_action(hover_act)
            follow_until_enter(robot, leader, "終了。leaderで待機姿勢(畳んだ形)に戻して Enter > ")
    finally:
        if rows:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        if trajs:
            np.savez(path.replace(".csv", "_traj.npz"), **{f"t{i + 1}_{k}": v for i, d in enumerate(trajs)
                                                         for k, v in d.items() if k != "instruction"},
                     instructions=np.array([d["instruction"] for d in trajs]))
        for dev in (leader, robot):
            if dev:
                try:
                    dev.disconnect()
                except Exception as ex:   # 電圧低下などで脱力に失敗しても、記録と集計は残す
                    print(f"切断時のエラー（記録は保存済み）: {ex}")

    if rows:
        for inst in ("drum", "bowl"):
            r = [x for x in rows if x["instruction"] == inst]
            if r:
                print(f"{inst}: 指示どおり {sum(x['correct'] for x in r)}/{len(r)}  "
                      f"最下点まで 平均{np.mean([x['peak_ms'] for x in r]):.0f}ms")
        print(f"全体: {sum(x['correct'] for x in rows)}/{len(rows)}  初回推論 平均{np.mean([x['infer_ms_first'] for x in rows]):.0f}ms"
              f"  → {path}")


if __name__ == "__main__":
    main()
