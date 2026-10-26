"""wrist に写る机の外（扇風機・箱・鞄など）を、学習データのときの配置に戻すための確認画面。
腕を演奏と同じ構え（学習データの構えの中央値）に置き、wrist のライブ映像を基準画像と並べて表示しながら、
ACT（bowl）の照準を0.5秒ごとに推論して画面に出す。物を動かして照準が +18° 以上で安定すれば、演奏でもボウルに届く。
使い方: python scripts/wrist_align.py            （leaderで構えに近い形にして Enter → 腕が構えへ移動 → 画面）
        python scripts/wrist_align.py --no-robot （腕を動かさず画面と推論だけ確認）
画面のキー: q=終了 / m=右画面の表示切替（ライブ↔重ね合わせ↔差分）
10/25: 照明・楽器を合わせても bowl 照準が −8° のままで、入力の入れ替えで原因が wrist の背景と判明したため作成。
"""
import argparse, time

import cv2
import numpy as np
import torch

from align_camera import measure
from strike_conductor import E
from strike_policy import JOINTS, TASK, load
from strike_spike import LEADER_PORT, PORT, follow_until_enter

AIM_OK = 18.0   # これ以上ならボウルの中央付近（10/23 の探索で中央 ≈ +18〜22°）


def open_cam(i):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(15):
        cap.read()
    return cap


def bowl_aim(model, front_bgr, wrist_bgr, state):
    policy, pre, post = model
    img = lambda b: torch.from_numpy(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
    policy.reset()
    with torch.inference_mode():
        chunk = post(policy.predict_action_chunk(pre({
            "observation.state": torch.tensor(state, dtype=torch.float32),
            "observation.images.front": img(front_bgr), "observation.images.wrist": img(wrist_bgr),
            "task": TASK["bowl"]})))[0].cpu().numpy()
    return float(chunk[int(np.argmax(chunk[:, E])), 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=int, default=1)
    ap.add_argument("--wrist", type=int, default=0)
    ap.add_argument("--ckpt-bowl", default="outputs/train/act_mix_bowl_gpu/last")
    ap.add_argument("--wrist-ref", default="refs/wrist_ref_mix.jpg")
    ap.add_argument("--hover-dataset", default="local/so101-strike-mix")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-robot", action="store_true", help="腕を動かさない（画面と推論の確認用）")
    ap.add_argument("--snapshot", help="画面を1枚この名前で保存して終了（動作確認用）")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(a.hover_dataset)
    starts = ds.meta.episodes["dataset_from_index"]
    canon = np.median(np.stack([ds[int(i)]["observation.state"].numpy() for i in starts]), axis=0)
    model = load("act", a.ckpt_bowl, a.device)
    ref = cv2.imread(a.wrist_ref)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    robot = leader = None
    if not a.no_robot:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        robot = SO101Follower(SO101FollowerConfig(port=PORT, id="follower_01", num_read_retries=5))
        robot.connect()
        leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader_01"))
        leader.connect()
        follow_until_enter(robot, leader, "leaderで演奏のときの構えに近い形にして Enter > ")
        p = robot.bus.sync_read("Present_Position")
        start = np.array([p[j] for j in JOINTS])
        for k in range(1, 46):                     # 1.5秒で構えの中央値へ（急な飛びは安全制限で切られる）
            v = start + (canon - start) * k / 45
            robot.send_action({f"{j}.pos": float(x) for j, x in zip(JOINTS, v)})
            time.sleep(1 / 30)

    fcam, wcam = open_cam(a.front), open_cam(a.wrist)
    mode, aim, hist, t_aim = 0, None, [], 0.0
    try:
        while True:
            okf, front = fcam.read()
            okw, wrist = wcam.read()
            if not (okf and okw):
                time.sleep(0.05)
                continue
            if time.perf_counter() - t_aim > 0.5:
                aim, t_aim = bowl_aim(model, front, wrist, canon), time.perf_counter()
                hist = (hist + [aim])[-6:]
            dx, dy, bright = measure(ref_gray, cv2.cvtColor(wrist, cv2.COLOR_BGR2GRAY))
            right = [wrist, cv2.addWeighted(wrist, 0.5, ref, 0.5, 0),
                     cv2.applyColorMap(cv2.absdiff(ref, wrist).max(axis=2), cv2.COLORMAP_INFERNO)][mode]
            view = np.hstack([ref, right])
            good = aim is not None and min(hist) >= AIM_OK
            cv2.putText(view, "REF (training)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(view, ["LIVE", "BLEND", "DIFF"][mode] + f"  bright={bright:.2f} shift=({dx:+.0f},{dy:+.0f})",
                        (650, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(view, f"ACT bowl aim {aim:+.0f} deg  (need >= +{AIM_OK:.0f})" if aim is not None else "...",
                        (650, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0) if good else (0, 0, 255), 2)
            if a.snapshot:
                cv2.imwrite(a.snapshot, view)
                print(f"保存 {a.snapshot}  bowl照準 {aim:+.1f}°  明るさ {bright:.2f}")
                break
            cv2.imshow("wrist_align (q: quit, m: mode)", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                mode = (mode + 1) % 3
    finally:
        fcam.release()
        wcam.release()
        cv2.destroyAllWindows()
        if robot:
            follow_until_enter(robot, leader, "終了。leaderで待機姿勢(畳んだ形)に戻して Enter > ")
            leader.disconnect()
            robot.disconnect()
    print(f"最後の照準: {', '.join(f'{x:+.0f}' for x in hist)}°")


if __name__ == "__main__":
    main()
