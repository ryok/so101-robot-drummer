"""演奏用: AIが「どの楽器を、どこへ狙うか」を決め、振り下ろしは Ableton Link の拍に合わせて固定の動きで打つ。

役割分担（10/23 の実測から）:
  - ポリシー（SmolVLA / ACT）: 演奏前に、構えのカメラ映像1枚と各楽器の指示文から推論し、
    チャンク中で肘が一番下がった瞬間の姿勢（机に対する絶対角）を「打撃姿勢」とする。
    照準姿勢 = 打撃姿勢から肘だけ --raise-deg 上げた姿勢（楽器の真上で待つ）
  - 打撃: 照準姿勢から打撃姿勢（＋必要なら少し深く）へ1回の命令で動かす。命令は拍の「前倒し量」だけ前に出す
    （ポリシーの動きそのものは叩く瞬間が ±50〜140ms ばらつき、拍に乗らないため）

楽譜: 1小節ずつ「楽器:16分音符のパターン」を空白区切りで並べる。x=打つ / .=休み（1小節=4拍、1文字=16分音符）
  例: --score "drum:x...x...x...x... bowl:x.......x....... drum:x.x.x.x........"
  楽器が変わるときは「肘を上げる→横へ回す→下ろす」で約0.6秒かけて移動する（楽器の縁に引っかけないため）。前の打点の戻りが間に合わない打点は飛ばして記録する。

使い方（自分のターミナルで。Ableton の Link を ON にしておく）:
  python scripts/strike_conductor.py --score "drum:x...x...x...x... bowl:x...x...x...x..."
  python scripts/strike_conductor.py --sim --score "..."      # ロボット無しで流れとタイミング計算だけ検証
  --calibrate 3: 演奏前に各楽器を3回ずつ打って「命令→当たる」を測り、前倒し量に使う（既定。0で --offset-* を使う）。
                 照準は深めに打ちながら pan を外側→内側へ4°刻みで動かし、音で当たった範囲の真ん中を使う。
                 その後、当たる最小の肘の深さ＋余裕を決め、命令→当たるの時間を測る。
                 照準の補正量（AI比）は表示・記録して、AIの照準がどれだけ外れていたかの指標にする
記録: outputs/perf/<日時>.csv（打点ごとの拍、当たり−拍[ms]、空振り/飛ばし）
"""
import argparse, asyncio, csv, math, os, statistics, time
from datetime import datetime

import numpy as np
import torch
from aalink import Link

from strike_policy import JOINTS, TASK, OfflineArm, load, to_policy_obs
from strike_spike import LEADER_PORT, PORT, follow_until_enter
from strike_sync import SimArm, wait_impact


def wait_impact_down(read_pos, target, t_cmd, timeout=0.8, still_deg=0.3, hold_s=0.03):
    """wait_impact の振り下ろし専用版: 構えから「下がる向き」に2°以上動いてから止まった時刻を当たりとする。
    前の打点から戻る途中（上がる向き）の停止を当たりと誤検出しない（10/23 実機で −250ms の誤検出が出た）。"""
    hist, p0, moved = [], None, False
    while time.perf_counter() - t_cmd < timeout:
        t, p = time.perf_counter(), read_pos()
        p0 = p if p0 is None else min(p0, p)          # 戻り途中なら、上がりきった位置を起点にし直す
        moved = moved or p - p0 > 2.0
        hist.append((t, p))
        hist = [q for q in hist if t - q[0] <= hold_s * 2]
        if moved:
            win = [q for q in hist if t - q[0] <= hold_s]
            if t - win[0][0] >= hold_s * 0.9 and max(q[1] for q in win) - min(q[1] for q in win) < still_deg:
                return win[0][0], abs(p - target) < 2.0
    return None, False

E = JOINTS.index("elbow_flex")


class Mic:
    """カメラ内蔵マイクで打音の立ち上がり時刻を取る（位置だけでは「当たった」と「サーボが目標の手前で止まった」
    を区別できない: 10/23 実機でログは当たり、目視は空振り）。時刻は time.perf_counter() の時計に揃える。"""

    def __init__(self, device, sr=48000):
        import collections
        import sounddevice as sd
        self.sr, self.buf = sr, collections.deque(maxlen=4000)   # 256サンプル×4000 ≈ 21秒分
        self.stream = sd.InputStream(device=device, channels=1, samplerate=sr, blocksize=256,
                                     callback=self._cb, latency="low")
        self.stream.start()
        time.sleep(0.2)
        self.off = time.perf_counter() - self.stream.time      # ストリームの時計 → perf_counter
        time.sleep(0.6)
        # 静かなときの音量 = 256サンプルごとの最大振幅の中央値。最大値にすると、測定中の一瞬の物音で閾値が
        # 10倍に跳ね、本物の打音（閾値の0.2〜0.5倍）を全部見逃した（10/24 実機）
        floor = float(np.median([np.abs(x).max() for _, x in list(self.buf)[-100:]] or [0.0]))
        self.th = None
        self.floor = floor

    def _cb(self, data, frames, t, status):
        self.buf.append((t.inputBufferAdcTime, data[:, 0].copy()))

    def set_threshold(self, k, minimum):
        self.th = max(self.floor * k, minimum)
        return self.th

    def onsets(self, t0, t1, refractory=0.15):
        """[t0, t1] の打音の立ち上がり時刻（perf_counter）の一覧。
        立ち上がり = 1ms ごとの振幅が閾値を超え、かつ直前50msの平均の2倍を超えた瞬間。
        金属のボウルは叩いた後も長く鳴るので、「閾値を超えている」だけで数えると余韻を何度も打音と数える
        （10/23 手で10回叩いて13回と出た）。また次の打点を余韻で早めに誤検出しないよう、急な立ち上がりだけを見る。"""
        blocks = [(adc + self.off, x) for adc, x in list(self.buf)
                  if adc + self.off + len(x) / self.sr >= t0 - 0.06 and adc + self.off <= t1]
        if not blocks:
            return []
        start = blocks[0][0]
        x = np.abs(np.concatenate([b for _, b in blocks]))
        hop = self.sr // 1000
        env = x[: len(x) // hop * hop].reshape(-1, hop).max(1)          # 1ms ごとの最大振幅
        out, last = [], -1e9
        for i in range(50, len(env)):
            tt = start + i / 1000
            if tt < t0 or tt > t1 or tt - last < refractory:
                continue
            if env[i] > self.th and env[i] > 2 * env[i - 50:i].mean():
                out.append(tt)
                last = tt
        return out

    def window_stats(self, t0, t1):
        """窓 [t0, t1] の最大振幅と、それが直前50msの平均の何倍か（空振りの原因切り分け用）。"""
        blocks = [(adc + self.off, x) for adc, x in list(self.buf)
                  if adc + self.off + len(x) / self.sr >= t0 - 0.06 and adc + self.off <= t1]
        if not blocks:
            return None
        start = blocks[0][0]
        x = np.abs(np.concatenate([b for _, b in blocks]))
        i0, i1 = int((t0 - start) * self.sr), int((t1 - start) * self.sr)
        pre = x[max(0, i0 - self.sr // 20):max(1, i0)].mean()
        peak = x[max(0, i0):max(i0 + 1, i1)].max()
        return float(peak), float(peak / max(pre, 1e-6))

    def onset(self, t0, t1):
        """[t0, t1] で最初の打音の立ち上がり時刻。無ければ None。"""
        on = self.onsets(t0, t1)
        return on[0] if on else None

    def wait_onset(self, t0, deadline):
        while time.perf_counter() < deadline:
            hit = self.onset(t0, deadline)
            if hit is not None:
                return hit
            time.sleep(0.01)
        time.sleep(0.03)                       # 最後のブロックが届くのを待って1回だけ確認
        return self.onset(t0, deadline)

    def close(self):
        self.stream.stop()
        self.stream.close()


def audio_note(rig):
    """1打ごとの音の情報: 窓内の最大振幅（閾値比）と直前50msからの立ち上がり倍率。"""
    if not rig.mic or not rig.last_audio:
        return ""
    peak, ratio = rig.last_audio
    return f"、音 {peak / rig.mic.th:.1f}×閾値・立ち上がり{ratio:.1f}倍{'、位置で判定' if rig.last_by == '位置' else ''}"


def as_act(v):
    return {f"{j}.pos": float(x) for j, x in zip(JOINTS, v)}


class Rig:
    """実機とシミュレーションの差を吸収する: 現在角、姿勢の補間移動、観測、肘の打撃と当たり検出。"""

    def __init__(self, sim, front, wrist, dataset):
        self.sim = sim
        self.robot = self.leader = None
        self.last_stop = 0.0
        self.last_audio = None
        self.last_by = "音"
        self.contact_deg = 4.0
        self.mic = None
        if sim:
            self.view = OfflineArm(dataset)       # 構えの観測（データセットの構えフレーム）
            self.view.reset_to(3)   # ep0 は bowl 指示でも左へ振る例外なので避ける
            self.pos = np.array([self.view.pos[f"{j}.pos"] for j in JOINTS])
            self.sim_arm = None
            return
        from lerobot.cameras.configs import Cv2Backends
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        cam = lambda i: OpenCVCameraConfig(index_or_path=i, fps=30, width=640, height=480,
                                           backend=Cv2Backends.AVFOUNDATION)
        self.robot = SO101Follower(SO101FollowerConfig(port=PORT, id="follower_01", num_read_retries=5,
                                                       max_relative_target=60.0,   # 打撃は1回の命令で肘を最大50°動かす。これより小さいと目標が切られ、
                                                       # 切られた位置で止まったのを「当たり」と誤判定する（10/23 ACTで発生）
                                                       cameras={"front": cam(front), "wrist": cam(wrist)}))
        self.robot.connect()
        self.leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader_01"))
        self.leader.connect()

    def current(self):
        if self.sim:
            return self.pos.copy()
        p = self.robot.bus.sync_read("Present_Position")
        return np.array([p[j] for j in JOINTS])

    def send(self, v):
        if self.sim:
            self.pos = np.array(v, dtype=float)
        else:
            self.robot.send_action(as_act(v))

    def move_to(self, target, secs):
        """現在角から target まで secs 秒で直線補間（急な飛びは max_relative_target で切られるため）。"""
        # 経過時間で補間する。回数で割ると、1回の送信（安全制限のための位置読み出しを含む）が1/30秒を超えたぶん
        # 全体が延び、楽器切替（名目0.7秒）が間に合わず切替直後の打点が毎回「飛ばし」になる（10/25 実機）
        start, t0, target = self.current(), time.perf_counter(), np.array(target, dtype=float)
        while True:
            k = min(1.0, (time.perf_counter() - t0) / secs) if secs > 0 else 1.0
            self.send(start + (target - start) * k)
            if k >= 1.0:
                return
            if not self.sim:
                time.sleep(1 / 60)

    def observe(self, task):
        if self.sim:
            return to_policy_obs(self.view.get_observation(), task)
        return to_policy_obs(self.robot.get_observation(), task)

    def strike(self, down, offset_s):
        """打撃姿勢 down へ1回の命令で動かす。肘の停止で当たりを判定し、当たり時刻と空振りかを返す。"""
        down = np.asarray(down, dtype=float)
        target_elbow = down[E]
        self.last_by = "音"
        t_cmd = time.perf_counter()
        if self.sim:
            self.sim_arm = SimArm(latency=offset_s, jitter=0.005)
            self.sim_arm.strike()
            t_hit, whiff = wait_impact(self.sim_arm.read_pos, 999.0, t_cmd)
        else:
            self.send(down)
            read = lambda: self.robot.bus.sync_read("Present_Position", ["elbow_flex"])["elbow_flex"]
            t_hit, whiff = wait_impact_down(read, target_elbow, t_cmd)
            self.last_stop = read() - target_elbow     # 止まった肘の角度 − 目標（負 = 目標の手前で楽器に止められた）
            if self.mic:
                # 音で判定する。ただし腕が動き出すときのモーター音・機構音（命令の約120ms後）を打音と誤認しないよう、
                # 腕が止まった時刻（位置で検出）の前後だけを見る（10/23 実機で 122ms の「当たり」を全打で誤検出）
                if t_hit is not None:
                    w0, w1 = t_hit - 0.08, t_hit + 0.05
                else:
                    w0, w1 = t_cmd + 0.18, t_cmd + 0.7
                t_pos = t_hit
                t_snd = self.mic.wait_onset(w0, w1 + 0.03)
                self.last_audio = self.mic.window_stats(w0, w1)
                if t_snd is not None and t_snd <= w1:
                    t_hit, whiff = t_snd, False
                elif t_pos is not None and self.last_stop <= -self.contact_deg:
                    # プラスチックのドラムは打音が小さく、腕のモーター音と同じくらい（閾値の2〜4倍）で音では拾えない
                    # （10/24 実機: 目視で全打当たりなのに音では0）。目標より contact_deg 以上手前で止まった＝楽器に
                    # 止められた、とみなす（サーボの誤差は2〜3°、空振りは0〜−2°）。時刻は位置の停止時刻
                    t_hit, whiff = t_pos, False
                    self.last_by = "位置"
                else:
                    t_hit, whiff = None, True
        return t_cmd, t_hit, whiff

    def close(self):
        if self.mic:
            self.mic.close()
        for dev in (self.leader, self.robot):
            if dev:
                try:
                    dev.disconnect()
                except Exception as ex:
                    print(f"切断時のエラー: {ex}")


class AbletonSong:
    """Ableton Live（AbletonMCP の Remote Script, TCP 9877）に直接つないで曲のクリップを鳴らす・止める。
    ableton コマンドを毎回起動すると1回0.3秒前後かかり、5トラック発火が小節をまたぐため、ソケットで直接送る。
    クリップは Live の発火クオンタイズ（既定1小節）で次の小節頭から鳴る。Link の quantum を4にしてあるので、
    その小節頭はロボットの小節頭と一致する。"""

    def __init__(self, tracks, host="localhost", port=9877):
        import socket
        self.tracks = tracks
        self.sock = socket.create_connection((host, port), timeout=5)

    def cmd(self, type_, **params):
        import json
        self.sock.sendall(json.dumps({"type": type_, "params": params}).encode())
        buf = b""
        while True:
            buf += self.sock.recv(65536)
            try:
                r = json.loads(buf.decode())
                break
            except ValueError:
                continue
        if r.get("status") == "error":
            raise RuntimeError(f"Ableton: {r.get('message')}")
        return r.get("result", {})

    def fire(self):
        for t in self.tracks:
            self.cmd("fire_clip", track_index=t, clip_index=0)

    def close(self):
        try:
            for t in self.tracks:
                self.cmd("stop_clip", track_index=t, clip_index=0)
            self.cmd("stop_playback")
        finally:
            self.sock.close()


def check_view(front_rgb, save_path, ref_path, tol, name="俯瞰"):
    """照準に使う俯瞰の1枚を保存し、学習データの基準画像との明るさ・ずれを表示する。
    ACT は同じ画像なら毎回同じ照準を出すので、照準が前回と変わったら原因は画像（照明・楽器・カメラ）。
    10/24 に同じ日の40分後で bowl +12° → -7° になり、後から条件を確かめられなかったため残す。"""
    import cv2
    from align_camera import measure
    bgr = cv2.cvtColor(np.asarray(front_rgb), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr)
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    cur = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), ref.shape[::-1])
    dx, dy, bright = measure(ref, cur)
    ok = abs(bright - 1) <= tol and math.hypot(dx, dy) <= 5
    print(f"{name}: 明るさ {bright:.2f}（基準=1.00）  ずれ dx={dx:+.1f} dy={dy:+.1f}px  → {'OK' if ok else '要確認'}"
          f"  保存 {save_path}")
    if not ok:
        print("  照明・カメラ・背景が学習データと違う。照準がずれやすい"
              f"（align_camera.py {'1' if name == '俯瞰' else '0'} --ref {ref_path} で合わせる）")


def aim_from_policy(models, inst, obs_fn, hover, raise_deg, samples):
    """構えの観測1枚と指示文から推論し、打撃姿勢と照準姿勢を返す（どちらも机に対する絶対角）。
    打撃姿勢 = チャンク中で肘が最も下がった瞬間の全関節（= お手本で楽器に当たった姿勢）。samples 回の中央値。
    照準姿勢 = 打撃姿勢から肘だけ raise_deg 上げた姿勢（楽器の真上、少し高い所で待つ）。
    構えからの相対角で組むと、人が教える構えが日によって ±18° ずれるだけで外れる（10/23 実機で判明）ので、
    ポリシーが出す絶対角をそのまま使う。グリッパーは構えのまま。
    SmolVLA は推論ごとにノイズから生成するので出力がばらつく。中央値で抑える。"""
    policy, pre, post = models[inst]
    peaks, ms = [], []
    for _ in range(samples):
        policy.reset()
        t0 = time.perf_counter()
        with torch.inference_mode():
            chunk = post(policy.predict_action_chunk(pre(obs_fn(TASK[inst]))))[0].cpu().numpy()
        ms.append((time.perf_counter() - t0) * 1000)
        peaks.append(chunk[int(np.argmax(chunk[:, E]))])
    down = np.median(np.array(peaks), axis=0)
    down[JOINTS.index("gripper")] = hover[JOINTS.index("gripper")]
    aim = down.copy()
    aim[E] -= raise_deg
    return aim, down, float(np.mean(ms)), [p[0] for p in peaks]


def parse_score(s):
    bars = []
    for tok in s.split():
        inst, pat = tok.split(":")
        inst = {"d": "drum", "b": "bowl"}.get(inst, inst)
        assert inst in TASK and set(pat) <= {"x", "."}, f"楽譜の書式が不正: {tok}"
        bars.append((inst, pat))
    return bars


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True, help="楽譜の文字列、またはそれを書いたファイルのパス（例: song/score.txt）")
    ap.add_argument("--song-tracks", help="演奏の頭で Ableton のこのトラック（枠0のクリップ）を鳴らす。例: 0,1,2,3,4")
    ap.add_argument("kind", nargs="?", default="smolvla", choices=["smolvla", "act"])
    ap.add_argument("--ckpt", default="outputs/train/smolvla_mix_gpu/last")
    ap.add_argument("--ckpt-drum", default="outputs/train/act_mix_drum_gpu/last")
    ap.add_argument("--ckpt-bowl", default="outputs/train/act_mix_bowl_gpu/last")
    ap.add_argument("--adapt", type=float, default=0.3, help="当たり−拍のずれを前倒し量に反映する割合（0=補正しない）")
    ap.add_argument("--samples", type=int, default=5, help="照準を決めるときの推論回数（中央値を使う）")
    ap.add_argument("--hover-dataset", default="local/so101-strike-mix",
                    help="この学習データの構えの中央値を構えに使う（空文字なら leader で教えた構えのまま）")
    ap.add_argument("--mic", type=int, default=4, help="打音を拾うマイクのデバイス番号（-1 で使わない＝位置だけで判定）")
    ap.add_argument("--ref", default="refs/front_ref_mix.jpg", help="俯瞰の基準画像（学習データの構えの中央値）")
    ap.add_argument("--wrist-ref", default="refs/wrist_ref_mix.jpg",
                    help="wrist の基準画像。wrist は机の外（扇風機・箱・鞄）まで写し、ACTの照準はこれで動く（10/25 判明）")
    ap.add_argument("--bright-tol", type=float, default=0.05, help="基準との明るさの比の許容（±）")
    ap.add_argument("--mic-k", type=float, default=5.0, help="閾値 = 静かなときの最大振幅 × これ")
    ap.add_argument("--mic-min", type=float, default=0.05, help="閾値の下限")
    ap.add_argument("--transit-lift", type=float, default=20.0, help="[deg] 楽器を切り替えるとき、照準姿勢よりさらに肘を上げる量")
    ap.add_argument("--transit-times", type=lambda x: [float(v) for v in x.split(",")], default=[0.12, 0.25, 0.25],
                    help="[s] 楽器切替の 上げ,回す,下ろす の時間。下ろしを0.25未満にすると直後の打が約60ms早くなる（10/24）")
    ap.add_argument("--search-depth", type=float, default=6.0,
                    help="[deg] 照準を探すときの肘の深さ（深すぎると楽器の横の机にも当たり範囲が決まらない）")
    ap.add_argument("--contact-deg", type=float, default=4.0,
                    help="[deg] 音が拾えなくても、目標よりこれ以上手前で止まったら当たりとみなす")
    ap.add_argument("--margin", type=float, default=4.0,
                    help="[deg] 当たった最小の深さに足す余裕（照準を当たる範囲の真ん中に置くので、滑りにくい）")
    ap.add_argument("--raise-deg", type=float, default=25.0, help="[deg] 照準姿勢で、打撃姿勢より肘を上げておく量")
    ap.add_argument("--max-extra", type=float, default=16.0, help="[deg] 空振り時に打撃姿勢より肘を深くする上限")
    ap.add_argument("--calibrate", type=int, default=3, help="演奏前に各楽器を何回打って前倒し量を測るか（0=測らない）")
    ap.add_argument("--offset-drum", type=float, default=184.0, help="[ms] --calibrate 0 のときの前倒し量")
    ap.add_argument("--offset-bowl", type=float, default=236.0)
    ap.add_argument("--bpm", type=float, default=100.0, help="Link 未接続時のテンポ")
    ap.add_argument("--settle", type=float, default=0.2,
                    help="[s] 前の打点の当たりから次の命令まで最低これだけ空ける（戻りが間に合わない打点は飛ばす）")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--front", type=int, default=1)
    ap.add_argument("--wrist", type=int, default=0)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--dataset", default="local/so101-strike-mix", help="--sim の構えの観測に使う")
    a = ap.parse_args()
    if a.mic is not None and a.mic < 0:
        a.mic = None

    bars = parse_score(open(a.score).read() if os.path.isfile(a.score) else a.score)
    for i in range(1, len(bars)):
        if bars[i][0] != bars[i - 1][0] and bars[i][1][0] == "x" and bars[i - 1][1][-len(bars[i - 1][1]) // 4:].count("x"):
            print(f"注意: 小節{i + 1}は楽器が変わる直後の頭打ち。照準の移動(約0.4秒)が間に合わず飛ばされやすい。"
                  f"前の小節の4拍目か、この小節の1拍目を休みにする")
    if a.kind == "smolvla":
        m = load("smolvla", a.ckpt, a.device)
        models = {"drum": m, "bowl": m}
    else:
        models = {"drum": load("act", a.ckpt_drum, a.device), "bowl": load("act", a.ckpt_bowl, a.device)}

    link = Link(a.bpm)
    link.quantum = 4      # 小節（4拍）単位で位相を合わせる。1のままだと beat の4の倍数が Live の小節頭と一致しない
    link.enabled = True
    await asyncio.sleep(1.0)
    song = None
    if a.song_tracks:
        song = AbletonSong([int(t) for t in a.song_tracks.split(",")])
        song.cmd("start_playback")      # 先にトランスポートだけ動かす（クリップは止めたまま＝無音）。止まったまま発火すると
        # その瞬間から再生が始まり、小節頭に揃わない
        print(f"Ableton: トラック {a.song_tracks} の曲を演奏の頭で鳴らす")
    print(f"Link: peers={link.num_peers} tempo={link.tempo:.1f}BPM "
          f"{'(Abletonに同期)' if link.num_peers else '(ローカル)'}")
    if song and not link.num_peers:
        msg = "Live の Link が OFF。小節頭が揃わないので、Live の右上の Link を ON にしてやり直す"
        if not a.sim:
            song.close()
            raise SystemExit(msg)
        print("注意: " + msg + "（シミュレーションなので続ける）")

    rig = Rig(a.sim, a.front, a.wrist, a.dataset)
    rig.contact_deg = a.contact_deg
    offset = {"drum": a.offset_drum / 1000, "bowl": a.offset_bowl / 1000}
    os.makedirs("outputs/perf", exist_ok=True)
    path = f"outputs/perf/{'sim_' if a.sim else ''}{a.kind}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    log = []
    try:
        if not a.sim:
            follow_until_enter(rig.robot, rig.leader, "leaderで『録画のときと同じ構え』に合わせて Enter > ")
        hover = rig.current()
        if a.hover_dataset and not a.sim:
            # 構えは人が毎回教えると ±10〜18° ずれ、ACT の照準まで変わる（構え pan −10° のとき bowl の照準が +9° に
            # 落ちて全打外れ: 10/24 実機。学習データの構えは pan −10〜+4°）。学習データの構えの中央値へ自動で移す
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            ds = LeRobotDataset(a.hover_dataset)
            starts = ds.meta.episodes["dataset_from_index"]
            canon = np.median(np.stack([ds[int(i)]["observation.state"].numpy() for i in starts]), axis=0)
            print("構え: 教えた姿勢 → 学習データの構えの中央値へ移動  差",
                  " ".join(f"{j.split('_')[-1]}{c - h:+.0f}" for j, c, h in zip(JOINTS, canon, hover)))
            rig.move_to(canon, 1.5)
            hover = canon.astype(float)
        if a.mic is not None and not a.sim:
            rig.mic = Mic(a.mic)
            th = rig.mic.set_threshold(a.mic_k, a.mic_min)
            print(f"マイク: {a.mic}  静かなときの最大振幅 {rig.mic.floor:.4f} → 打音の閾値 {th:.4f}"
                  f"（クリックはヘッドホンで聞く。スピーカーだと打音と誤検出する）")
            if rig.mic.floor < 1e-4:   # 完全な無音＝別のデバイスを開いている（USBの抜き差しで番号が入れ替わる）
                raise SystemExit(f"マイク {a.mic} が無音です。python scripts/mic_check.py --mic N で"
                                 f"音が入る番号を確かめて --mic N を付けてください")
        insts = sorted({i for i, _ in bars})
        # 照準は演奏の前に、同じ構え・同じ観測から楽器ごとに推論して決める（指示文だけを変える）。
        # 演奏中に構えへ戻って推論し直すと、カメラ映像が動きに遅れて照準がずれる（10/23 実機で pan±0° になった）
        if not a.sim:
            time.sleep(1.0)
            for _ in range(5):                       # カメラの古いフレームを捨てる
                rig.robot.get_observation()
        raw = rig.view.get_observation() if a.sim else rig.robot.get_observation()
        if not a.sim:
            check_view(raw["front"], path.replace(".csv", "_front.jpg"), a.ref, a.bright_tol)
            check_view(raw["wrist"], path.replace(".csv", "_wrist.jpg"), a.wrist_ref, a.bright_tol, "wrist")
        obs_fn = lambda task: to_policy_obs(raw, task)
        for inst in insts:                            # 準備運転（初回推論の遅さ）
            policy, pre, post = models[inst]
            with torch.inference_mode():
                policy.reset()
                policy.predict_action_chunk(pre(obs_fn(TASK[inst])))
        aims, downs, depth = {}, {}, {}
        print("=== 照準（同じ構えの映像に、指示文だけ変えて推論）===")
        for inst in insts:
            aims[inst], downs[inst], infer_ms, pans = aim_from_policy(models, inst, obs_fn, hover, a.raise_deg, a.samples)
            depth[inst] = 0.0
            print(f"  {inst}: 打撃姿勢 pan{downs[inst][0]:+.0f}°（{a.samples}回: {' '.join(f'{p:+.0f}' for p in pans)}）"
                  f" 肩{downs[inst][1]:+.0f}° 肘{downs[inst][2]:+.0f}° 手首{downs[inst][3]:+.0f}°（絶対角）"
                  f"  構えとの差 pan{downs[inst][0] - hover[0]:+.0f}°  推論 平均{infer_ms:.0f}ms")

        def target(inst):
            down = downs[inst].copy()
            down[E] += depth[inst]          # 打撃姿勢よりさらに depth だけ肘を下げた所を目標にする（楽器で止まる前提）
            return down

        if a.calibrate:
            print("=== 照準と深さの確認・前倒し量の測定（音で当たりを判定）===")
            for inst in insts:
                pan_ai = downs[inst][0]
                side = 1.0 if pan_ai >= hover[0] else -1.0          # 外側 = AIが振った向き

                def try_strike(pan, extra):
                    aims[inst][0] = downs[inst][0] = pan
                    depth[inst] = extra
                    rig.move_to(aims[inst], 0.3)
                    time.sleep(0.25)
                    t_cmd, t_hit, whiff = rig.strike(target(inst), offset[inst])
                    rig.move_to(aims[inst], 0.15)
                    ok = bool(t_hit) and not whiff
                    print(f"    {inst}: pan{pan:+.0f}°（AI比 {pan - pan_ai:+.0f}°） 肘+{extra:.0f}° → {'当たり' if ok else '空振り'}"
                          f"（止まった位置: 目標{rig.last_stop:+.0f}°{audio_note(rig)}）")
                    time.sleep(0.3)
                    return ok, (t_hit - t_cmd) if ok else None

                # 1) 照準: 浅め（--search-depth）に打って、当たる pan の範囲を探し、その真ん中を使う（端に当たると滑って
                #    外れる: 10/23）。深く打つと楽器の横の机や縁にまで当たり、どの pan でも「当たり」になって範囲が
                #    決まらない（10/24: +16°で pan +2〜+26° の全部が当たり）。伏せたボウルは中央が一番高いので、
                #    浅く打つほど中央付近だけが当たる。外れは「外側への振り不足」が多いので、外側を先に探す
                hits, lat_search = [], []
                for d in [0, 4, 8, 12, 16, -4, -8]:
                    pan = pan_ai + d * side
                    ok, l = try_strike(pan, a.search_depth)
                    if ok:
                        hits.append(pan)
                        lat_search.append(l)
                    elif hits:
                        break                       # 当たる範囲を通り過ぎた
                if hits:
                    if len(hits) == 1:              # 1点だけなら、さらに外側も確かめて範囲の端か確認
                        ok, _ = try_strike(hits[0] + 4 * side, a.search_depth)
                        if ok:
                            hits.append(hits[0] + 4 * side)
                    pan_best = float(np.mean(hits))
                else:
                    pan_best = pan_ai
                # 2) 深さ: 浅い所から当たる最小の深さを探し、余裕を足す
                #    ただし探索の深さより浅くはしない。浅すぎると当たっても音が小さく、偶然1回当たった深さ（+0°）を
                #    基準にすると本番の打が軽く・不安定になる（10/25: +0°で1回当たり → +4°で3回とも判定できず）
                extra_best = a.max_extra
                for extra in np.arange(0, a.max_extra + 0.1, 4):
                    ok, _ = try_strike(pan_best, float(extra))
                    if ok:
                        extra_best = min(a.max_extra, max(float(extra) + a.margin, a.search_depth if hits else 0.0))
                        break
                # 3) 前倒し量。半分以上当たらなければ4°深くしてもう一度
                for _ in range(2):
                    lat = [l for l in (try_strike(pan_best, extra_best)[1] for _ in range(a.calibrate)) if l]
                    if 2 * len(lat) > a.calibrate or extra_best >= a.max_extra:
                        break
                    extra_best = min(a.max_extra, extra_best + 4)
                lat = lat or lat_search          # 測定の打が当たらなくても、探索で当たった打の時間を使う（既定値は実機と60ms以上違う）
                if lat:
                    offset[inst] = statistics.mean(lat)
                aims[inst][0] = downs[inst][0] = pan_best
                depth[inst] = extra_best
                rig.move_to(aims[inst], 0.3)
                print(f"  {inst}: 照準 pan{pan_best:+.0f}°（AI比 {pan_best - pan_ai:+.0f}°、当たった範囲 "
                      f"{' '.join(f'{p - pan_ai:+.0f}' for p in hits) or 'なし'}） 肘+{extra_best:.0f}°"
                      f"  命令→当たる {[round(x * 1000) for x in lat]}ms → 前倒し {offset[inst] * 1000:.0f}ms"
                      f"{'' if lat else '（当たらず。楽器の位置を確認）'}")

        print(f"=== 演奏（{link.tempo:.1f}BPM、1小節 {240 / link.tempo:.2f}秒）===")
        spb = 60.0 / link.tempo
        start = (math.floor(link.beat / 4) + 2) * 4     # 次の次の小節頭から
        if song:
            # 演奏頭の3拍前（=1つ前の小節の中）に発火 → Live のクオンタイズで start の小節頭から鳴る
            # 1つ前の小節に入った直後に発火する。発火は1トラック約0.3秒かかる（Remote Script が Live の主スレッドで
            # 順に処理する）ので、5トラックで約1.5秒。全部がこの小節のうちに届けば start の小節頭から同時に鳴る
            await asyncio.sleep(max(0.0, (start - 3.9 - link.beat) * 60.0 / link.tempo))
            print(f"  曲を発火（拍 {link.beat:.2f}、頭は {start}）")
            fire_task = asyncio.create_task(asyncio.to_thread(song.fire))
        current, last_hit = None, 0.0
        for bi, (inst, pat) in enumerate(bars):
            bar_beat = start + 4 * bi
            if inst != current:
                # 楽器の切り替え: 肘を上げてから横へ回し、下ろす。照準姿勢どうしを直線で結ぶと、
                # 途中でスティックがドラムの縁に引っかかる（10/23 実機）
                up_here, up_there = rig.current(), aims[inst].copy()
                up_here[E] -= a.transit_lift
                up_there[E] -= a.transit_lift
                t_tr = time.perf_counter()
                rig.move_to(up_here, a.transit_times[0])
                rig.move_to(up_there, a.transit_times[1])
                rig.move_to(aims[inst], a.transit_times[2])       # 下ろしが速いと揺れが残り、直後の打点だけ約60ms早く当たる（10/24）
                t_first = time.perf_counter() + (bar_beat + pat.index("x") * 4 / len(pat) - link.beat) * 60.0 / link.tempo \
                    - offset[inst] if "x" in pat else None
                if current is not None and t_first is not None:
                    print(f"  （{inst}へ切替 {(time.perf_counter() - t_tr) * 1000:.0f}ms、"
                          f"最初の命令まで残り {(t_first - time.perf_counter()) * 1000:+.0f}ms）")
                current = inst
            aim = aims[inst]
            for si, ch in enumerate(pat):
                if ch != "x":
                    continue
                nb = bar_beat + si * 4 / len(pat)
                spb = 60.0 / link.tempo
                t_beat = time.perf_counter() + (nb - link.beat) * spb
                t_cmd_plan = t_beat - offset[inst]
                row = dict(bar=bi + 1, step=si, instrument=inst, beat=round(nb, 3))
                rig.last_audio = None
                if t_cmd_plan < time.perf_counter() + 0.005 or t_cmd_plan - last_hit < a.settle:
                    row.update(err_ms="", result="skip")
                    print(f"  小節{bi + 1} {si:2d} {inst}: 飛ばし（準備か前の打点の戻りが間に合わない）")
                else:
                    await asyncio.sleep(max(0.0, t_cmd_plan - time.perf_counter()))
                    t_cmd, t_hit, whiff = rig.strike(target(inst), offset[inst])
                    rig.send(aim)
                    last_hit = t_hit or t_cmd
                    if t_hit is None or whiff:
                        row.update(err_ms="", result="whiff" if whiff else "no_stop")
                        print(f"  小節{bi + 1} {si:2d} {inst}: 空振り（止まった位置: 目標{rig.last_stop:+.0f}°{audio_note(rig)}）")
                    else:
                        err = (t_hit - t_beat) * 1000
                        late = (t_cmd - t_cmd_plan) * 1000
                        row.update(err_ms=round(err, 1), cmd_late_ms=round(late, 1), result="hit")
                        # 演奏中も前倒し量を少しずつ補正する。1打の外れ値で大きく振れないよう1回 ±12ms まで
                        offset[inst] += float(np.clip(err * a.adapt, -12, 12)) / 1000
                        print(f"  小節{bi + 1} {si:2d} {inst}: 当たり−拍 {err:+.0f}ms（命令の遅れ {late:+.0f}ms{audio_note(rig)}）")
                if rig.last_audio:
                    row.update(audio_peak=round(rig.last_audio[0], 4), audio_rise=round(rig.last_audio[1], 2))
                log.append(row)
        rig.move_to(hover, 0.6)
        if song:
            await fire_task
        if song:   # 最後の小節が終わり、余韻を1小節聞かせてから止める（クリップはループするので止めないと頭に戻る）
            await asyncio.sleep(max(0.0, (start + 4 * (len(bars) + 1) - link.beat) * 60.0 / link.tempo))
            song.close()
            song = None
        if not a.sim:
            follow_until_enter(rig.robot, rig.leader, "終了。leaderで待機姿勢(畳んだ形)に戻して Enter > ")
    finally:
        if song:
            song.close()
        if log:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["bar", "step", "instrument", "beat", "err_ms", "cmd_late_ms", "result",
                                                  "audio_peak", "audio_rise"])
                w.writeheader()
                w.writerows(log)
        rig.close()

    print("\n=== まとめ ===")
    for inst in ("drum", "bowl"):
        r = [x for x in log if x["instrument"] == inst]
        if not r:
            continue
        errs = [x["err_ms"] for x in r if x["result"] == "hit"]
        s = f"{inst}: 打点{len(r)} 当たり{len(errs)} 空振り{sum(x['result'] in ('whiff', 'no_stop') for x in r)} " \
            f"飛ばし{sum(x['result'] == 'skip' for x in r)}"
        if errs:
            s += f"  当たり−拍 平均{statistics.mean(errs):+.0f}ms ±{statistics.pstdev(errs):.0f}ms"
        print(s)
    print("→", path)


if __name__ == "__main__":
    asyncio.run(main())
