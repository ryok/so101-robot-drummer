"""Ableton Link の拍に合わせて肘で叩き、実際に当たった時刻と拍のずれを測る。
使い方:
  python scripts/strike_sync.py --sim                                  # ロボット無しで拍と命令タイミングの計算だけ検証
  python scripts/strike_sync.py --offset 280 --every 2 --strikes 8      # 実機（leaderで構えを教えてから）
  --offset: 拍の何ms前に命令を出すか（初期値 = strike_spike の「命令→当たる」）
  --every : 何拍ごとに叩くか（120BPMなら 2 で1秒間隔。上限は約100回/分）
  Ableton を起動して Link を ON にすると、その拍に同期する（peers>=1）。未起動ならローカルの --bpm。
出力: 各打の「当たり − 拍」[ms]、平均と標準偏差、次に使う --offset の推奨値。
当たりの判定は角度（目標の手前で止まった＝コップに当たった）。目標まで届いたら空振り扱い。
"""
import argparse, asyncio, math, random, statistics, time

from aalink import Link

from strike_spike import LEADER_PORT, PORT, follow_until_enter


def wait_impact(read_pos, target, t_cmd, timeout=0.8, still_deg=0.3, hold_s=0.03):
    """命令後、動き出してから止まった時刻を返す。(停止時刻 or None, 空振りか)"""
    hist, p0, moved = [], None, False
    while time.perf_counter() - t_cmd < timeout:
        t, p = time.perf_counter(), read_pos()
        p0 = p if p0 is None else p0
        moved = moved or abs(p - p0) > 1.0
        hist.append((t, p))
        if moved:
            win = [q for q in hist if t - q[0] <= hold_s]
            if t - win[0][0] >= hold_s * 0.9 and max(q[1] for q in win) - min(q[1] for q in win) < still_deg:
                return win[0][0], abs(p - target) < 2.0
    return None, False


class SimArm:
    """ロボット無し検証用: 命令から latency±jitter 後に『当たって止まる』位置を返す。"""
    def __init__(self, latency, jitter):
        self.latency, self.jitter, self.t_cmd, self.hit_at = latency, jitter, None, None

    def strike(self):
        self.t_cmd = time.perf_counter()
        self.hit_at = self.t_cmd + self.latency + random.gauss(0, self.jitter)

    def read_pos(self):
        # 実機のバス読み出し（約4ms）を模す。待ち無しだと wait_impact の履歴が数千件に膨らみ、
        # 窓の計算が遅れて当たり検出が数百ms遅れる（シミュレーションだけで起きる見かけの誤差）
        time.sleep(0.004)
        t = time.perf_counter()
        if self.t_cmd is None or t < self.t_cmd + 0.09:
            return 0.0                                   # まだ動き出していない
        return 10.0 if t >= self.hit_at else 10.0 * (t - self.t_cmd) / (self.hit_at - self.t_cmd)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", default="elbow_flex")
    ap.add_argument("--delta", type=float, default=20.0, help="振り下ろし量[deg]（肘は + が下）")
    ap.add_argument("--offset", type=float, default=280.0, help="拍の何ms前に命令するか")
    ap.add_argument("--every", type=int, default=2, help="何拍ごとに叩くか")
    ap.add_argument("--strikes", type=int, default=8)
    ap.add_argument("--bpm", type=float, default=120.0, help="Link未接続時のテンポ")
    ap.add_argument("--sim", action="store_true", help="ロボット無しでタイミング計算だけ検証")
    a = ap.parse_args()

    link = Link(a.bpm)
    link.enabled = True
    await asyncio.sleep(1.0)
    print(f"Link: peers={link.num_peers} tempo={link.tempo:.1f}BPM  "
          f"{'(Abletonに同期)' if link.num_peers else '(ローカル。AbletonのLinkはOFFか未起動)'}")

    robot = leader = None
    if a.sim:
        arm = SimArm(latency=a.offset / 1000, jitter=0.005)
        target = 999.0                                   # シミュレーションでは常に「手前で停止」
        strike, read_pos, back = arm.strike, arm.read_pos, (lambda: None)
    else:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        robot = SO101Follower(SO101FollowerConfig(port=PORT, id="follower_01"))
        robot.connect()
        leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader_01"))
        leader.connect()
        follow_until_enter(robot, leader, "leaderで『叩く楽器の3〜5cm上に先端が来る構え』に合わせて Enter > ")
        home = robot.bus.sync_read("Present_Position")
        up = {f"{k}.pos": v for k, v in home.items()}
        down = dict(up)
        target = home[a.joint] + a.delta
        down[f"{a.joint}.pos"] = target
        robot.send_action(up)
        await asyncio.sleep(1.0)
        strike = lambda: robot.send_action(down)
        read_pos = lambda: robot.bus.sync_read("Present_Position", [a.joint])[a.joint]
        back = lambda: robot.send_action(up)

    errs, misses = [], 0
    try:
        for i in range(1, a.strikes + 1):
            spb = 60.0 / link.tempo
            b, now = link.beat, time.perf_counter()
            nb = (math.floor(b / a.every) + 1) * a.every            # 次の「every の倍数」の拍
            if (nb - b) * spb < a.offset / 1000 + 0.05:              # 前倒しが間に合わなければ次の打点へ
                nb += a.every
            t_beat = now + (nb - b) * spb
            await asyncio.sleep(max(0.0, t_beat - a.offset / 1000 - time.perf_counter()))
            t_cmd = time.perf_counter()
            strike()
            t_hit, whiff = wait_impact(read_pos, target, t_cmd)
            if t_hit is None or whiff:
                misses += 1
                print(f"[{i}] 拍{nb:.0f}: {'空振り（目標まで届いた＝コップに当たらず）' if whiff else '停止を検出できず'}")
            else:
                err = (t_hit - t_beat) * 1000
                errs.append(err)
                print(f"[{i}] 拍{nb:.0f}: 当たり−拍 = {err:+.0f}ms  （命令→当たる {(t_hit - t_cmd)*1000:.0f}ms）")
            back()
            await asyncio.sleep(0.05)
    finally:
        if leader:
            follow_until_enter(robot, leader, "終了。leaderで待機姿勢(畳んだ形)に戻して Enter > ")
            leader.disconnect()
        if robot:
            robot.disconnect()

    print("\n=== まとめ ===")
    if errs:
        mean = statistics.mean(errs)
        sd = statistics.stdev(errs) if len(errs) > 1 else 0.0
        print(f"当たり−拍: 平均{mean:+.0f}ms ±{sd:.0f}ms（n={len(errs)}、空振り{misses}）")
        print(f"次に使う --offset ≈ {a.offset + mean:.0f}ms（平均のずれを0にする値）")
    else:
        print(f"当たりなし（空振り{misses}）。構えを下げるか --delta を大きくする")


if __name__ == "__main__":
    asyncio.run(main())
