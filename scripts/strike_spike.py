"""打楽器ストライクの遅延スパイク: 1関節を振り下ろす命令から、動き出し・停止（到達 or 接触）までを測る。
使い方:
  python scripts/strike_spike.py --dry                        # 動かさず現在角と負荷を表示（方向確認）
  python scripts/strike_spike.py --joint wrist_flex --delta -10 --reps 3   # 小さく試して向きを確認
  python scripts/strike_spike.py --joint wrist_flex --delta -25 --reps 8   # 本計測（下に対象物を置くと接触を測れる）
出力: 各回の t_move（命令→動き出し）、t_stop（命令→停止。自由空間なら到達、対象があれば接触）、負荷ピーク。
注意: follower にトルクが入って動く。周囲を空け、--delta は小さく始める。
      終了時はトルクOFF（脱力）するので、アームが落ちてもよい姿勢で終えること。
"""
import argparse, os, statistics, time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ポートは lerobot-find-port で調べて環境変数で渡す
PORT = os.environ.get("SO101_FOLLOWER_PORT", "/dev/cu.usbmodemXXXX")
LEADER_PORT = os.environ.get("SO101_LEADER_PORT", "/dev/cu.usbmodemYYYY")


def follow_until_enter(robot, leader, msg):
    """leader の動きを follower に流し続け、Enter が押されたら止める（構えを教える用）。"""
    import threading
    done = threading.Event()
    threading.Thread(target=lambda: (input(msg), done.set()), daemon=True).start()
    while not done.is_set():
        robot.send_action(leader.get_action())
        time.sleep(1 / 30)


def load_pct(raw: int) -> float:
    """Present_Load の生値 → 符号付き%（下位10bitが0〜1000=0〜100.0%、bit10が方向）。"""
    mag = (raw & 0x3FF) / 10.0
    return -mag if raw & 0x400 else mag


def first_stop(series, t_move, still_deg=0.3, hold_s=0.03):
    """動き出し以降で、hold_s の間ほぼ動かなくなった最初の時刻（到達 or 接触）。"""
    pts = [s for s in series if s[0] >= t_move]
    for i, (t, p, _) in enumerate(pts):
        win = [q for q in pts[i:] if q[0] - t <= hold_s]
        if win[-1][0] - t >= hold_s * 0.9 and max(q[1] for q in win) - min(q[1] for q in win) < still_deg:
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", default="wrist_flex")
    ap.add_argument("--delta", type=float, default=-10.0, help="振り下ろし量[deg]（符号=方向）")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--window", type=float, default=1.0, help="1回の計測時間[s]")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--teach", action="store_true",
                    help="leaderで構えを教えてから計測し、最後にleaderで待機姿勢へ戻す（自分のターミナルで実行）")
    a = ap.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=PORT, id="follower_01"))
    robot.connect()
    leader = None
    if a.teach:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader_01"))
        leader.connect()
    bus, j = robot.bus, a.joint
    try:
        if leader:
            follow_until_enter(robot, leader, "leaderで『叩く対象の真上の構え』に合わせて Enter > ")
        home = bus.sync_read("Present_Position")
        print("現在角[deg]:", {k: round(v, 1) for k, v in home.items()})
        print("負荷[%]:", {k: load_pct(v) for k, v in bus.sync_read("Present_Load").items()})
        if a.dry:
            return

        up = {f"{k}.pos": v for k, v in home.items()}
        down = dict(up)
        target = home[j] + a.delta
        down[f"{j}.pos"] = target
        moves, stops = [], []
        for r in range(1, a.reps + 1):
            robot.send_action(up)
            time.sleep(1.0)
            if r == 1:
                # 目標=現在角で落ち着いた後の負荷（Enter直後はまだleaderの角度を目標にしている）
                print("構え保持中の負荷[%]:", {k: load_pct(v) for k, v in bus.sync_read("Present_Load").items()})
            t0 = time.perf_counter()
            robot.send_action(down)
            series = []
            while (t := time.perf_counter() - t0) < a.window:
                p = bus.sync_read("Present_Position", [j])[j]
                l = load_pct(bus.sync_read("Present_Load", [j])[j])
                series.append((t, p, l))
            p0 = series[0][1]
            t_move = next((t for t, p, _ in series if abs(p - p0) > 1.0), None)
            t_stop = first_stop(series, t_move) if t_move is not None else None
            p_end = series[-1][1]
            hz = len(series) / a.window
            reached = abs(p_end - target) < 2.0
            # 加速中（動き出し前後）と、止まった瞬間（当たり）の前後で負荷を分けて見る
            acc = [abs(s[2]) for s in series if t_move is not None and s[0] <= t_move + 0.05]
            hit = [abs(s[2]) for s in series if t_stop is not None and t_stop - 0.02 <= s[0] <= t_stop + 0.06]
            cruise = [abs(s[2]) for s in series
                      if t_move is not None and t_stop is not None and t_move + 0.05 < s[0] < t_stop - 0.03]
            fmt = lambda xs, f: f"{f(xs):.0f}%" if xs else "-"
            print(f"[{r}] t_move={t_move and t_move*1000:.0f}ms  t_stop={t_stop and t_stop*1000:.0f}ms  "
                  f"終端={p_end:.1f}°(目標{target:.1f}°, {'到達' if reached else '途中で停止=接触?'})  "
                  f"負荷 加速{fmt(acc, max)}/移動中{fmt(cruise, statistics.median)}/当たり前後{fmt(hit, max)}  "
                  f"サンプル{hz:.0f}Hz")
            if t_move is not None:
                moves.append(t_move)
            if t_stop is not None:
                stops.append(t_stop)
        robot.send_action(up)
        time.sleep(0.8)

        def summ(xs):
            return f"平均{statistics.mean(xs)*1000:.0f}ms ±{(statistics.stdev(xs) if len(xs) > 1 else 0)*1000:.0f}ms (n={len(xs)})" if xs else "なし"
        print("\n=== まとめ ===")
        print("命令→動き出し:", summ(moves))
        print("命令→停止   :", summ(stops))
        if stops:
            cyc = 2 * statistics.mean(stops)
            print(f"往復の目安 ≈ {cyc*1000:.0f}ms → 1打/{cyc:.2f}s ≈ {60/cyc:.0f}打/分 が上限の目安")
        if leader:
            follow_until_enter(robot, leader, "計測終了。leaderで待機姿勢(畳んだ形)に戻して Enter > ")
    finally:
        if leader:
            leader.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
