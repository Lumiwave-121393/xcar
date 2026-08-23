#!/usr/bin/env python3
"""
====================================
  手动坐标发送工具
  向目标 IP 发送小车定位数据（UDP）
====================================

用法:
  python send_coord.py <x_mm> <y_mm> [yaw_deg] [选项]

示例:
  python send_coord.py 1500 2500
  python send_coord.py 1500 2500 45
  python send_coord.py 1500 2500 0 --ip 192.168.213.7 --port 9006
  python send_coord.py 1500 2500 0 --loop 0.1   # 每 0.1 秒持续发送

交互模式:
  python send_coord.py --interactive
"""

import argparse
import json
import socket
import time
import sys


def send_position(ip, port, x_mm, y_mm, yaw_deg=0.0,
                  z_m=0.0, pitch_deg=0.0, roll_deg=0.0, verbose=True):
    """
    发送小车位姿到目标设备。
    参数与 main.py 中 UdpPositionSender.send() 的映射规则一致。
    """
    # 坐标系转换（与 udp_sender.py 一致）
    pos_x = x_mm / 1000.0          # tag_X → 世界 X (米)
    pos_y = z_m                     # 高度 (米)
    pos_z = y_mm / 1000.0          # tag_Y → 世界 Z (米)
    remapped_yaw = -yaw_deg        # Z 轴翻转导致 yaw 方向翻转

    message = json.dumps({
        "type": "robot_position",
        "pos": [pos_x, pos_y, pos_z],
        "euler": [pitch_deg, remapped_yaw, roll_deg],
        "ts": time.time()
    })

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(message.encode("utf-8"), (ip, port))
        if verbose:
            xyz = f"({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f}) m"
            euler = f"({pitch_deg:.1f}, {remapped_yaw:.1f}, {roll_deg:.1f})°"
            print(f"[OK] 已发送 → {ip}:{port}")
            print(f"     坐标: {xyz}")
            print(f"     姿态: {euler}")
            print(f"     原始输入: X={x_mm:.0f}mm  Y={y_mm:.0f}mm  Yaw={yaw_deg:.1f}°")
    except Exception as e:
        print(f"[ERROR] 发送失败: {e}", file=sys.stderr)
    finally:
        sock.close()


def interactive_mode(ip, port):
    """交互模式：持续读取用户输入的坐标"""
    print("=" * 50)
    print("  交互模式 — 输入坐标发送，按 Ctrl+C 退出")
    print("=" * 50)
    print(f"  目标: {ip}:{port}")
    print("  格式: x_mm y_mm [yaw_deg]")
    print("  示例: 1500 2500 45")
    print()

    while True:
        try:
            line = input(">>> ").strip()
            if not line:
                continue
            parts = line.split()
            x_mm = float(parts[0])
            y_mm = float(parts[1])
            yaw_deg = float(parts[2]) if len(parts) > 2 else 0.0
            send_position(ip, port, x_mm, y_mm, yaw_deg)
            print()
        except KeyboardInterrupt:
            print("\n[INFO] 退出")
            break
        except (ValueError, IndexError):
            print("[ERROR] 格式错误，请使用: x_mm y_mm [yaw_deg]")


def loop_mode(ip, port, x_mm, y_mm, yaw_deg, interval):
    """循环发送模式：按固定间隔持续发送坐标"""
    print(f"[INFO] 开始循环发送 → {ip}:{port}")
    print(f"      坐标: X={x_mm:.0f}mm  Y={y_mm:.0f}mm  Yaw={yaw_deg:.1f}°")
    print(f"      间隔: {interval:.2f}秒  (按 Ctrl+C 停止)")
    print()

    try:
        while True:
            send_position(ip, port, x_mm, y_mm, yaw_deg, verbose=False)
            print(f"\r[{time.strftime('%H:%M:%S')}] 已发送 "
                  f"X={x_mm:.0f} Y={y_mm:.0f} Yaw={yaw_deg:.1f}°  ", end="", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[INFO] 停止循环发送")


def main():
    parser = argparse.ArgumentParser(
        description="向目标 IP 发送小车定位坐标（UDP）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s 1500 2500
  %(prog)s 1500 2500 45
  %(prog)s 1500 2500 --ip 192.168.213.7 --port 9006
  %(prog)s 1500 2500 --loop 0.05   # 每秒 20 次
  %(prog)s --interactive
        """,
    )

    # 位置参数
    parser.add_argument("x_mm", nargs="?", type=float,
                        help="X 坐标（毫米）")
    parser.add_argument("y_mm", nargs="?", type=float,
                        help="Y 坐标（毫米）")
    parser.add_argument("yaw_deg", nargs="?", type=float, default=0.0,
                        help="偏航角（度，可选，默认 0）")

    # 选项
    parser.add_argument("--ip", default="192.168.213.7",
                        help="目标 IP 地址（默认: 192.168.213.7）")
    parser.add_argument("--port", type=int, default=9005,
                        help="目标端口（默认: 9005）")
    parser.add_argument("--loop", type=float, metavar="间隔秒",
                        help="循环发送模式：按指定间隔（秒）持续发送")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="交互模式：逐行输入坐标发送")

    args = parser.parse_args()

    # 交互模式
    if args.interactive:
        interactive_mode(args.ip, args.port)
        return

    # 检查坐标参数
    if args.x_mm is None or args.y_mm is None:
        parser.print_help()
        print()
        print("[ERROR] 请指定 X, Y 坐标，或使用 --interactive 进入交互模式")
        sys.exit(1)

    # 循环发送模式
    if args.loop:
        loop_mode(args.ip, args.port, args.x_mm, args.y_mm, args.yaw_deg, args.loop)
    else:
        # 单次发送
        send_position(args.ip, args.port, args.x_mm, args.y_mm, args.yaw_deg)


if __name__ == "__main__":
    main()
