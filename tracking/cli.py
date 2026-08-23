"""
cli.py —— 命令行入口

提供三种运行模式：
  1. 摄像头模式（独立调试）
  2. 共享内存模式（与 setup_webui 配合）
  3. 单图测试
"""

import cv2
import time
import struct
import logging
import numpy as np
from multiprocessing import shared_memory, resource_tracker

from .tracker import SegmentTracker
from .config import SHM_NAME, SHM_HEADER_SIZE
from .viz import draw_control_hud

logger = logging.getLogger("tracking.cli")

# 车辆控制（可选）
try:
    from vehicle_control import VehicleController
    _has_control = True
except ImportError:
    _has_control = False


# ====================================================================
# 共享内存辅助
# ====================================================================

def _remove_shm_tracker():
    try:
        resource_tracker.unregister('/' + SHM_NAME, 'shared_memory')
    except Exception:
        pass


def _read_shm_frame_consistent(shm):
    """
    从共享内存中原子性地读取一帧，检测并跳过撕裂帧。

    采用"读 header → 读 image → 重读 header"的轻量一致性检查：
    如果两次 header 不同，说明写入端在读取期间覆写了数据，跳过该帧。

    返回:
        (fid, frame_bgr, w, h)  — 成功
        (None, None, 0, 0)      — 帧撕裂或数据无效，应跳过
    """
    try:
        # 1. 第一次读 header
        header1 = bytes(shm.buf[:SHM_HEADER_SIZE])
        fid1, w, h = struct.unpack('QII', header1)

        # 基本合法性检查
        if w <= 0 or h <= 0 or w > 4096 or h > 4096:
            return None, None, 0, 0

        # 2. 读图像数据
        size = w * h * 3
        view = np.ndarray(
            (h, w, 3), dtype=np.uint8,
            buffer=shm.buf[SHM_HEADER_SIZE:SHM_HEADER_SIZE + size],
        )
        frame = view.copy()
        del view

        # 3. 重读 header，检测写入端是否在读取期间覆写了数据
        header2 = bytes(shm.buf[:SHM_HEADER_SIZE])
        fid2, w2, h2 = struct.unpack('QII', header2)

        if fid1 != fid2 or w != w2 or h != h2:
            # 帧撕裂：header 在两次读取之间变化了
            return None, None, 0, 0

        return fid1, frame, w, h

    except (ValueError, struct.error, BufferError):
        return None, None, 0, 0


# ====================================================================
# 共享内存模式
# ====================================================================

def run_shm_client(model_dir=None, TPEs=2, hand="left"):
    """
    共享内存客户端模式。

    从 setup_webui 写入的共享内存段中读取视频帧，
    运行语义分割循迹，可选择同时输出串口控制。
    """
    tracker = SegmentTracker(model_dir=model_dir, TPEs=TPEs, hand=hand)

    controller = None
    if _has_control:
        try:
            controller = VehicleController()
            controller.open()
            controller.start_heartbeat()
            print("[OK] 车辆控制器已连接")
        except Exception as e:
            print(f"[WARN] 车辆控制器不可用: {e}")

    print("[等待] 语义分割循迹客户端就绪，等待服务端...")

    while True:
        shm = None
        try:
            try:
                shm = shared_memory.SharedMemory(name=SHM_NAME)
                _remove_shm_tracker()
                print("[OK] 已连接共享内存")
            except FileNotFoundError:
                time.sleep(1.0)
                continue

            last_fid = 0
            last_t = time.time()   # D 项真实 dt 用（上一帧处理时刻）
            fps_t = time.time()
            fps_n = 0
            cur_fps = 0.0
            # 诊断统计
            diag_t = time.time()
            diag_frames = 0
            diag_tears = 0
            diag_skips = 0

            while True:
                try:
                    # ---- 一致性读取（读-验证-重读） ----
                    fid, frame_raw, w, h = _read_shm_frame_consistent(shm)

                    if fid is None:
                        # 帧撕裂或数据无效
                        diag_tears += 1
                        time.sleep(0.001)
                        if cv2.waitKey(1) == 27:
                            raise KeyboardInterrupt
                        continue

                    if fid == last_fid:
                        diag_skips += 1
                        time.sleep(0.002)
                        if cv2.waitKey(1) == 27:
                            raise KeyboardInterrupt
                        continue

                    last_fid = fid

                    frame = cv2.flip(frame_raw, 0)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # 循迹（含诊断计时）
                    t_proc = time.perf_counter()
                    qdepth = tracker._pool.queue.qsize()
                    offset, viz = tracker.process_frame(frame)
                    t_proc = 1000 * (time.perf_counter() - t_proc)

                    # 控制（offset_to_servo 同时取 PID 分解供 HUD 显示）
                    # D 项真实 dt（2026-08-16）：每秒差分，帧率波动不漂参数
                    now = time.time()
                    dt = min(max(now - last_t, 0.005), 0.2)
                    last_t = now
                    details = tracker.offset_to_servo(offset, dt=dt, return_details=True)
                    servo = details["servo"]
                    speed = tracker.compute_speed(offset)
                    if controller is not None:
                        controller.send_control(servo, speed)

                    # FPS
                    fps_n += 1
                    diag_frames += 1
                    if time.time() - fps_t >= 1.0:
                        cur_fps = fps_n / (time.time() - fps_t)
                        fps_n, fps_t = 0, time.time()

                    # 诊断输出（每秒一次）
                    if time.time() - diag_t >= 1.0:
                        diag_elapsed = time.time() - diag_t
                        logger.info(
                            "fid=%d fps=%.1f proc=%.0fms q=%d tears=%d skips=%d",
                            fid, diag_frames / diag_elapsed if diag_elapsed > 0 else 0,
                            t_proc, qdepth, diag_tears, diag_skips,
                        )
                        diag_t = time.time()
                        diag_frames = 0
                        diag_tears = 0
                        diag_skips = 0

                    cv2.putText(viz, f"FPS: {cur_fps:.1f}",
                                (15, viz.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # 舵机打角 + PID 分解 HUD（2026-08-16）
                    draw_control_hud(viz, servo, details, speed=speed)
                    cv2.imshow("Semantic Tracking", viz)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        raise KeyboardInterrupt
                    elif key == ord('r'):
                        tracker.reset()
                        print("[重置] 帧间状态已清零")

                except (ValueError, struct.error, BufferError):
                    raise FileNotFoundError

        except KeyboardInterrupt:
            print("\n[退出] 语义分割循迹客户端关闭")
            break
        except FileNotFoundError:
            print("[等待] 连接丢失，尝试重连...")
            if shm:
                shm.close()
            cv2.destroyAllWindows()
            time.sleep(1.0)
        finally:
            if shm:
                try:
                    shm.close()
                except Exception:
                    pass

    cv2.destroyAllWindows()
    if controller:
        controller.close()
    tracker.release()


# ====================================================================
# 摄像头模式
# ====================================================================

def run_camera_mode(device_id, width, height, model_dir=None, TPEs=1, hand="left"):
    """摄像头直连模式（独立调试）"""
    tracker = SegmentTracker(model_dir=model_dir, TPEs=TPEs, hand=hand)

    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {device_id}")
        return

    print(f"[OK] 摄像头 {device_id} ({width}x{height})，ESC退出 r键重置")

    controller = None
    if _has_control:
        try:
            controller = VehicleController()
            controller.open()
            controller.start_heartbeat()
            print("[OK] 车辆控制器已连接")
        except Exception as e:
            print(f"[WARN] 车辆控制器不可用: {e}")

    last_t = time.time()   # D 项真实 dt 用（上一帧处理时刻）
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[错误] 读取视频帧失败")
            break

        offset, viz = tracker.process_frame(frame)

        now = time.time()
        dt = min(max(now - last_t, 0.005), 0.2)
        last_t = now
        details = tracker.offset_to_servo(offset, dt=dt, return_details=True)
        servo = details["servo"]
        speed = tracker.compute_speed(offset)
        if controller is not None:
            controller.send_control(servo, speed)
        draw_control_hud(viz, servo, details, speed=speed)

        cv2.imshow("Semantic Tracking", viz)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            tracker.reset()
            print("[重置] 帧间状态已清零")

    cap.release()
    cv2.destroyAllWindows()
    if controller:
        controller.close()
    tracker.release()


# ====================================================================
# 主入口
# ====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic Tracking 语义分割循迹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m tracking -c 0                 # 摄像头模式\n"
            "  python visual_tracking.py -c 0           # 同上（兼容）\n"
            "  python visual_tracking.py --shm          # 共享内存模式\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('-c', '--camera', type=int, default=None, nargs='?', const=0,
                      help='摄像头设备号 (默认: 0)')
    mode.add_argument('--shm', action='store_true',
                      help='共享内存客户端模式 (与 setup_webui 配合)')
    mode.add_argument('-i', '--image', type=str, default=None,
                      help='单张图片测试')

    parser.add_argument('--model', type=str, default=None,
                        help='模型目录路径（内含 .rknn 文件）')
    parser.add_argument('--tpes', type=int, default=1,
                        help='推理线程数 (默认: 1)')
    parser.add_argument('--hand', type=str, choices=['left', 'right'],
                        default='left',
                        help='循迹手性: left=沿左边线(默认) right=沿右边线(预留)')
    parser.add_argument('--width', type=int, default=640,
                        help='图像宽度')
    parser.add_argument('--height', type=int, default=480,
                        help='图像高度')

    args = parser.parse_args()

    try:
        if args.shm:
            run_shm_client(model_dir=args.model, TPEs=args.tpes, hand=args.hand)
        elif args.image:
            tracker = SegmentTracker(model_dir=args.model, TPEs=1, hand=args.hand)
            img = cv2.imread(args.image)
            if img is None:
                print(f"[错误] 无法读取图片: {args.image}")
                return
            offset, viz = tracker.process_frame(img)
            details = tracker.offset_to_servo(offset, return_details=True)
            draw_control_hud(viz, details["servo"], details)
            print(f"Offset: {offset:+.3f}  Servo: {details['servo']}us "
                  f"P:{details['p']:+.1f} D:{details['d']:+.1f} FF:{details['ff']:+.1f}")
            cv2.imshow("Semantic Tracking", viz)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            tracker.release()
        else:
            cam_id = args.camera if args.camera is not None else 0
            run_camera_mode(cam_id, args.width, args.height,
                            model_dir=args.model, TPEs=args.tpes, hand=args.hand)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        print("请指定正确的模型路径，例如:")
        print("  python visual_tracking.py -c 0 --model seg_python/model")
    except KeyboardInterrupt:
        print("\n[退出] 用户中断")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    main()
