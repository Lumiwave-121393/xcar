"""
serial_comm.py —— UART 串口通信与二进制协议封装
================================================

底层串口通信模块，不包含任何车辆控制逻辑。

职责：
  1. 串口连接管理（打开/关闭/状态查询）
  2. 二进制通信帧的构建、校验和、发送
  3. 心跳保活线程
  4. 系统串口扫描

通信协议（与 TC264 cpu0_main.c 对应）:
  AA 55 [cmd 1B] [len 1B] [data NB] [XOR 1B]

硬件连接：
  香橙派 (RK3588)                     TC264 (AURIX)
  ┌─────────────────┐                ┌──────────────────┐
  │  UART TX        │───────────────→│  RX (P14.1)      │
  │  UART RX        │←───────────────│  TX (P14.0)      │
  │  GND            │────────────────│  GND             │
  └─────────────────┘                └──────────────────┘

  香橙派侧串口设备通常为 /dev/ttyS4 或 /dev/ttyAMA*，具体取决于设备树配置。

通信协议（二进制定长帧，每帧 8 字节）：

  ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
  │ 帧头 2B  │ 命令 1B  │ 长度 1B  │ 舵机 2B  │ 电机 2B  │ 校验 1B  │
  │0xAA 0x55 │  0x01    │   0x04   │  int16   │  int16   │  XOR     │
  └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

  字段说明：
  - 帧头:   固定 0xAA 0x55，用于帧同步
  - 命令:   0x01 = 舵机+电机控制, 0x02 = 心跳, 0x03 = 参数配置
  - 长度:   数据区字节数（控制帧固定 4）
  - 舵机值: 有符号 16 位整数，大端，范围 [500, 2500]（对应 PWM 脉宽 us）
  - 电机值: 有符号 16 位整数，大端，范围 [-1000, 1000]（正值前进，负值后退）
  - 校验和: 从命令字节到校验和前一字节的 XOR 异或和
"""
import struct
import serial
import serial.tools.list_ports
import threading
import time
import logging

logger = logging.getLogger("SerialComm")


# ====================================================================
# 协议常量
# ====================================================================

FRAME_HEADER = b'\xAA\x55'       # 帧头
CMD_CONTROL   = 0x01             # 舵机+电机控制
CMD_HEARTBEAT = 0x02             # 心跳保活
CMD_CONFIG          = 0x03       # 参数配置（预留）
CMD_ENCODER_REPORT  = 0x04       # 编码器上行帧（MCU → Host）

# ====================================================================
# 工具函数
# ====================================================================

def list_available_ports():
    """列出系统中所有可用串口"""
    ports = serial.tools.list_ports.comports()
    result = []
    for p in ports:
        result.append({
            'device': p.device,
            'name': p.name,
            'description': p.description,
            'hwid': p.hwid,
        })
    return result


def xor_checksum(data: bytes) -> int:
    """计算 XOR 校验和"""
    result = 0
    for b in data:
        result ^= b
    return result & 0xFF


# ====================================================================
# SerialComm —— 串口通信类
# ====================================================================

class SerialComm:
    """
    UART 串口通信封装

    用法:
        comm = SerialComm()
        comm.open('/dev/ttyUSB0', 115200)
        comm.send_control(1500, 300)   # 舵机直行，电机300
        comm.send_heartbeat()
        comm.close()
    """

    def __init__(self):
        self.ser = None
        self._heartbeat_thread = None
        self._heartbeat_interval = 0.5
        self._running = False

        # 重连相关
        self._port = None
        self._baudrate = None
        self._timeout = None
        self._last_reconnect_time = 0.0
        self._reconnect_cooldown = 2.0  # 重连冷却（秒）

        # ── 上行帧接收 ──
        self._reader_thread = None
        self._reader_running = False

        # 接收状态机: 0=HEADER1, 1=HEADER2, 2=CMD, 3=LEN, 4=DATA, 5=CHK
        self._rx_state = 0
        self._rx_cmd = 0
        self._rx_len = 0
        self._rx_data = bytearray()
        self._rx_idx = 0
        self._rx_calc_xor = 0

        # 编码器数据容器
        self._encoder_data = None            # dict 或 None
        self._encoder_callbacks = []         # list[callable]

    # ================================================================
    # 串口连接管理
    # ================================================================

    def open(self, port: str, baudrate: int = 115200, timeout: float = 0.05) -> bool:
        """打开串口，8N1 格式"""
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout

        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout,
                write_timeout=timeout
            )
            logger.info(f"串口已打开: {port} @ {baudrate} baud")
            self._start_reader()
            return True
        except serial.SerialException as e:
            logger.error(f"无法打开串口 {port}: {e}")
            return False

    def close(self):
        """关闭串口"""
        self._running = False
        self._reader_running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
                logger.info("串口已关闭")
            except Exception:
                pass
        self.ser = None

    def is_connected(self) -> bool:
        """查询串口是否已连接"""
        return self.ser is not None and self.ser.is_open

    # ================================================================
    # 自动重连
    # ================================================================

    def _auto_reconnect(self, force=False) -> bool:
        """
        尝试自动重连 USB 串口。

        策略：
          1. 有冷却时间（默认 2 秒），避免高频重试
          2. 先试原设备号（如 ttyUSB0 → 可能已恢复）
          3. 失败后扫描所有 ttyUSB* 设备（设备号可能已变）

        参数:
            force: 突破冷却时间强制重试

        返回:
            bool: 是否成功连接
        """
        now = time.time()
        if not force and now - self._last_reconnect_time < self._reconnect_cooldown:
            return False
        self._last_reconnect_time = now

        # 关闭旧连接（如有）
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

        # ---- 策略1: 试原设备号 ----
        if self._port:
            logger.info(f"尝试重连原串口: {self._port}")
            try:
                self.ser = serial.Serial(
                    port=self._port,
                    baudrate=self._baudrate or 115200,
                    timeout=self._timeout or 0.05,
                    write_timeout=self._timeout or 0.05,
                )
                if self.ser.is_open:
                    logger.info(f"串口重连成功: {self._port}")
                    return True
            except serial.SerialException:
                self.ser = None

        # ---- 策略2: 扫描所有 ttyUSB* ----
        logger.info("原串口不可用，扫描所有 USB 串口...")
        try:
            candidates = serial.tools.list_ports.comports()
            for p in candidates:
                dev = p.device
                # 优先匹配 USB 串口
                if 'USB' not in dev.upper() and 'ttyUSB' not in dev:
                    continue
                if dev == self._port:
                    continue  # 原设备号已试过，跳过
                logger.info(f"尝试新串口: {dev}")
                try:
                    self.ser = serial.Serial(
                        port=dev,
                        baudrate=self._baudrate or 115200,
                        timeout=self._timeout or 0.05,
                        write_timeout=self._timeout or 0.05,
                    )
                    if self.ser.is_open:
                        self._port = dev  # 更新为新的设备号
                        logger.info(f"串口重连成功(新设备): {self._port}")
                        return True
                except serial.SerialException:
                    if self.ser:
                        try:
                            self.ser.close()
                        except Exception:
                            pass
                        self.ser = None
                        continue
        except Exception as e:
            logger.warning(f"串口扫描异常: {e}")

        logger.warning("串口重连失败，稍后将继续尝试")
        return False

    # ================================================================
    # 帧构建与发送
    # ================================================================

    def _build_frame(self, cmd: int, payload: bytes) -> bytes:
        """构建完整通信帧: 帧头 + 命令 + 长度 + 数据 + 校验"""
        body = bytes([cmd, len(payload)]) + payload
        checksum = xor_checksum(body)
        return FRAME_HEADER + body + bytes([checksum])

    def _send_frame(self, cmd: int, payload: bytes):
        """发送一帧数据（发送失败不抛异常）"""
        if not self.is_connected():
            # 断开状态 → 尝试自动重连
            if self._auto_reconnect():
                logger.info(f"自动重连成功，重新发送指令 cmd=0x{cmd:02x}")
            else:
                logger.warning("串口未连接，指令未发送")
                return

        frame = self._build_frame(cmd, payload)
        try:
            self.ser.write(frame)
        except serial.SerialException as e:
            logger.error(f"串口写入失败: {e}")
            # IO 错误（如设备被拔出）→ 关闭旧连接，触发强制重连
            if "Input/output error" in str(e) or "device" in str(e).lower():
                logger.info("检测到串口 IO 错误，触发自动重连...")
                self.ser = None
                self._auto_reconnect(force=True)

    # ================================================================
    # 公开指令
    # ================================================================

    def send_control(self, servo: int, motor: int):
        """
        发送舵机+电机控制帧

        servo: [500, 2500] us     motor: [-1000, 1000]
        """
        payload = struct.pack('>Hh', servo, motor)
        self._send_frame(CMD_CONTROL, payload)

    def send_heartbeat(self):
        """发送心跳帧（无载荷）"""
        self._send_frame(CMD_HEARTBEAT, b'')

    # ================================================================
    # 心跳线程
    # ================================================================

    def _heartbeat_loop(self):
        while self._running:
            self.send_heartbeat()
            time.sleep(self._heartbeat_interval)

    def start_heartbeat(self):
        """启动心跳发送线程"""
        if self._running:
            return
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="serial-heartbeat"
        )
        self._heartbeat_thread.start()
        logger.info("心跳线程已启动")

    def stop_heartbeat(self):
        """停止心跳线程"""
        self._running = False

    # ================================================================
    # 上行帧接收（MCU → Host）
    # ================================================================

    def _parse_rx_byte(self, byte):
        """
        逐字节解析上行帧，状态机与 TC264 端 protocol_parse_byte() 镜像对称。

        帧结构: AA 55 [cmd 1B] [len 1B] [data NB] [XOR 1B]
        """
        if self._rx_state == 0:            # 等待帧头1 (0xAA)
            if byte == 0xAA:
                self._rx_calc_xor = 0
                self._rx_state = 1
        elif self._rx_state == 1:          # 等待帧头2 (0x55)
            if byte == 0x55:
                self._rx_state = 2
            elif byte != 0xAA:
                self._rx_state = 0
        elif self._rx_state == 2:          # 命令字节
            self._rx_cmd = byte
            self._rx_calc_xor ^= byte
            self._rx_state = 3
        elif self._rx_state == 3:          # 长度字节
            self._rx_len = byte
            self._rx_calc_xor ^= byte
            self._rx_idx = 0
            if self._rx_len == 0:
                self._rx_state = 5
            else:
                self._rx_data = bytearray(self._rx_len)
                self._rx_state = 4
        elif self._rx_state == 4:          # 数据字节
            self._rx_data[self._rx_idx] = byte
            self._rx_calc_xor ^= byte
            self._rx_idx += 1
            if self._rx_idx >= self._rx_len:
                self._rx_state = 5
        elif self._rx_state == 5:          # 校验字节
            if self._rx_calc_xor == byte:
                self._handle_frame(self._rx_cmd, bytes(self._rx_data))
            self._rx_state = 0

    def _handle_frame(self, cmd, data):
        """
        处理完整的上行帧。

        目前仅处理 CMD_ENCODER_REPORT (0x04)，
        后续可扩展其他上行帧类型。
        """
        if cmd == CMD_ENCODER_REPORT:
            if len(data) == 6:
                # 旧版帧（兼容）：仅右轮编码器
                enc_r  = int.from_bytes(data[0:2], byteorder='big', signed=True)
                spd_r  = int.from_bytes(data[2:4], byteorder='big', signed=True)
                tgt    = int.from_bytes(data[4:6], byteorder='big', signed=True)
                self._encoder_data = {
                    "encoder_r": enc_r,  "speed_r": spd_r,
                    "encoder_l": 0,      "speed_l": 0,
                    "target_speed": tgt,
                }
            elif len(data) == 10:
                # 新版帧：左右轮编码器
                enc_r  = int.from_bytes(data[0:2], byteorder='big', signed=True)
                spd_r  = int.from_bytes(data[2:4], byteorder='big', signed=True)
                enc_l  = int.from_bytes(data[4:6], byteorder='big', signed=True)
                spd_l  = int.from_bytes(data[6:8], byteorder='big', signed=True)
                tgt    = int.from_bytes(data[8:10], byteorder='big', signed=True)
                self._encoder_data = {
                    "encoder_r": enc_r,  "speed_r": spd_r,
                    "encoder_l": enc_l,  "speed_l": spd_l,
                    "target_speed": tgt,
                }

            # 通知所有注册的回调
            for cb in self._encoder_callbacks:
                try:
                    cb(self._encoder_data)
                except Exception:
                    pass

    def _reader_loop(self):
        """
        后台线程：持续从串口读取字节，送入状态机解析。

        使用 ser.read(1) 配合 timeout 实现阻塞-唤醒交替，
        超时后自动检查 _reader_running 标志，避免线程无法退出。
        """
        while self._reader_running:
            try:
                if self.ser and self.ser.is_open:
                    b = self.ser.read(1)
                    if b:
                        self._parse_rx_byte(b[0])
                else:
                    time.sleep(0.1)
            except serial.SerialException:
                time.sleep(0.1)
            except Exception:
                time.sleep(0.1)

    def _start_reader(self):
        """启动后台读取线程"""
        if self._reader_running:
            return
        self._reader_running = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="serial-reader"
        )
        self._reader_thread.start()
        logger.info("串口读取线程已启动")

    def get_encoder_data(self):
        """
        返回最新编码器数据。

        返回:
            dict: {"encoder_count": int, "actual_speed": int, "target_speed": int}
            None: 尚未收到任何编码器数据
        """
        return self._encoder_data

    def on_encoder(self, callback):
        """
        注册编码器数据回调。

        参数:
            callback: callable(dict)，收到新数据时调用
        """
        self._encoder_callbacks.append(callback)
