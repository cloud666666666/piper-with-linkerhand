#!/usr/bin/env python3
"""
灵心巧手（LinkerHand O6）连接探测脚本。

O6 手支持两种通信协议，本脚本对两者都做一次自动探测，并输出
config.yaml 中 linker_hand_modbus / linker_hand_can 应该填的值：

1. Modbus RTU 串口：115200, 8N1，功能码 04/16，
   左手站号 0x28(40)、右手站号 0x27(39)，需 USB-485 转接器；
2. CAN：1Mbps，CANID 0x27(R)/0x28(L)，首字节为指令码
   （读指令只发 1 字节、回复 7 字节），需 USB-CAN 盒。

不依赖本仓库 config，独立运行：
    .venv/bin/python probe_linker_hand.py
"""

import glob
import subprocess
import sys
import time

# 手可能用到的 CAN 读指令码（0xC2 软件版本 / 0xC1 硬件版本 / 0xC0 出厂编码）
REPLY_CMDS = (0xC2, 0xC1, 0xC0, 0x64)
# 左手 0x28、右手 0x27
HAND_IDS = (0x28, 0x27)


def list_serial_ports():
    """列出 USB 串口设备（不含板载 /dev/ttyS* UART，那些不是转接器）。"""
    ports = set()
    for pat in ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyCH*"):
        ports.update(glob.glob(pat))
    return sorted(ports)


def probe_modbus(port: str):
    """在单个串口上尝试 Modbus RTU 查询（左右手两个站号）。"""
    from pymodbus.client import ModbusSerialClient

    for hid in HAND_IDS:
        cli = ModbusSerialClient(
            port=port,
            baudrate=115200,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0.25,
        )
        try:
            if not cli.connect():
                continue
            # 读输入寄存器 30 起 15 个（设备编号/版本块，功能码 04）
            rsp = cli.read_input_registers(address=30, count=15, slave=hid)
            if not rsp.isError():
                print(f"  ✅ 站号 {hid} (0x{hid:02x}) 应答: {list(rsp.registers)}")
                return hid
        except Exception as e:
            print(f"  ⚠️ {port} 站号 {hid} 读失败: {e}")
        finally:
            cli.close()
    return None


def list_can_ifaces():
    """列出所有 CAN 接口名。"""
    out = subprocess.run(["ip", "-o", "link", "show"], capture_output=True, text=True).stdout
    ifaces = []
    for line in out.splitlines():
        parts = line.split(": ")
        if len(parts) >= 2 and parts[1].startswith("can"):
            ifaces.append(parts[1])
    return ifaces


def probe_can(iface: str, reset: bool = True):
    """在单个 CAN 接口上发 O6 协议读指令并等应答。

    注意：必须关掉 socketcan 本地回显（receive_own_messages=False），
    否则自己发出的 1 字节查询帧会被当成手应答（假阳性）。
    手应答帧长度 ≥3（指令码 + 版本/状态数据），回显帧长度 =1。
    """
    import can

    if reset:
        # 板载口可能处于 ERROR-PASSIVE，先重置一下再探测
        subprocess.run(
            ["sudo", "-n", "ip", "link", "set", iface, "down"],
            capture_output=True,
        )
        subprocess.run(
            ["sudo", "-n", "ip", "link", "set", iface, "up", "type", "can", "bitrate", "1000000"],
            capture_output=True,
        )
        time.sleep(0.3)
    try:
        bus = can.interface.Bus(
            channel=iface, interface="socketcan", receive_own_messages=False
        )
    except Exception as e:
        print(f"  ⚠️ 打开 {iface} 失败: {e}")
        return None
    try:
        for hid in HAND_IDS:
            for cmd in (0xC2, 0xC1):
                try:
                    bus.send(can.Message(arbitration_id=hid, data=[cmd]))
                except can.CanError:
                    break
                deadline = time.time() + 0.4
                while time.time() < deadline:
                    rx = bus.recv(timeout=0.1)
                    # 应答帧须带数据且长度 ≥3（如 0xC2 + 3 字节版本号）
                    if (
                        rx is not None
                        and rx.data
                        and len(rx.data) >= 3
                        and rx.data[0] in REPLY_CMDS
                    ):
                        print(
                            f"  ✅ id={hex(rx.arbitration_id)} 应答 "
                            f"dlc={len(rx.data)} data={[hex(b) for b in rx.data]}"
                        )
                        return rx.arbitration_id
    finally:
        bus.shutdown()
    return None


def watch():
    """持续监视：每 2 秒探测一次串口和 CAN，发现手立即返回。

    CAN 接口已处于 UP 的不做重置（避免干扰正在工作的 Piper 臂），
    只发 2 帧只读查询。
    """
    import can  # noqa: F401

    print("👀 进入持续探测模式（Ctrl+C 退出）……")
    print("   现在请插转接器/切协议开关，这边会自动发现。")
    try:
        while True:
            for port in list_serial_ports():
                hid = probe_modbus(port)
                if hid is not None:
                    hand = "left" if hid == 0x28 else "right"
                    print(f"\n🎉 发现 Modbus 手（{hand}）。config.yaml 应设：")
                    print(f"   linker_hand_type: {hand}")
                    print(f"   linker_hand_modbus: {port}")
                    return 0
            for iface in list_can_ifaces():
                hid = probe_can(iface, reset=False)
                if hid is not None:
                    hand = "left" if hid == 0x28 else "right"
                    print(f"\n🎉 发现 CAN 手（{hand}）。config.yaml 应设：")
                    print(f"   linker_hand_type: {hand}")
                    print(f"   linker_hand_modbus: None")
                    print(f"   linker_hand_can: {iface}")
                    return 0
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n已退出持续探测。")
        return 1


def main():
    if "--watch" in sys.argv:
        return watch()

    print("=" * 55)
    print("  灵心巧手（O6）连接探测：Modbus RTU + CAN 双模")
    print("=" * 55)

    found = False

    # ── 1. Modbus RTU 串口 ──
    print("\n[1/2] Modbus RTU 串口探测（115200 8N1）...")
    ports = list_serial_ports()
    if not ports:
        print("  ❌ 未发现任何串口设备（/dev/ttyUSB* 等）。")
        print("     → 如果手走 Modbus，请把 USB-485 转接器插到 Jetson 再跑一次。")
    else:
        print(f"  发现串口: {', '.join(ports)}")
        for port in ports:
            print(f"  探测 {port} ...")
            hid = probe_modbus(port)
            if hid is not None:
                hand = "left" if hid == 0x28 else "right"
                print(f"  🎉 找到 Modbus 手（{hand}）。config.yaml 应设：")
                print(f"     linker_hand_type: {hand}")
                print(f"     linker_hand_modbus: {port}")
                found = True
                break

    # ── 2. CAN 总线 ──
    print("\n[2/2] CAN 总线探测（1Mbps，O6 协议）...")
    ifaces = list_can_ifaces()
    if not ifaces:
        print("  ❌ 未发现任何 CAN 接口。")
        print("     → 如果手走 CAN，请把 USB-CAN 盒插到 Jetson 再跑一次。")
    else:
        print(f"  发现 CAN 接口: {', '.join(ifaces)}")
        for iface in ifaces:
            print(f"  探测 {iface} ...")
            hid = probe_can(iface)
            if hid is not None:
                hand = "left" if hid == 0x28 else "right"
                print(f"  🎉 找到 CAN 手（{hand}）。config.yaml 应设：")
                print(f"     linker_hand_type: {hand}")
                print(f"     linker_hand_modbus: None")
                print(f"     linker_hand_can: {iface}")
                found = True
                break

    # ── 汇总 ──
    print("\n" + "=" * 55)
    if found:
        print("  探测成功：修改 config.yaml 后运行 test_linker_hand.py")
    else:
        print("  未找到手。请检查：")
        print("  1. 手部 24V 电源是否上电")
        print("  2. 手的通信线是否接到 Jetson：")
        print("     Modbus 模式 → USB-485 转接器插 Jetson USB")
        print("     CAN 模式   → USB-CAN 盒插 Jetson USB")
        print("  3. 插上后重新运行本脚本")
    print("=" * 55)
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
