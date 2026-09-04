"""
LLDP neighbors of the host's network interfaces: which switch each port is cabled to.

Two hosts whose ports report the same chassis id hang off the same leaf switch,
which is the closest thing to a rack that a host can learn on its own, without a
fabric scan or a CMDB. Switches advertise LLDP on their own (every 30s by
default), so this only has to listen.

Listening is a raw ``AF_PACKET`` socket on ethertype ``0x88CC``, which needs
``CAP_NET_RAW`` -- the privilege a device detector already runs with. No
``lldpd`` or ``systemd-networkd`` is assumed, because neither is a given on a
bare-metal accelerator host.
"""

from __future__ import annotations as __future_annotations__

import logging
import socket
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import envs
from ..logging import debug_log_exception

logger = logging.getLogger(__name__)

ETH_P_LLDP = 0x88CC

# The LLDP multicast group. A NIC drops multicast it was not asked for unless
# it is promiscuous, so the group has to be joined per interface -- which is
# also why a bare AF_PACKET socket hears nothing while tcpdump does.
LLDP_MULTICAST_MAC = bytes.fromhex("0180c200000e")

_SOL_PACKET = 263
_PACKET_ADD_MEMBERSHIP = 1
_PACKET_MR_MULTICAST = 0

# LLDP switches announce every 30s by default (IEEE 802.1AB msgTxInterval), so
# a listen shorter than that misses ports at random; a little over one interval
# catches every port that is going to speak at all. Overridable through
# GPUSTACK_RUNTIME_DETECT_LLDP_LISTEN_SECONDS for switches on a longer interval.
DEFAULT_LISTEN_SECONDS = 35.0

_ETH_HEADER_LEN = 14
_TLV_END = 0
_TLV_CHASSIS_ID = 1
_TLV_PORT_ID = 2
_TLV_PORT_DESCRIPTION = 4
_TLV_SYSTEM_NAME = 5
_TLV_SYSTEM_DESCRIPTION = 6
_TLV_MGMT_ADDRESS = 8

_CHASSIS_SUBTYPE_MAC = 4
_PORT_SUBTYPE_MAC = 3
_MGMT_ADDR_FAMILY_IPV4 = 1
_MGMT_ADDR_FAMILY_IPV6 = 2


@dataclass
class LLDPNeighbor:
    """
    What the switch at the far end of one interface says about itself.
    """

    interface: str
    """
    The local interface the advertisement arrived on.
    """
    chassis_id: str
    """
    The switch's chassis id, a MAC in ``aa:bb:cc:dd:ee:ff`` form when the switch
    identifies by MAC (they nearly all do), otherwise the string it sent.
    """
    port_id: str = ""
    """
    The switch port, as the switch names it (``200GE1/0/9``, ``8`` ...).
    """
    port_description: str = ""
    system_name: str = ""
    system_description: str = ""
    management_addresses: list[str] = field(default_factory=list)


def parse_lldp_frame(frame: bytes, interface: str = "") -> LLDPNeighbor | None:
    """
    Parse one LLDP Ethernet frame into a neighbor.

    Args:
        frame:
            The raw frame, Ethernet header included.
        interface:
            The local interface it was captured on.

    Returns:
        The neighbor, or None if the frame is not a parseable LLDPDU.

    """
    if len(frame) < _ETH_HEADER_LEN + 2:
        return None
    if struct.unpack("!H", frame[12:14])[0] != ETH_P_LLDP:
        return None

    chassis_id: str | None = None
    port_id = ""
    port_description = ""
    system_name = ""
    system_description = ""
    mgmt: list[str] = []

    offset = _ETH_HEADER_LEN
    while offset + 2 <= len(frame):
        header = struct.unpack("!H", frame[offset : offset + 2])[0]
        tlv_type, tlv_len = header >> 9, header & 0x1FF
        value = frame[offset + 2 : offset + 2 + tlv_len]
        offset += 2 + tlv_len
        if tlv_type == _TLV_END:
            break
        if len(value) != tlv_len:
            break
        if tlv_type == _TLV_CHASSIS_ID and value:
            chassis_id = _identity(value[0], value[1:], _CHASSIS_SUBTYPE_MAC)
        elif tlv_type == _TLV_PORT_ID and value:
            port_id = _identity(value[0], value[1:], _PORT_SUBTYPE_MAC)
        elif tlv_type == _TLV_PORT_DESCRIPTION:
            port_description = _text(value)
        elif tlv_type == _TLV_SYSTEM_NAME:
            system_name = _text(value)
        elif tlv_type == _TLV_SYSTEM_DESCRIPTION:
            system_description = _text(value)
        elif tlv_type == _TLV_MGMT_ADDRESS:
            address = _management_address(value)
            if address:
                mgmt.append(address)

    if chassis_id is None:
        return None

    return LLDPNeighbor(
        interface=interface,
        chassis_id=chassis_id,
        port_id=port_id,
        port_description=port_description,
        system_name=system_name,
        system_description=system_description,
        management_addresses=mgmt,
    )


def _identity(subtype: int, data: bytes, mac_subtype: int) -> str:
    if subtype == mac_subtype and len(data) == 6:
        return ":".join(f"{b:02x}" for b in data)
    return _text(data)


def _text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").strip("\x00").strip()


def _management_address(value: bytes) -> str | None:
    # [addr string length][addr subtype][addr bytes ...][iface subtype][iface number][oid len][oid]
    if len(value) < 2:
        return None
    addr_len = value[0]
    family = value[1]
    addr = value[2 : 1 + addr_len]
    if family == _MGMT_ADDR_FAMILY_IPV4 and len(addr) == 4:
        return socket.inet_ntop(socket.AF_INET, bytes(addr))
    if family == _MGMT_ADDR_FAMILY_IPV6 and len(addr) == 16:
        return socket.inet_ntop(socket.AF_INET6, bytes(addr))
    return None


def physical_interfaces() -> list[str]:
    """
    The host's physical network interfaces: those backed by a device.

    Bridges, veths, bonds and VLANs have no ``device`` link in sysfs and are
    left out -- a switch is cabled to a port, and only ports hear it. Bond
    members are included as themselves, which is what LLDP reports anyway.

    Returns:
        Interface names, sorted.

    """
    root = Path("/sys/class/net")
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "device").exists())


def _join_lldp_group(sock: socket.socket, interface: str) -> None:
    try:
        index = socket.if_nametoindex(interface)
        sock.setsockopt(
            _SOL_PACKET,
            _PACKET_ADD_MEMBERSHIP,
            struct.pack(
                "IHH8s",
                index,
                _PACKET_MR_MULTICAST,
                len(LLDP_MULTICAST_MAC),
                LLDP_MULTICAST_MAC,
            ),
        )
    except OSError:
        debug_log_exception(logger, "Failed to join the LLDP group on %s", interface)


def detect_lldp_neighbors(
    interfaces: list[str] | None = None,
    listen_seconds: float | None = None,
) -> list[LLDPNeighbor]:
    """
    Listen for LLDP advertisements and report one neighbor per interface.

    Blocks for up to ``listen_seconds``, or until every interface in
    ``interfaces`` has reported when that list is given. Callers that cannot
    afford the wait should run it off the hot path and cache: the answer only
    changes when someone recables the host.

    Args:
        interfaces:
            The interfaces to wait for, or None for every physical interface.
        listen_seconds:
            How long to listen for. None takes
            ``GPUSTACK_RUNTIME_DETECT_LLDP_LISTEN_SECONDS`` (35s by default,
            one LLDP interval).

    Returns:
        The neighbors heard, in arrival order; empty when nothing was heard or
        the socket could not be opened (no ``CAP_NET_RAW``, no ``AF_PACKET``).

    """
    if listen_seconds is None:
        listen_seconds = (
            envs.GPUSTACK_RUNTIME_DETECT_LLDP_LISTEN_SECONDS or DEFAULT_LISTEN_SECONDS
        )
    wanted = set(interfaces) if interfaces else None
    heard: dict[str, LLDPNeighbor] = {}

    try:
        sock = socket.socket(
            socket.AF_PACKET,
            socket.SOCK_RAW,
            socket.htons(ETH_P_LLDP),
        )
    except (OSError, AttributeError):
        debug_log_exception(logger, "Failed to open an LLDP listening socket")
        return []

    for interface in interfaces or physical_interfaces():
        _join_lldp_group(sock, interface)

    deadline = time.monotonic() + listen_seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if wanted is not None and wanted <= heard.keys():
                break
            sock.settimeout(remaining)
            try:
                frame, address = sock.recvfrom(65535)
            except TimeoutError:
                break
            except OSError:
                debug_log_exception(logger, "Failed to receive an LLDP frame")
                break
            interface = address[0]
            if wanted is not None and interface not in wanted:
                continue
            if interface in heard:
                continue
            neighbor = parse_lldp_frame(frame, interface)
            if neighbor is not None:
                heard[interface] = neighbor
    finally:
        sock.close()

    return list(heard.values())
