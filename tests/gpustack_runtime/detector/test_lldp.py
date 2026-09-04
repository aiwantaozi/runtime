from __future__ import annotations

import socket
import struct

import pytest

from gpustack_runtime.detector import LLDPNeighbor, detect_lldp_neighbors
from gpustack_runtime.detector import lldp as lldp_mod
from gpustack_runtime.detector.lldp import ETH_P_LLDP, parse_lldp_frame

# --------------------------------------------------------------------------- #
# A frame as a switch actually sends it: captured off a TP-Link TL-SE2109      #
# (systemd-networkd's /run/systemd/netif/lldp cache, header stripped).        #
# --------------------------------------------------------------------------- #

_TPLINK_FRAME = bytes.fromhex(
    "0180c200000e"  # dst: LLDP multicast
    "7439891c70a0"  # src
    "88cc"
    "0207 04 7439891c70a0"  # chassis id, subtype MAC
    "0402 02 38"  # port id, subtype interface alias, "8"
    "0602 0078"  # ttl 120
    "0a09 544c2d534532313039"  # system name "TL-SE2109"
    "100c 05 01 c0a80001 02 00000001 00"  # mgmt addr 192.168.0.1
    "0e04 00040004"  # capabilities
    "0000",  # end
)


def _tlv(tlv_type: int, value: bytes) -> bytes:
    return struct.pack("!H", (tlv_type << 9) | len(value)) + value


def _frame(*tlvs: bytes) -> bytes:
    return (
        b"\x01\x80\xc2\x00\x00\x0e"
        b"\xc0\xf9\xb0\xc7\x13\x71"
        + struct.pack("!H", ETH_P_LLDP)
        + b"".join(tlvs)
        + _tlv(0, b"")
    )


def test_parses_a_switch_frame():
    neighbor = parse_lldp_frame(_TPLINK_FRAME, "eno1")

    assert neighbor == LLDPNeighbor(
        interface="eno1",
        chassis_id="74:39:89:1c:70:a0",
        port_id="8",
        system_name="TL-SE2109",
        management_addresses=["192.168.0.1"],
    )


def test_parses_the_tlvs_a_datacenter_switch_sends():
    # The shape hccn_tool showed on a Huawei CE8875: named port, a port
    # description carrying the far-end location tag, IPv4 and IPv6 management
    # addresses.
    frame = _frame(
        _tlv(1, b"\x04" + bytes.fromhex("c0f9b0c71371")),
        _tlv(2, b"\x05" + b"200GE1/0/9"),
        _tlv(3, struct.pack("!H", 120)),
        _tlv(4, b"dT:[ZS2F1-SPOD1-OS07]-FLEXIOB1-Port1"),
        _tlv(5, b"POD1-S-JR-ROCE-CE8875-50"),
        _tlv(6, b"Huawei Versatile Routing Platform Software\nHUAWEI CE8875-24BQ8DQ\n"),
        _tlv(
            8,
            b"\x05\x01"
            + socket.inet_aton("10.12.188.204")
            + b"\x02"
            + struct.pack("!I", 4)
            + b"\x00",
        ),
        _tlv(
            8,
            b"\x11\x02"
            + socket.inet_pton(socket.AF_INET6, "2409:808f:5fbc:1::a0c:bccc")
            + b"\x02"
            + struct.pack("!I", 4)
            + b"\x00",
        ),
    )

    neighbor = parse_lldp_frame(frame, "enp67s0f0np0")

    assert neighbor is not None
    assert neighbor.chassis_id == "c0:f9:b0:c7:13:71"
    assert neighbor.port_id == "200GE1/0/9"
    assert neighbor.port_description == "dT:[ZS2F1-SPOD1-OS07]-FLEXIOB1-Port1"
    assert neighbor.system_name == "POD1-S-JR-ROCE-CE8875-50"
    assert neighbor.system_description.startswith("Huawei Versatile")
    assert neighbor.management_addresses == [
        "10.12.188.204",
        "2409:808f:5fbc:1::a0c:bccc",
    ]


def test_keeps_a_non_mac_chassis_id_as_text():
    frame = _frame(_tlv(1, b"\x07leaf-sw-3"), _tlv(2, b"\x07" + b"Ethernet1/1"))

    neighbor = parse_lldp_frame(frame)

    assert neighbor is not None
    assert neighbor.chassis_id == "leaf-sw-3"
    assert neighbor.port_id == "Ethernet1/1"


def test_rejects_a_frame_of_another_ethertype():
    frame = bytearray(_TPLINK_FRAME)
    frame[12:14] = b"\x08\x00"

    assert parse_lldp_frame(bytes(frame)) is None


def test_rejects_a_frame_without_a_chassis_id():
    assert parse_lldp_frame(_frame(_tlv(5, b"nameless"))) is None


def test_survives_a_truncated_tlv():
    frame = (
        _frame(_tlv(1, b"\x04" + bytes.fromhex("c0f9b0c71371")))[:-2]
        + struct.pack("!H", (5 << 9) | 40)
        + b"short"
    )

    neighbor = parse_lldp_frame(frame)

    assert neighbor is not None
    assert neighbor.chassis_id == "c0:f9:b0:c7:13:71"
    assert neighbor.system_name == ""


# --------------------------------------------------------------------------- #
# detect_lldp_neighbors: one neighbor per interface, stop once all are heard.  #
# --------------------------------------------------------------------------- #


class _FakeSocket:
    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False

    def settimeout(self, _seconds):
        pass

    def setsockopt(self, *_args):
        pass

    def recvfrom(self, _size):
        if not self.frames:
            raise TimeoutError
        interface, frame = self.frames.pop(0)
        return frame, (interface, ETH_P_LLDP, 0, 1, b"")

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _no_host_interfaces(monkeypatch):
    monkeypatch.setattr(lldp_mod, "physical_interfaces", list)


def test_detect_reports_one_neighbor_per_interface(monkeypatch):
    sock = _FakeSocket(
        [
            ("eno1", _TPLINK_FRAME),
            ("eno1", _TPLINK_FRAME),
            ("eno2", _frame(_tlv(1, b"\x04" + bytes.fromhex("2087ec3bd871")))),
        ],
    )
    monkeypatch.setattr(lldp_mod.socket, "socket", lambda *_args: sock)

    neighbors = detect_lldp_neighbors(listen_seconds=5)

    assert [n.interface for n in neighbors] == ["eno1", "eno2"]
    assert sock.closed


def test_detect_stops_once_every_wanted_interface_has_spoken(monkeypatch):
    sock = _FakeSocket(
        [
            ("eno2", _TPLINK_FRAME),
            ("eno1", _TPLINK_FRAME),
            ("eno3", _TPLINK_FRAME),
        ],
    )
    monkeypatch.setattr(lldp_mod.socket, "socket", lambda *_args: sock)

    neighbors = detect_lldp_neighbors(interfaces=["eno1"], listen_seconds=5)

    assert [n.interface for n in neighbors] == ["eno1"]
    # eno3 was never read: the listen ended when eno1 had reported.
    assert [i for i, _ in sock.frames] == ["eno3"]


def test_detect_returns_nothing_without_a_raw_socket(monkeypatch):
    def _refuse(*_args):
        raise PermissionError

    monkeypatch.setattr(lldp_mod.socket, "socket", _refuse)

    assert detect_lldp_neighbors(listen_seconds=1) == []
