from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from scapy.all import ICMP, IP, TCP, UDP, sniff

from .agent_service import build_prediction_report
from .model_service import predict_records


SERVICE_PORT_MAP = {
    20: "ftp_data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "domain_u",
    80: "http",
    110: "pop_3",
    123: "ntp_u",
    143: "imap4",
    179: "bgp",
    443: "http_443",
}


def format_live_capture_error(exc: Exception) -> str:
    message = str(exc).strip()
    if isinstance(exc, PermissionError) or (
        isinstance(exc, OSError) and getattr(exc, "errno", None) == 1
    ) or "Operation not permitted" in message:
        return (
            "Live capture failed: packet capture permission denied. "
            "Run the backend as Administrator/root and ensure capture drivers are installed "
            "(Windows: Npcap; Linux/macOS: sudo or grant capture capabilities)."
        )
    if not message:
        message = exc.__class__.__name__
    return f"Live capture failed: {message}"


def _tcp_flag_to_kdd_flag(packet) -> str:
    if TCP not in packet:
        return "SF"
    flags = int(packet[TCP].flags)
    if flags & 0x04:
        return "REJ"
    if flags & 0x02 and not flags & 0x10:
        return "S0"
    if flags & 0x02 and flags & 0x10:
        return "SF"
    if flags & 0x01:
        return "SH"
    return "SF"


def _service_name(packet) -> str:
    port = None
    if TCP in packet:
        port = int(packet[TCP].dport)
    elif UDP in packet:
        port = int(packet[UDP].dport)
    return SERVICE_PORT_MAP.get(port, "private")


def _protocol_name(packet) -> str:
    if TCP in packet:
        return "tcp"
    if UDP in packet:
        return "udp"
    if ICMP in packet:
        return "icmp"
    return "tcp"


def capture_live_window(interface: str | None = None, packet_limit: int = 30, timeout: int = 10) -> Dict[str, Any]:
    packets = sniff(iface=interface or None, count=packet_limit, timeout=timeout, store=True)

    packet_count = len(packets)
    bytes_seen = sum(len(packet) for packet in packets)
    syn_packets = 0
    tcp_packets = 0
    source_counter: Counter[str] = Counter()
    destination_counter: Counter[str] = Counter()
    service_counter: Counter[str] = Counter()
    flag_counter: Counter[str] = Counter()
    protocol_counter: Counter[str] = Counter()
    wrong_fragments = 0
    urgent_packets = 0
    src_bytes = 0
    dst_bytes = 0
    same_service_packets = 0
    different_service_packets = 0
    tcp_rejects = 0
    host_service_counts: Counter[tuple[str, str]] = Counter()
    source_port_counter: Counter[int] = Counter()

    for packet in packets:
        if IP in packet:
            source_counter[packet[IP].src] += 1
            destination_counter[packet[IP].dst] += 1
            src_bytes += len(packet)
            dst_bytes += len(packet)
            wrong_fragments += int(getattr(packet[IP], "frag", 0) != 0)
        if TCP in packet:
            tcp_packets += 1
            flags = packet[TCP].flags
            if int(flags) & 0x02:
                syn_packets += 1
            if int(flags) & 0x04:
                tcp_rejects += 1
            if int(flags) & 0x20:
                urgent_packets += 1
            source_port_counter[int(packet[TCP].sport)] += 1
        if UDP in packet:
            source_port_counter[int(packet[UDP].sport)] += 1

        protocol = _protocol_name(packet)
        service = _service_name(packet)
        flag = _tcp_flag_to_kdd_flag(packet)
        protocol_counter[protocol] += 1
        service_counter[service] += 1
        flag_counter[flag] += 1
        if IP in packet:
            host_service_counts[(packet[IP].dst, service)] += 1

    top_source = source_counter.most_common(1)[0][0] if source_counter else None
    top_destination = destination_counter.most_common(1)[0][0] if destination_counter else None
    syn_ratio = (syn_packets / tcp_packets) if tcp_packets else 0.0
    unique_destinations = len(destination_counter)
    dominant_protocol = protocol_counter.most_common(1)[0][0] if protocol_counter else "tcp"
    dominant_service = service_counter.most_common(1)[0][0] if service_counter else "private"
    dominant_flag = flag_counter.most_common(1)[0][0] if flag_counter else "SF"
    same_service_packets = service_counter[dominant_service]
    different_service_packets = max(packet_count - same_service_packets, 0)

    serror_rate = syn_ratio
    srv_serror_rate = syn_ratio if same_service_packets else 0.0
    rerror_rate = (tcp_rejects / tcp_packets) if tcp_packets else 0.0
    srv_rerror_rate = rerror_rate if same_service_packets else 0.0
    same_srv_rate = (same_service_packets / packet_count) if packet_count else 0.0
    diff_srv_rate = (different_service_packets / packet_count) if packet_count else 0.0
    srv_diff_host_rate = (unique_destinations / packet_count) if packet_count else 0.0
    dst_host_count = destination_counter[top_destination] if top_destination else 0
    dst_host_srv_count = host_service_counts[(top_destination, dominant_service)] if top_destination else 0
    dst_host_same_srv_rate = (dst_host_srv_count / dst_host_count) if dst_host_count else 0.0
    dst_host_diff_srv_rate = 1.0 - dst_host_same_srv_rate if dst_host_count else 0.0
    dst_host_same_src_port_rate = (
        source_port_counter.most_common(1)[0][1] / packet_count if source_port_counter and packet_count else 0.0
    )
    dst_host_srv_diff_host_rate = srv_diff_host_rate
    dst_host_serror_rate = serror_rate
    dst_host_srv_serror_rate = srv_serror_rate
    dst_host_rerror_rate = rerror_rate
    dst_host_srv_rerror_rate = srv_rerror_rate

    feature_record = {
        "duration": timeout,
        "protocol_type": dominant_protocol,
        "service": dominant_service,
        "flag": dominant_flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "land": 1 if top_source and top_destination and top_source == top_destination else 0,
        "wrong_fragment": wrong_fragments,
        "urgent": urgent_packets,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 0,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": packet_count,
        "srv_count": same_service_packets,
        "serror_rate": round(serror_rate, 4),
        "srv_serror_rate": round(srv_serror_rate, 4),
        "rerror_rate": round(rerror_rate, 4),
        "srv_rerror_rate": round(srv_rerror_rate, 4),
        "same_srv_rate": round(same_srv_rate, 4),
        "diff_srv_rate": round(diff_srv_rate, 4),
        "srv_diff_host_rate": round(srv_diff_host_rate, 4),
        "dst_host_count": dst_host_count,
        "dst_host_srv_count": dst_host_srv_count,
        "dst_host_same_srv_rate": round(dst_host_same_srv_rate, 4),
        "dst_host_diff_srv_rate": round(dst_host_diff_srv_rate, 4),
        "dst_host_same_src_port_rate": round(dst_host_same_src_port_rate, 4),
        "dst_host_srv_diff_host_rate": round(dst_host_srv_diff_host_rate, 4),
        "dst_host_serror_rate": round(dst_host_serror_rate, 4),
        "dst_host_srv_serror_rate": round(dst_host_srv_serror_rate, 4),
        "dst_host_rerror_rate": round(dst_host_rerror_rate, 4),
        "dst_host_srv_rerror_rate": round(dst_host_srv_rerror_rate, 4),
    }
    prediction = predict_records([feature_record])[0]
    report = build_prediction_report(prediction.predicted_label, prediction.confidence, feature_record)

    return {
        "packet_count": packet_count,
        "bytes_seen": bytes_seen,
        "source_ip": top_source,
        "destination_ip": top_destination,
        "protocol": dominant_protocol,
        "service": dominant_service,
        "flag": dominant_flag,
        "predicted_label": prediction.predicted_label,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "feature_record": feature_record,
        "severity": report.severity,
        "summary": report.summary,
        "rationale": report.rationale,
        "recommended_action": report.recommended_action,
    }
