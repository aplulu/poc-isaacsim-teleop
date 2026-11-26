"""Self-signed TLS certificate generation.

WebXR (and getUserMedia/WebRTC in some browsers) only works in a *secure context*, so the Quest browser
must reach the web app over HTTPS. A self-signed certificate covering localhost and every local IPv4
address is generated on first start (and regenerated if the machine's addresses change).
"""

from __future__ import annotations

import datetime
import ipaddress
import json
import os
import socket


def local_ipv4_addresses() -> list[str]:
    ips: set[str] = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    # Address used for the default route (most likely the LAN address the headset will use)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(ips)


def primary_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def ensure_certificate(cert_dir: str, extra_hosts: list[str] | None = None) -> tuple[str, str]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "server.crt")
    key_path = os.path.join(cert_dir, "server.key")
    meta_path = os.path.join(cert_dir, "san.json")

    ips = local_ipv4_addresses()
    names = sorted({"localhost", socket.gethostname(), *(extra_hosts or [])})
    wanted = {"ips": ips, "names": names}
    if os.path.exists(cert_path) and os.path.exists(key_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                have = json.load(f)
            if set(wanted["ips"]) <= set(have.get("ips", [])) and set(wanted["names"]) <= set(have.get("names", [])):
                return cert_path, key_path
        except (OSError, ValueError):
            pass

    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Isaac Sim VR Teleop (self-signed)")])
    san: list[x509.GeneralName] = [x509.DNSName(n) for n in names]
    for ip in ips:
        san.append(x509.IPAddress(ipaddress.ip_address(ip)))
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(wanted, f)
    return cert_path, key_path
