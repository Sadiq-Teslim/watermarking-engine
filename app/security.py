"""SSRF guard for caller-supplied source URLs."""
import ipaddress
import socket
from urllib.parse import urlparse

from fastapi import HTTPException, status

_BLOCKED_HOST_PREFIXES = ("localhost",)


def _is_private_ip(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        # Can't resolve -> treat as unsafe.
        return True
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return True
    return False


def validate_source_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="source_url must be http(s)")
    host = parsed.hostname or ""
    if not host or host.lower().startswith(_BLOCKED_HOST_PREFIXES):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="source_url host not allowed")
    if _is_private_ip(host):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="source_url resolves to a private address")
