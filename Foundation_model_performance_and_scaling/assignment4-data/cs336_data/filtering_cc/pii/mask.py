from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


def mask_emails(text: str) -> tuple[str, int]:
    count = 0

    def _replace(m: re.Match) -> str:
        nonlocal count
        count += 1
        return "|||EMAIL_ADDRESS|||"

    masked = _EMAIL_RE.sub(_replace, text)
    return masked, count


# ---------------------------------------------------------------------------
# Phone numbers  (common US formats)
# ---------------------------------------------------------------------------
_PHONE_RE = re.compile(
    r"(?<!\d)\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"  # NXX-NXX-XXXX variants
    r"|\b\d{10}\b"                                        # 10 bare digits
)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    count = 0

    def _replace(m: re.Match) -> str:
        nonlocal count
        count += 1
        return "|||PHONE_NUMBER|||"

    masked = _PHONE_RE.sub(_replace, text)
    return masked, count


# ---------------------------------------------------------------------------
# IPv4 addresses
# ---------------------------------------------------------------------------
_IP_RE = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")


def _valid_ip(m: re.Match) -> bool:
    return all(0 <= int(g) <= 255 for g in m.groups())


def mask_ips(text: str) -> tuple[str, int]:
    count = 0

    def _replace(m: re.Match) -> str:
        nonlocal count
        if not _valid_ip(m):
            return m.group(0)
        count += 1
        return "|||IP_ADDRESS|||"

    masked = _IP_RE.sub(_replace, text)
    return masked, count