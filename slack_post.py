#!/usr/bin/env python3
"""Slack file upload + summary post using bot token.

Uses the files.getUploadURLExternal -> upload -> files.completeUploadExternal
sequence (the old files.upload was deprecated in 2025). Bot must be in the
target channel; add it via /invite in Slack if needed.
"""

import logging
import os

import requests

log = logging.getLogger(__name__)

SLACK_API = "https://slack.com/api"


def _resolve_channel_id(channel: str, token: str) -> str | None:
    """If channel is a name (#hiring or hiring), look up its ID. IDs pass through."""
    if channel.startswith(("C", "G", "D")) and len(channel) >= 9:
        return channel
    name = channel.lstrip("#")
    cursor = ""
    for _ in range(20):
        params = {"limit": "1000", "types": "public_channel,private_channel"}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{SLACK_API}/conversations.list",
                         headers={"Authorization": f"Bearer {token}"},
                         params=params, timeout=30)
        data = r.json()
        if not data.get("ok"):
            log.warning(f"conversations.list error: {data.get('error')}")
            return None
        for c in data.get("channels", []):
            if c.get("name") == name:
                return c.get("id")
        cursor = (data.get("response_metadata") or {}).get("next_cursor", "")
        if not cursor:
            break
    return None


def upload_file(file_path: str, channel: str, token: str,
                initial_comment: str = "") -> dict | None:
    """Upload a file to a Slack channel. Returns the file object or None."""
    size = os.path.getsize(file_path)
    filename = os.path.basename(file_path)

    channel_id = _resolve_channel_id(channel, token)
    if not channel_id:
        log.error(f"Could not resolve Slack channel {channel!r}")
        return None

    # 1. Get upload URL
    r = requests.get(
        f"{SLACK_API}/files.getUploadURLExternal",
        headers={"Authorization": f"Bearer {token}"},
        params={"filename": filename, "length": str(size)},
        timeout=30,
    )
    data = r.json()
    if not data.get("ok"):
        log.error(f"getUploadURLExternal error: {data.get('error')}")
        return None
    upload_url = data["upload_url"]
    file_id = data["file_id"]

    # 2. PUT the file body
    with open(file_path, "rb") as f:
        r = requests.post(upload_url, files={"file": (filename, f)}, timeout=120)
    if r.status_code != 200:
        log.error(f"file upload failed: {r.status_code} {r.text[:200]}")
        return None

    # 3. Complete the upload
    payload = {
        "files": [{"id": file_id, "title": filename}],
        "channel_id": channel_id,
    }
    if initial_comment:
        payload["initial_comment"] = initial_comment
    r = requests.post(
        f"{SLACK_API}/files.completeUploadExternal",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json=payload,
        timeout=30,
    )
    data = r.json()
    if not data.get("ok"):
        log.error(f"completeUploadExternal error: {data.get('error')}")
        return None
    files = data.get("files") or []
    return files[0] if files else None


def post_message(text: str, channel: str, token: str) -> bool:
    """Post a plain text (or mrkdwn) message to a channel."""
    channel_id = _resolve_channel_id(channel, token) or channel
    r = requests.post(
        f"{SLACK_API}/chat.postMessage",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json={"channel": channel_id, "text": text, "mrkdwn": True},
        timeout=30,
    )
    data = r.json()
    if not data.get("ok"):
        log.error(f"chat.postMessage error: {data.get('error')}")
        return False
    return True


def build_summary(per_cat_rows: dict, all_rows: list[dict], top_n: int = 3) -> str:
    """Build a short mrkdwn summary: counts per position + top N per tab."""
    lines = [f":clipboard: *Weekly researcher discovery* — {len(all_rows)} candidates across {len(per_cat_rows)} positions"]
    for cat_name, rows in per_cat_rows.items():
        if not rows:
            lines.append(f"\n*{cat_name}* — 0")
            continue
        lines.append(f"\n*{cat_name}* — {len(rows)}")
        for r in rows[:top_n]:
            name = r.get("name", "")
            inst = r.get("institution", "") or "?"
            h = r.get("h_index", 0)
            stage = r.get("career_stage", "") or "?"
            lines.append(f"  • {name} — {inst} · h={h} · {stage}")
    return "\n".join(lines)
