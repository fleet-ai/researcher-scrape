#!/usr/bin/env python3
"""Thorough personal-email hunting for the recruiter shortlist.

Sources, in order of preference:
  1. The person's homepage + linked contact/about/CV pages (incl. PDF CVs),
     with obfuscation decoding: "name [at] gmail [dot] com" variants and
     Cloudflare's data-cfemail hex encoding.
  2. GitHub commit author emails via the public events API (researchers'
     commits usually carry their personal address).
  3. First page of their own arXiv papers (author emails are printed there).

Policy: personal-provider addresses (gmail etc.) win; university addresses
are acceptable; corporate addresses are NEVER returned — mailing someone's
work address at the employer we're recruiting them away from is worse than
no email at all.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
import time
import urllib.parse

import requests

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

PERSONAL_PROVIDERS = {
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com",
    "msn.com", "yahoo.com", "ymail.com", "proton.me", "protonmail.com",
    "pm.me", "icloud.com", "me.com", "mac.com", "fastmail.com", "hey.com",
    "qq.com", "163.com", "126.com", "foxmail.com", "gmx.com", "gmx.de",
    "gmx.net", "web.de", "naver.com", "yandex.com", "zoho.com", "aol.com",
    "duck.com", "mail.com", "posteo.de", "tutanota.com",
}

_ACADEMIC_RE = re.compile(
    r"(\.edu$|\.edu\.[a-z]{2}$|\.ac\.[a-z]{2}$|\.ac\.[a-z]{2}\.[a-z]{2}$"
    r"|\.uni-[a-z]+\.de$|\.ethz\.ch$|\.epfl\.ch$|\.mila\.quebec$"
    r"|\.mpg\.de$|\.inria\.fr$|\.cnrs\.fr$|\.edu\.au$|\.uwaterloo\.ca$"
    r"|\.utoronto\.ca$|\.mcgill\.ca$|\.ubc\.ca$|\.ox\.ac\.uk$|\.cam\.ac\.uk$)"
)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
OBFUSCATED_RE = re.compile(
    r"([\w.+-]{2,})\s*[\[\({<]?\s*(?:at|AT|@)\s*[\]\)}>]?\s*"
    r"([\w-]{2,}(?:\s*[\[\({<]?\s*(?:dot|DOT|\.)\s*[\]\)}>]?\s*[\w-]{2,})+)",
)
CFEMAIL_RE = re.compile(r'data-cfemail="([0-9a-fA-F]+)"')

JUNK_LOCALPARTS = {"info", "admin", "webmaster", "office", "contact", "support",
                   "press", "hello", "noreply", "no-reply", "sales", "help"}
JUNK_DOMAINS = {"example.com", "email.com", "domain.com", "test.com",
                "sentry.io", "wixpress.com", "github.com", "users.noreply.github.com"}


def domain_kind(email: str) -> str:
    """'personal' | 'academic' | 'corporate' | 'junk'."""
    email = email.lower().strip()
    if "@" not in email:
        return "junk"
    local, dom = email.rsplit("@", 1)
    if dom in JUNK_DOMAINS or local in JUNK_LOCALPARTS or "noreply" in dom:
        return "junk"
    if dom in PERSONAL_PROVIDERS:
        return "personal"
    if _ACADEMIC_RE.search("." + dom):
        return "academic"
    return "corporate"


def _name_tokens(name: str) -> set:
    return {t.lower() for t in re.split(r"[\s,.-]+", name) if len(t) > 2}


def _matches_name(email: str, name: str) -> bool:
    local = email.split("@")[0].lower()
    toks = _name_tokens(name)
    return any(t in local for t in toks) or any(
        t[0] + u in local for t in toks for u in toks if t != u and len(u) > 3
    )


def _decode_cfemail(hexstr: str) -> str:
    try:
        key = int(hexstr[:2], 16)
        return "".join(chr(int(hexstr[i:i + 2], 16) ^ key) for i in range(2, len(hexstr), 2))
    except Exception:
        return ""


def _emails_from_text(text: str) -> list[str]:
    found = list(EMAIL_RE.findall(text))
    # De-obfuscate "name [at] gmail [dot] com" forms
    for m in OBFUSCATED_RE.finditer(text):
        local, rest = m.group(1), m.group(2)
        dom = re.sub(r"\s*[\[\({<]?\s*(?:dot|DOT)\s*[\]\)}>]?\s*|\s+", ".", rest)
        dom = re.sub(r"\.{2,}", ".", dom).strip(".")
        if "." in dom:
            found.append(f"{local}@{dom}")
    return found


def _fetch(url: str, timeout: int = 12) -> tuple[str, str]:
    """Return (text, content_type). PDF bytes are converted via pdftotext."""
    try:
        resp = requests.get(url, timeout=timeout, headers=UA)
        if resp.status_code != 200:
            return "", ""
        ctype = resp.headers.get("content-type", "")
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(resp.content)
                path = f.name
            try:
                out = subprocess.run(["pdftotext", "-l", "3", path, "-"],
                                     capture_output=True, timeout=20, text=True)
                return out.stdout or "", "pdf"
            finally:
                os.unlink(path)
        text = resp.text
        # Cloudflare-obfuscated emails decode from the raw HTML
        for hexstr in CFEMAIL_RE.findall(text):
            decoded = _decode_cfemail(hexstr)
            if decoded:
                text += f" {decoded} "
        return text, "html"
    except Exception:
        return "", ""


def hunt_homepage(name: str, website: str) -> list[str]:
    """Homepage + contact/about/CV subpages (incl. PDF CVs)."""
    if not website:
        return []
    url = website if website.startswith("http") else "https://" + website
    text, kind = _fetch(url)
    if not text:
        return []
    emails = _emails_from_text(text)

    if kind == "html":
        host = urllib.parse.urlparse(url).netloc
        links = re.findall(r'href=["\']([^"\']+)["\']', text, re.I)
        followed = 0
        for link in links:
            if followed >= 4:
                break
            low = link.lower()
            if not any(k in low for k in ("contact", "about", "cv", "resume", "vitae")):
                continue
            full = urllib.parse.urljoin(url, link)
            if urllib.parse.urlparse(full).netloc != host and not low.endswith(".pdf"):
                continue
            sub_text, _ = _fetch(full)
            emails += _emails_from_text(sub_text)
            followed += 1
            time.sleep(0.5)
    return emails


def hunt_github(name: str, website: str) -> list[str]:
    """Commit author emails from the person's public GitHub events."""
    user = ""
    if website:
        m = re.search(r"([\w-]+)\.github\.io", website)
        if m:
            user = m.group(1)
    if not user and website:
        text, _ = _fetch(website if website.startswith("http") else "https://" + website)
        m = re.search(r"github\.com/([\w-]+)[\"'/]", text or "")
        if m:
            user = m.group(1)
    if not user:
        return []

    headers = dict(UA)
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(f"https://api.github.com/users/{user}/events/public",
                            headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        emails = []
        for ev in resp.json():
            for c in (ev.get("payload", {}).get("commits") or []):
                e = (c.get("author") or {}).get("email", "")
                if e and "noreply" not in e:
                    emails.append(e)
        return emails
    except Exception:
        return []


def hunt_arxiv(name: str, paper_titles: list[str]) -> list[str]:
    """Author emails from the first pages of the person's arXiv papers."""
    emails = []
    for title in paper_titles[:2]:
        title = (title or "").strip()
        if len(title) < 10:
            continue
        try:
            q = urllib.parse.quote(f'ti:"{title[:100]}"')
            resp = requests.get(
                f"https://export.arxiv.org/api/query?search_query={q}&max_results=1",
                timeout=15, headers=UA)
            m = re.search(r"arxiv\.org/abs/([\w.]+)", resp.text or "")
            if not m:
                continue
            text, _ = _fetch(f"https://arxiv.org/pdf/{m.group(1)}")
            emails += _emails_from_text(text)
            time.sleep(1)
        except Exception:
            continue
    return emails


def _pick(cands: list[str], name: str) -> str:
    """Best candidate: personal+name-match > personal > academic+name-match."""
    seen, personal_match, personal, academic_match = [], [], [], []
    for e in cands:
        e = e.strip().strip(".,;:()<>[]")
        if not e or e.lower() in seen:
            continue
        seen.append(e.lower())
        kind = domain_kind(e)
        if kind == "junk" or kind == "corporate":
            continue
        match = _matches_name(e, name)
        if kind == "personal" and match:
            personal_match.append(e)
        elif kind == "personal":
            personal.append(e)
        elif kind == "academic" and match:
            academic_match.append(e)
    for bucket in (personal_match, personal, academic_match):
        if bucket:
            return bucket[0]
    return ""


def hunt_personal_email(name: str, website: str, paper_titles: list[str],
                        cache: dict | None = None) -> str:
    """Full hunt. Returns '' rather than a corporate address."""
    key = re.sub(r"\s+", " ", name.lower().strip())
    if cache is not None and key in cache:
        return cache[key]

    cands = hunt_homepage(name, website)
    best = _pick(cands, name)
    if not best or domain_kind(best) != "personal":
        cands += hunt_github(name, website)
        best = _pick(cands, name)
    if not best or domain_kind(best) != "personal":
        cands += hunt_arxiv(name, paper_titles)
        best = _pick(cands, name)

    if cache is not None:
        cache[key] = best
    return best
