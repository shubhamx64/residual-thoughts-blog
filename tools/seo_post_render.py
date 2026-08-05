"""Normalize and verify the indexing signals in the rendered Quarto site."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree


PROJECT_DIR = Path(__file__).resolve().parents[1]
SITE_DIR = PROJECT_DIR / "_site"
SITEMAP_PATH = SITE_DIR / "sitemap.xml"


def pretty_url(url: str) -> str:
    """Match Quarto's canonical URLs for directory index documents."""
    parts = urlsplit(url)
    if not parts.path.endswith("/index.html"):
        return url
    path = parts.path[: -len("index.html")]
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def normalize_sitemap() -> None:
    sitemap = SITEMAP_PATH.read_text(encoding="utf-8")

    def replace_location(match: re.Match[str]) -> str:
        return f"<loc>{pretty_url(match.group(1))}</loc>"

    normalized = re.sub(r"<loc>([^<]+)</loc>", replace_location, sitemap)
    SITEMAP_PATH.write_text(normalized, encoding="utf-8", newline="\n")


def fail(message: str) -> None:
    print(f"SEO audit failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def audit_site() -> None:
    html_files = [
        path
        for path in SITE_DIR.rglob("*.html")
        if "site_libs" not in path.relative_to(SITE_DIR).parts
    ]
    if not html_files:
        fail("no rendered HTML pages were found")

    canonical_urls: set[str] = set()
    for path in html_files:
        html = path.read_text(encoding="utf-8")
        canonicals = re.findall(
            r'<link\s+rel="canonical"\s+href="([^"]+)"', html, re.IGNORECASE
        )
        relative_path = path.relative_to(SITE_DIR)
        if len(canonicals) != 1:
            fail(f"{relative_path} has {len(canonicals)} canonical links")
        if re.search(r"(?:name|content)=[\"'][^\"']*noindex", html, re.IGNORECASE):
            fail(f"{relative_path} contains a noindex directive")
        canonical_urls.add(canonicals[0])

    sitemap_root = ElementTree.parse(SITEMAP_PATH).getroot()
    namespace = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    sitemap_urls = {
        element.text
        for element in sitemap_root.findall("sitemap:url/sitemap:loc", namespace)
        if element.text
    }
    if sitemap_urls != canonical_urls:
        missing = sorted(canonical_urls - sitemap_urls)
        unexpected = sorted(sitemap_urls - canonical_urls)
        fail(f"sitemap/canonical mismatch; missing={missing}, unexpected={unexpected}")

    robots = (SITE_DIR / "robots.txt").read_text(encoding="utf-8")
    if "https://www.residual-thoughts.com/sitemap.xml" not in robots:
        fail("robots.txt does not advertise the sitemap")

    index_html = (SITE_DIR / "index.html").read_text(encoding="utf-8")
    if not (SITE_DIR / "index.xml").is_file():
        fail("RSS feed was not generated")
    if 'type="application/rss+xml"' not in index_html:
        fail("the home page does not advertise the RSS feed")

    print(
        f"SEO audit passed: {len(html_files)} canonical pages, "
        "canonical sitemap, robots discovery, and RSS discovery."
    )


if __name__ == "__main__":
    normalize_sitemap()
    audit_site()
