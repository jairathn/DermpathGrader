"""Fetch an image from a URL, safely, for the app and the test suite.

Why this needs care
-------------------
The deployed app holds an API key and runs inside somebody's network. A
plain `requests.get(user_supplied_url)` on a server is a server-side
request forgery primitive: a visitor could point it at
`http://169.254.169.254/` (cloud instance metadata), `http://localhost:*`
(anything else on the box), or a private RFC1918 address, and the server
would fetch it for them.

So every fetch here:

- requires https (http is allowed only when `allow_http=True`, which the
  test suite uses for its own local fixture server and the app never does);
- resolves the hostname and rejects loopback, private, link-local,
  reserved, and multicast addresses, **on every redirect hop**, not just
  the first URL;
- caps the response size while streaming, so a hostile endpoint cannot
  stream gigabytes into memory;
- has a hard timeout;
- checks the declared content type, and then hands the bytes to
  `image_utils`, which checks the actual magic bytes. The declared type
  is a hint; the bytes are the check.

The size cap and timeout also protect against an honest mistake, like
pasting a link to a whole-slide export.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request

MAX_BYTES = 40 * 1024 * 1024        # generous for a tile, tiny for a WSI
TIMEOUT_S = 30
MAX_REDIRECTS = 4
USER_AGENT = ("DermpathGrader/2.1 (research image fetch; "
              "https://github.com/jairathn/DermpathGrader)")

ACCEPTED_CONTENT_TYPES = (
    "image/jpeg", "image/jpg", "image/png", "image/tiff", "image/bmp",
    "application/octet-stream",      # some CDNs are unhelpful; bytes decide
)


class ImageFetchError(RuntimeError):
    """The URL could not be fetched, or was not allowed to be."""


def _resolved_addresses(host: str) -> list[ipaddress._BaseAddress]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ImageFetchError(f"cannot resolve host {host!r}: {exc}") from exc
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def _assert_public(url: str, *, allow_http: bool) -> None:
    """Reject anything that is not a public http(s) endpoint."""
    parts = urllib.parse.urlparse(url)
    scheme = parts.scheme.lower()

    if scheme == "http" and not allow_http:
        raise ImageFetchError(
            f"{url}: http is not allowed, use https "
            f"(an http URL can be intercepted, and is usually a sign the "
            f"link points somewhere internal)")
    if scheme not in ("http", "https"):
        raise ImageFetchError(f"{url}: only http(s) URLs are supported")
    if not parts.hostname:
        raise ImageFetchError(f"{url}: no hostname")

    for address in _resolved_addresses(parts.hostname):
        if (address.is_private or address.is_loopback or address.is_reserved
                or address.is_link_local or address.is_multicast
                or address.is_unspecified):
            raise ImageFetchError(
                f"{url}: refuses to fetch {address}, which is not a public "
                f"address. Server-side fetches are restricted to the public "
                f"internet so that this app cannot be used to reach hosts "
                f"inside its own network.")


def fetch_image_bytes(url: str, *, allow_http: bool = False,
                      max_bytes: int = MAX_BYTES,
                      timeout_s: int = TIMEOUT_S) -> tuple[bytes, str]:
    """Return (raw_bytes, final_url). Raises ImageFetchError.

    Redirects are followed manually so each hop is validated; the stdlib
    would otherwise follow a redirect from a public host to a private one
    without asking.
    """
    seen = url
    for _ in range(MAX_REDIRECTS + 1):
        _assert_public(seen, allow_http=allow_http)
        request = urllib.request.Request(
            seen, headers={"User-Agent": USER_AGENT, "Accept": "image/*"})
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            response = opener.open(request, timeout=timeout_s)
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308):
                location = exc.headers.get("Location")
                if not location:
                    raise ImageFetchError(
                        f"{seen}: redirect with no Location") from exc
                seen = urllib.parse.urljoin(seen, location)
                continue
            raise ImageFetchError(f"{seen}: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise ImageFetchError(f"{seen}: {exc.reason}") from exc

        with response:
            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if content_type and content_type not in ACCEPTED_CONTENT_TYPES:
                raise ImageFetchError(
                    f"{seen}: server says Content-Type {content_type!r}, "
                    f"which is not an image type")

            declared = response.headers.get("Content-Length")
            if declared and int(declared) > max_bytes:
                raise ImageFetchError(
                    f"{seen}: {int(declared)} bytes exceeds the "
                    f"{max_bytes}-byte limit; this looks like a whole-slide "
                    f"export rather than a tile")

            chunks, total = [], 0
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ImageFetchError(
                        f"{seen}: response exceeded the {max_bytes}-byte "
                        f"limit while downloading")
                chunks.append(chunk)
        return b"".join(chunks), seen

    raise ImageFetchError(f"{url}: too many redirects")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface redirects as HTTPError so each hop can be re-validated."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_case_images(urls: dict[str, str], *, allow_http: bool = False
                      ) -> dict[str, bytes]:
    """magnification -> URL, fetched into magnification -> bytes.

    Errors name the magnification, because "one of your four links is
    broken" is not an actionable message.
    """
    out: dict[str, bytes] = {}
    for magnification, url in urls.items():
        url = (url or "").strip()
        if not url:
            continue
        try:
            out[magnification], _ = fetch_image_bytes(url, allow_http=allow_http)
        except ImageFetchError as exc:
            raise ImageFetchError(f"{magnification}: {exc}") from exc
    return out
