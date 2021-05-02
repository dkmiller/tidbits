from http.cookies import SimpleCookie
import jwt
from typing import Dict
from haralyzer import HarEntry, HarPage, HarParser
import logging
from urllib.parse import parse_qs, urlparse


log = logging.getLogger(__name__)


class HttpArchiveVisitor:
    def __init__(self, truncate: int = 40):
        self.truncate = truncate
        self._visited = {}
        self._visited_values = set()

    def _log_visit(self, o, name: str = None) -> None:
        try:
            self._visited_values.add(o)
        except TypeError:
            # Some values are unhashable, can't de-duplicate.
            pass
        name = name or type(o).__name__

        if name in self._visited:
            self._visited[name] += 1
        else:
            self._visited[name] = 1

        s = str(o)
        if len(s) > self.truncate:
            s = s[: self.truncate] + "..."
        log.debug(f"Visiting {name} '{s}'")

    def summarize(self):
        log.info(f"Visited {len(self._visited)} kinds of objects")
        total = sum(self._visited.values())
        log.info(f"Visited {total} total objects")

    def visit(self, archive: HarParser):
        self._log_visit(archive)
        pages = archive.pages
        log.debug(f"Visiting {len(pages)} pages")
        for page in pages:
            self.visit_page(page)

    def visit_cookie(self, cookie: SimpleCookie):
        for key, value in cookie.items():
            self.visit_value(key, value.value)

    def visit_entry(self, entry: HarEntry):
        self._log_visit(entry)
        self.visit_url(entry.url)
        headers = entry.request.headers
        log.debug(f"Visiting {len(headers)} headers")
        for header in headers:
            self.visit_header(header)

    def visit_header(self, header: Dict[str, str]):
        self._log_visit(header, "Header")
        name = header["name"]
        value = header["value"]
        if name.lower() == "cookie":
            cookie = SimpleCookie(value)
            self.visit_cookie(cookie)
        self.visit_header_value(name, value)

    def visit_header_value(self, name: str, value: str):
        self._log_visit(value, name)

    def visit_page(self, page: HarPage):
        self._log_visit(page)
        self.visit_url(page.url)
        entries = page.entries
        log.debug(f"Visiting {len(entries)} entries")
        for entry in entries:
            self.visit_entry(entry)

    def visit_url(self, url: str):
        self._log_visit(url, name="URL")
        parsed_url = urlparse(url)
        query = parse_qs(parsed_url.query)
        for key, values in query.items():
            for value in values:
                self.visit_value(key, value)

    def visit_value(self, key: str, value: str):
        if value in self._visited_values:
            return
        
        self._log_visit(value, name=key)

        key = key.lower()

        if key == "appid":
            log.info(f"App ID {value}")
        elif key == "muid":
            log.info(f"MUID {value}")

        values = value.split(" ")
        for val in values:
            try:
                token = jwt.get_unverified_header(val)
                log.info(f"JWT token {token}")
            except:
                pass

#
# g UserAuthentication 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1...'
# DEBUG:visitors:Visiting FedAuth '77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGlu...'
# DEBUG:visitors:Visiting FedAuth1 'UkN6OHVuNlBnaGRudzB3a2RqLzJ1U1JvcTUyZVkx...'
# DEBUG:visitors:Visiting FedAuth2 'cml0eUNvbnRleHRUb2tlbj4='
# DEBUG:visitors:Visiting HostAuthentication 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1...'