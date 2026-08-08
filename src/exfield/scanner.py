"""Stream primitives for reading EX files.

Mirrors the behaviour of Zinc's ``IO_stream`` functions as used by
``EXReader`` in ``import_finite_element.cpp``:

* ``scan(" literal %d")``-style matching follows C ``scanf`` semantics: a
  whitespace character in the pattern matches any run of whitespace
  (including none); other characters must match exactly; ``%d``/``%f``
  skip leading whitespace; ``%1[c]`` matches exactly one character from a
  set with **no** leading whitespace skip.
* ``read_charset("^,\\n\\r")`` mirrors ``IO_stream_read_string`` with a
  scanset format: reads the maximal (possibly empty) run of characters in
  (or not in) the set.

The EX format is a stream, not a line grammar — values wrap lines freely
and several tokens may share a line — so the scanner is offset-based over
one in-memory string and line numbers are derived from the offset for
error reporting only.
"""

import re

from .errors import ExSyntaxError

_WHITESPACE = " \t\n\r\f\v"
_WS_RE = re.compile(r"[ \t\n\r\f\v]*")
_REAL_RE = re.compile(
    r"[ \t\n\r\f\v]*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
    r"|[+-]?(?:nan|inf(?:inity)?))", re.IGNORECASE)
_INT_RE = re.compile(r"[ \t\n\r\f\v]*([+-]?\d+)")
_CHARSET_RE_CACHE = {}


class Scanner:
    """Offset-based scanner over the full EX file contents."""

    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.length = len(text)

    # ------------------------------------------------------------ location

    @property
    def line(self):
        """1-based line number at the current position (for errors)."""
        return self.text.count("\n", 0, self.pos) + 1

    def error(self, message):
        return ExSyntaxError(message, line=self.line)

    def at_eof(self):
        return self.pos >= self.length

    # ---------------------------------------------------- char primitives

    def peekc(self):
        """Next character without consuming, or '' at EOF."""
        if self.pos < self.length:
            return self.text[self.pos]
        return ""

    def getc(self):
        """Consume and return next character, or '' at EOF."""
        c = self.peekc()
        if c:
            self.pos += 1
        return c

    def next_non_space_char(self):
        """Mirror of ``EXReader::readNextNonSpaceChar``: skip ' ' only
        (not tabs or newlines) and consume the next character."""
        while self.peekc() == " ":
            self.pos += 1
        return self.getc()

    def check_consume_next_char(self, test_char):
        """Mirror of ``EXReader::checkConsumeNextChar``."""
        if self.peekc() == test_char:
            self.pos += 1
            return True
        return False

    def skip_whitespace(self):
        """Skip any whitespace (scanf ' ' semantics)."""
        self.pos = _WS_RE.match(self.text, self.pos).end()

    # ------------------------------------------------------- scanf-alikes

    def match_literal(self, pattern):
        """Match ``pattern`` with scanf literal semantics.

        Whitespace in the pattern matches any run of whitespace (including
        none); all other characters must match exactly. On failure the
        position is restored and False returned.
        """
        start = self.pos
        for ch in pattern:
            if ch in _WHITESPACE:
                self.skip_whitespace()
            else:
                if self.peekc() == ch:
                    self.pos += 1
                else:
                    self.pos = start
                    return False
        return True

    def match_one_of(self, chars):
        """Mirror of scanf ``%1[chars]``: match exactly one character from
        the set, with no whitespace skip. Restores position on failure."""
        if self.peekc() in chars and self.peekc() != "":
            self.pos += 1
            return True
        return False

    def read_int(self, message="Missing integer"):
        """scanf %d: skip whitespace, read optional sign + digits."""
        m = _INT_RE.match(self.text, self.pos)
        if m is None:
            raise self.error(message)
        self.pos = m.end()
        return int(m.group(1))

    def try_read_int(self):
        try:
            return self.read_int()
        except ExSyntaxError:
            return None

    def read_real(self, message="Missing real value"):
        """scanf %lf: skip whitespace, read a C-style floating point
        number (also accepts nan/inf, which callers must reject where the
        reference implementation does)."""
        m = _REAL_RE.match(self.text, self.pos)
        if m is None:
            raise self.error(message)
        self.pos = m.end()
        return float(m.group(1))

    def read_charset(self, scanset):
        """Mirror of ``IO_stream_read_string(file, "[scanset]", ...)``.

        ``scanset`` is the scanf set contents, e.g. ``"^,\\n\\r"`` for
        "everything up to a comma or end of line". Returns the (possibly
        empty) matched run. Does not skip leading whitespace.
        """
        pattern = _CHARSET_RE_CACHE.get(scanset)
        if pattern is None:
            pattern = re.compile("[" + scanset.replace("\\", "\\\\") + "]*")
            _CHARSET_RE_CACHE[scanset] = pattern
        m = pattern.match(self.text, self.pos)
        result = m.group(0)
        self.pos = m.end()
        return result

    def read_rest_of_line(self):
        """Read to (not including) the end-of-line characters."""
        return self.read_charset("^\n\r")

    def read_blank_to_end_of_line(self):
        """Mirror of ``EXReader::readBlankToEndOfLine``: rest of line must
        be blank; then consume the EOL and following whitespace."""
        rest = self.read_rest_of_line()
        if rest.strip(" \t"):
            raise self.error(f"Unexpected text '{rest.strip()}' on line")
        self.skip_whitespace()

    # ------------------------------------------------------- EX strings

    def read_ex_string(self, delimiters=" ,;=\n\r\t"):
        """Mirror of ``EXReader::readString``.

        Skips leading whitespace. If the first character is a single or
        double quote, reads until the matching unescaped quote (backslash
        escapes; ``\\n``/``\\t``/``\\r`` translated, ``\\$`` -> ``$``),
        which may span lines. Otherwise reads until any delimiter
        character. Raises on empty or unterminated string.
        """
        self.skip_whitespace()
        quote = ""
        if self.check_consume_next_char('"'):
            quote = '"'
        elif self.check_consume_next_char("'"):
            quote = "'"
        if not quote:
            s = self.read_charset("^" + delimiters)
            if not s:
                raise self.error("Missing string")
            return s
        out = []
        while True:
            c = self.getc()
            if c == "":
                raise self.error("Missing end quote on string")
            if c == "\\":
                e = self.getc()
                if e == "":
                    raise self.error("End of file after escape character in string")
                out.append({"n": "\n", "t": "\t", "r": "\r"}.get(e, e))
            elif c == quote:
                nxt = self.peekc()
                if nxt and nxt not in _WHITESPACE and nxt != ",":
                    # Zinc requires whitespace or EOF after the end quote;
                    # a comma follows in key=value lists so allow it.
                    raise self.error(
                        "Require whitespace after end quote on string")
                return "".join(out)
            else:
                out.append(c)

    def read_key_value_map(self, initial_separator=""):
        """Mirror of ``EXReader::readKeyValueMap``.

        Reads to end of line extracting comma-separated ``key=value``
        pairs. If ``initial_separator`` is given (e.g. ``","``), a pair
        must follow it; a bare end of line ends the list. Consumes the end
        of line. Returns a dict preserving insertion order.
        """
        result = {}
        separator = initial_separator
        while True:
            if separator:
                c = self.next_non_space_char()
                if c != separator:
                    if c in ("\n", "\r", ""):
                        break
                    rest = self.read_rest_of_line()
                    raise self.error(
                        f"Unexpected text '{c}{rest}' where only "
                        f"'{separator} key=value[, ...]' allowed")
            key = self.read_charset("^=\n\r").strip()
            c = self.getc()
            if c != "=":
                if not separator and not key:
                    break
                raise self.error(
                    f"Unexpected text '{key}' where only 'key=value[, ...]' allowed")
            if not key:
                raise self.error("Invalid key=value key")
            value = self.read_ex_string()
            if key in result:
                raise self.error("Duplicate key in key=value list")
            result[key] = value
            separator = ","
        # consume end of line characters
        while self.peekc() in ("\n", "\r") and self.peekc():
            self.pos += 1
        return result


class LineTokenizer:
    """Mirror of Zinc's ``nexttoken``: tokenise a string retaining the
    separator character that terminated each token.

    In ``d/ds1(2)+d/ds2`` the ``(`` announces a version and the ``+``
    announces another term; the terminating separator is the information
    that distinguishes the two, so it is returned with every token.
    """

    def __init__(self, s):
        self.s = s
        self.pos = 0

    def next_token(self, sepchars):
        """Return ``(token, nextchar)`` where ``nextchar`` is the
        separator or whitespace character that ended the token ('' at end
        of string). Advances past the token, trailing spaces, and at most
        one separator character."""
        s = self.s
        n = len(s)
        while self.pos < n and s[self.pos] == " ":
            self.pos += 1
        start = self.pos
        while self.pos < n and not (s[self.pos] in _WHITESPACE or s[self.pos] in sepchars):
            self.pos += 1
        token = s[start:self.pos]
        nextchar = s[self.pos] if self.pos < n else ""
        if self.pos < n:
            self.pos += 1
        if nextchar == " ":
            while self.pos < n and s[self.pos] == " ":
                self.pos += 1
            if self.pos < n and s[self.pos] in sepchars:
                nextchar = s[self.pos]
                self.pos += 1
        return token, nextchar

    def skip_spaces(self):
        while self.pos < len(self.s) and self.s[self.pos] == " ":
            self.pos += 1

    def peek(self):
        return self.s[self.pos] if self.pos < len(self.s) else ""
