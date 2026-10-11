"""Minimal dependency-free XLSX reader: stream a worksheet to rows of strings.

No third-party libraries (openpyxl/xlrd unavailable offline). Reads the OOXML
zip directly with xml.etree.iterparse. Handles shared strings, inline strings,
formula strings, booleans and numbers. Column position is taken from the cell
reference so blank cells do not shift columns.
"""
from __future__ import annotations
import re
import zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
COLLETTER = re.compile(r"([A-Z]+)(\d+)")


def _col_to_idx(ref: str) -> int:
    m = COLLETTER.match(ref)
    letters = m.group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_shared_strings(z: zipfile.ZipFile) -> list[str]:
    out: list[str] = []
    if "xl/sharedStrings.xml" not in z.namelist():
        return out
    for _, el in ET.iterparse(z.open("xl/sharedStrings.xml"), events=("end",)):
        if el.tag == NS + "si":
            # concatenate all <t> descendants, preserving nothing else
            txt = "".join(t.text or "" for t in el.iter(NS + "t"))
            out.append(txt)
            el.clear()
    return out


def iter_rows(z: zipfile.ZipFile, part: str, shared: list[str]):
    """Yield list-of-strings per row."""
    for _, el in ET.iterparse(z.open(part), events=("end",)):
        if el.tag != NS + "row":
            continue
        cells: dict[int, str] = {}
        for c in el.findall(NS + "c"):
            ref = c.get("r") or ""
            t = c.get("t")
            if t == "s":
                v = c.find(NS + "v")
                val = shared[int(v.text)] if v is not None and v.text is not None else ""
            elif t == "inlineStr":
                is_ = c.find(NS + "is")
                val = "".join(tt.text or "" for tt in is_.iter(NS + "t")) if is_ is not None else ""
            elif t == "b":
                v = c.find(NS + "v")
                val = "1" if (v is not None and v.text == "1") else "0"
            else:
                # numeric or formula-string (t="str")
                v = c.find(NS + "v")
                val = v.text if (v is not None and v.text is not None) else ""
            if ref:
                cells[_col_to_idx(ref)] = val
        if not cells:
            yield []
            el.clear()
            continue
        width = max(cells) + 1
        yield [cells.get(i, "") for i in range(width)]
        el.clear()
