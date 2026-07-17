"""Parse an IPUMS DDI codebook (cps_000XX.xml) into a fixed-width read spec.

IPUMS fixed-width extracts do not use a fixed, predictable column layout --
positions depend on which variables were selected and in what order for a
given extract. The DDI codebook is the only reliable source of truth for
where each variable lives in the .dat file, so we parse it directly rather
than hardcoding offsets.

Verified against cps_00001.xml / cps_00001.dat: the DDI does not populate
<location StartPos=.../> for this extract, so positions are computed as a
running cumulative sum of <location width=.../> in document order (which is
also the on-disk column order). This was checked against the raw file: the
sum of all variable widths equals the exact line length (256), and spot
values (YEAR, AGE, INCWAGE, ...) decode to plausible values at those offsets.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

DDI_NS = {"d": "ddi:codebook:2_5"}


@dataclass(frozen=True)
class VarSpec:
    name: str
    start: int  # inclusive, 0-indexed
    end: int  # exclusive
    width: int
    decimals: int  # implied decimal places (e.g. ASECWT has decimals=4)


def parse_ddi(xml_path: str) -> list[VarSpec]:
    """Return column specs for every variable, in on-disk order."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    specs: list[VarSpec] = []
    pos = 0
    for var in root.findall(".//d:dataDscr/d:var", DDI_NS):
        name = var.attrib["name"]
        loc = var.find("d:location", DDI_NS)
        width = int(loc.attrib["width"])
        decimals = int(var.attrib.get("dcml", 0))
        specs.append(VarSpec(name, pos, pos + width, width, decimals))
        pos += width

    return specs


def colspecs_for_fwf(specs: list[VarSpec]) -> list[tuple[int, int]]:
    """(start, end) tuples in the format pandas.read_fwf expects."""
    return [(s.start, s.end) for s in specs]


def names_for_fwf(specs: list[VarSpec]) -> list[str]:
    return [s.name for s in specs]


def decimals_map(specs: list[VarSpec]) -> dict[str, int]:
    return {s.name: s.decimals for s in specs if s.decimals > 0}
