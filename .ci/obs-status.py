#!/usr/bin/env python3

#    This file is part of darktable.
#    Copyright (C) 2026 darktable developers.
#
#    darktable is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    darktable is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with darktable.  If not, see <http://www.gnu.org/licenses/>.

# reports the OBS builds of darktable master and fails when one of them
# failed or cannot start. Some of these failures come from OBS itself, such
# as a distribution adding a second package that satisfies a build
# dependency, and no change in this repository can catch those

import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

PROJECT = "graphics:darktable:master"
PACKAGE = "darktable"
API = "https://api.opensuse.org/public/build"
BROKEN = {"failed", "unresolvable", "broken"}


def main():
    url = f"{API}/{PROJECT}/_result?package={PACKAGE}"
    with urllib.request.urlopen(url, timeout=60) as response:
        root = ET.parse(response).getroot()

    rows = []
    for result in root.iter("result"):
        status = result.find("status")
        code = status.get("code") if status is not None else "unknown"
        details = (status.findtext("details") or "") if status is not None else ""
        rows.append((result.get("repository"), result.get("arch"), code, details.strip()))
    rows.sort()

    table = ["| repository | arch | state | details |", "|---|---|---|---|"]
    table += [f"| {r} | {a} | {c} | {d.replace('|', '/')} |" for r, a, c, d in rows]
    print("\n".join(table))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"## {PACKAGE} in {PROJECT}\n\n" + "\n".join(table) + "\n")

    broken = [row for row in rows if row[2] in BROKEN]
    for repository, arch, code, _ in broken:
        log = f"{API}/{PROJECT}/{repository}/{arch}/{PACKAGE}/_log"
        print(f"::error::{repository} {arch} is {code}, log: {log}")
    if not rows:
        print("::error::OBS returned no build results")
    return 1 if broken or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
