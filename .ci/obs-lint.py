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

# checks packaging/obs for mistakes that OBS would only report after the
# change is merged, without building anything

import re
import sys
from pathlib import Path

OBS_DIR = Path(__file__).resolve().parent.parent / "packaging" / "obs"


def build_depends(path):
    # the entries of the Build-Depends field of a Debian control or .dsc file
    entries = []
    in_field = False
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("Build-Depends:"):
            in_field = True
            line = line[len("Build-Depends:"):]
        elif in_field and not line[:1].isspace():
            break
        if in_field:
            entries += [e.strip() for e in line.split(",") if e.strip()]
    return entries


def main():
    errors = []

    # obs_scm exports files only, so a subdirectory would never reach OBS
    for path in sorted(OBS_DIR.iterdir()):
        if not path.is_file():
            errors.append(f"{path.name}: packaging/obs has to stay flat")

    # OBS installs the Build-Depends of the .dsc, while dpkg-buildpackage
    # checks those of debian/control, so both lists have to agree
    dsc = build_depends(OBS_DIR / "darktable.dsc")
    control = build_depends(OBS_DIR / "debian.control")
    if not dsc or not control:
        errors.append("no Build-Depends found in darktable.dsc or debian.control")
    for entry in sorted(set(control) - set(dsc)):
        errors.append(f"'{entry}' is in debian.control but not in darktable.dsc")
    for entry in sorted(set(dsc) - set(control)):
        errors.append(f"'{entry}' is in darktable.dsc but not in debian.control")

    # every source except the tarball OBS generates has to be in the directory
    spec = (OBS_DIR / "darktable.spec").read_text()
    for kind, number, name in re.findall(r"^(Source|Patch)(\d*):\s*(\S+)", spec, re.M):
        if kind == "Source" and number in ("", "0"):
            continue
        name = name.replace("%{pkg_name}", "darktable").replace("%{name}", "darktable")
        if not (OBS_DIR / name).is_file():
            errors.append(f"{kind}{number} of darktable.spec, {name}, is missing")

    for error in errors:
        print(f"packaging/obs: {error}")
    if not errors:
        print(f"packaging/obs: {len(control)} Debian build dependencies, spec sources present")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
