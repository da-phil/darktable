#!/bin/bash

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

# Builds the RPM packages from packaging/obs the way the openSUSE Build
# Service does, then installs them and runs darktable-cli. It installs the
# build dependencies with zypper or dnf, so run it as root in a throwaway
# openSUSE or Fedora container.
#   SRC_DIR - darktable source tree including its submodules, defaults to
#             the tree this script belongs to

set -ex

SRC_DIR=${SRC_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
TOP_DIR=$(mktemp -d)
SPEC="$TOP_DIR/SPECS/darktable.spec"

if command -v zypper > /dev/null; then
  zypper --non-interactive install rpm-build tar xz
else
  dnf -y install rpm-build tar xz 'dnf5-command(builddep)'
fi

mkdir -p "$TOP_DIR/SOURCES" "$TOP_DIR/SPECS"
cp "$SRC_DIR"/packaging/obs/* "$TOP_DIR/SOURCES/"
# OBS replaces the multibuild flavor, which is empty for this package
sed 's/@BUILD_FLAVOR@//' "$SRC_DIR/packaging/obs/darktable.spec" > "$SPEC"

# the tarball the obs_scm and tar services produce: no .git, and everything
# under one directory named after the package version
VERSION=$(rpmspec -q --srpm --qf '%{version}' "$SPEC")
tar -C "$SRC_DIR" --exclude=.git --transform "s,^\.,darktable-$VERSION," -cf - . \
  | xz -T0 > "$TOP_DIR/SOURCES/darktable-$VERSION.tar.xz"

if command -v zypper > /dev/null; then
  # zypper takes a versioned capability as one argument without spaces
  rpmspec -q --buildrequires "$SPEC" | tr -d ' ' | xargs zypper --non-interactive install
else
  dnf -y builddep "$SPEC"
fi

rpmbuild -bb --define "_topdir $TOP_DIR" "$SPEC"

# the packages have to install and start, not only build
if command -v zypper > /dev/null; then
  zypper --non-interactive install --allow-unsigned-rpm "$TOP_DIR"/RPMS/*/darktable-"$VERSION"-*.rpm
else
  dnf -y install "$TOP_DIR"/RPMS/*/darktable-"$VERSION"-*.rpm
fi
darktable-cli --version
