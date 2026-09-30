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

# Builds the Debian packages from packaging/obs the way the openSUSE Build
# Service does, then installs them and runs darktable-cli. It installs the
# build dependencies with apt, so run it as root in a throwaway Debian or
# Ubuntu container.
#   SRC_DIR - darktable source tree including its submodules, defaults to
#             the tree this script belongs to

set -ex

SRC_DIR=${SRC_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
WORK_DIR=$(mktemp -d)

export DEBIAN_FRONTEND=noninteractive
DEB_BUILD_OPTIONS="parallel=$(nproc)"
export DEB_BUILD_OPTIONS

# OBS builds from a tarball of the checkout, which has no .git
mkdir "$WORK_DIR/darktable"
tar -C "$SRC_DIR" --exclude=.git -cf - . | tar -C "$WORK_DIR/darktable" -xf -
cd "$WORK_DIR/darktable"

# what debtransform does on OBS: every debian.<name> becomes debian/<name>,
# the source format comes from the .dsc and the changelog gets an entry for
# the version being built
mkdir -p debian/source
for f in packaging/obs/debian.*; do
  cp "$f" "debian/${f##*/debian.}"
done
sed -n 's/^Format: //p' packaging/obs/darktable.dsc > debian/source/format
MAINTAINER=$(sed -n 's/^Maintainer: //p' debian/control)
{
  printf 'darktable (0-1) experimental; urgency=low\n\n'
  printf '  * build of packaging/obs\n\n'
  printf ' -- %s  %s\n\n' "$MAINTAINER" "$(date -R)"
  cat packaging/obs/debian.changelog
} > debian/changelog

apt-get update
apt-get install -y --no-install-recommends dpkg-dev
apt-get build-dep -y ./
dpkg-buildpackage -b -us -uc

# the packages have to install and start, not only build
apt-get install -y "$WORK_DIR"/darktable_*.deb
darktable-cli --version
