# OBS packaging

The files in this directory are the package sources the openSUSE Build
Service uses to build packages of darktable `master` for openSUSE
Tumbleweed, Fedora, Debian and Ubuntu, published in
[graphics:darktable:master](https://build.opensuse.org/package/show/graphics:darktable:master/darktable).

They live next to the code so that a change which adds a dependency or an
installed file can update the packaging in the same pull request.

## Layout

OBS copies every file in this directory into the package, and it cannot
copy subdirectories, so the directory has to stay flat.

- `darktable.spec`: RPM builds (openSUSE, Fedora). `darktable-rpmlintrc`,
  `README.openSUSE`, `series` and the patch are sources it references.
- `darktable.dsc`: Debian and Ubuntu builds. OBS installs the
  `Build-Depends` listed here.
- `debian.<name>`: the Debian packaging. OBS turns each of these files into
  `debian/<name>`, and adds `debian/source/format` and a changelog entry
  for the build.

## Adding or removing a dependency

- `BuildRequires` in `darktable.spec`. Prefer `pkgconfig(...)` names, which
  are the same on openSUSE and Fedora.
- `Build-Depends` in `debian.control` and in `darktable.dsc`, identically.

A new installed file needs a matching entry in `%files` in the spec and in
`debian.darktable.install`.

## The OBS side

The OBS package keeps only `_service`, `_constraints` and the generated
`darktable.changes`. Its `_service` exports this directory with one more
parameter to the existing `obs_scm` service:

    <param name="extract">packaging/obs/*</param>

When that line is added, the copies of these files in the OBS package have
to be deleted: OBS refuses to build when a Debian file exists both in a
`debian.tar.xz` and as `debian.<name>`.

Project-wide settings stay in OBS: the distributions and architectures to
build for, and the `Prefer:` rules in the project configuration that pick
between two packages of a distribution that both satisfy a build
dependency.
