#!/usr/bin/env python3
"""Assemble a self-contained, relocatable COLMAP bundle on macOS.

Copies every dependency the binary needs -- Homebrew/local absolute paths AND
@rpath / @executable_path resolved ones -- transitively into <bundle>/libs,
rewrites all load commands to @loader_path, strips external LC_RPATHs (so the
bundle cannot silently fall back to /opt/homebrew), ad-hoc re-signs every
modified Mach-O, and finally VERIFIES the dependency closure is self-contained.

Usage:
  scripts/macos/bundle_macos.py [SRC_BINARY]
Paths are derived from this script's location (repo = ../..), so it is
reproducible from any checkout. Override the repo with $COLMAP_REPO.
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("COLMAP_REPO") or os.path.abspath(
    os.path.join(HERE, "..", "..")
)
SRC_BIN = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(REPO, "build/src/colmap/exe/colmap")
)
ROOT = os.path.join(REPO, "output/bundle/colmap")
LIB = os.path.join(ROOT, "libs")
BIN = os.path.join(ROOT, "colmap")

EXT_PREFIXES = ("/opt/homebrew", "/usr/local/", "/opt/local/")


def run(cmd, check=True):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            "command failed (%d): %s\n%s"
            % (p.returncode, " ".join(cmd), p.stderr.strip())
        )
    return p.stdout


def is_external(p):
    return p.startswith(EXT_PREFIXES)


def install_id(f):
    out = run(["otool", "-D", f]).splitlines()
    return out[1].strip() if len(out) >= 2 and out[1].strip() else None


def load_deps(f):
    """Dependencies of f, excluding system libs and f's own install id."""
    out = run(["otool", "-L", f]).splitlines()[1:]
    iid = install_id(f)
    deps = []
    for ln in out:
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split(" ")[0]
        if p == iid:  # the dylib's own LC_ID_DYLIB, not a dependency
            continue
        deps.append(p)
    return deps


def rpaths(f):
    out = run(["otool", "-l", f]).splitlines()
    res, want = [], False
    for ln in out:
        s = ln.strip()
        if s.startswith("cmd LC_RPATH"):
            want = True
        elif want and s.startswith("path "):
            res.append(s.split(" ")[1])
            want = False
    return res


def resolve(dep, origin_file):
    """Resolve a dependency string to a real file path, or None."""
    if is_external(dep):
        return os.path.realpath(dep) if os.path.exists(dep) else None
    if dep.startswith("@rpath/"):
        rel = dep[len("@rpath/") :]
        for rp in rpaths(origin_file):
            base = rp.replace(
                "@loader_path", os.path.dirname(origin_file)
            ).replace("@executable_path", os.path.dirname(origin_file))
            cand = os.path.join(base, rel)
            if os.path.exists(cand):
                return os.path.realpath(cand)
        return None
    if dep.startswith(("@executable_path/", "@loader_path/")):
        rel = dep.split("/", 1)[1]
        cand = os.path.join(os.path.dirname(origin_file), rel)
        return os.path.realpath(cand) if os.path.exists(cand) else None
    return None


copied = {}  # basename -> (dest_path, origin_realpath)
collisions = []
worklist = []


def reference_form(base, kind):
    return (
        ("@loader_path/libs/" + base)
        if kind == "bin"
        else ("@loader_path/" + base)
    )


def process(f, origin_file, kind):
    for dep in load_deps(f):
        if dep.startswith("@loader_path/"):
            continue  # already rewritten by us
        if not (
            is_external(dep) or dep.startswith(("@rpath/", "@executable_path/"))
        ):
            continue  # system library -- leave as-is
        real = resolve(dep, origin_file)
        if real is None:
            raise RuntimeError(f"UNRESOLVED dependency {dep!r} of {f}")
        base = os.path.basename(real)
        if base in copied:
            if copied[base][1] != real:
                collisions.append((base, copied[base][1], real))
        else:
            dst = os.path.join(LIB, base)
            shutil.copy2(real, dst)
            os.chmod(dst, 0o644)
            copied[base] = (dst, real)
            worklist.append(base)
        run(
            ["install_name_tool", "-change", dep, reference_form(base, kind), f]
        )


def verify():
    problems = []
    names = {os.path.basename(d) for d, _ in copied.values()}
    for f in [BIN] + [d for d, _ in copied.values()]:
        for dep in load_deps(f):
            if is_external(dep):
                problems.append(f"external ref {dep} in {os.path.basename(f)}")
            elif (
                dep.startswith(
                    ("@rpath/", "@executable_path/", "@loader_path/")
                )
                and os.path.basename(dep) not in names
            ):
                problems.append(f"unbundled {dep} in {os.path.basename(f)}")
        for rp in rpaths(f):
            if is_external(rp):
                problems.append(
                    f"external LC_RPATH {rp} in {os.path.basename(f)}"
                )
    return problems


def main():
    shutil.rmtree(LIB, ignore_errors=True)
    os.makedirs(LIB)
    shutil.copy2(SRC_BIN, BIN)
    os.chmod(BIN, 0o755)

    process(BIN, SRC_BIN, "bin")
    while worklist:
        base = worklist.pop()
        dst, origin = copied[base]
        run(["install_name_tool", "-id", "@rpath/" + base, dst])
        process(dst, origin, "lib")

    # Force resolution through the bundle only: drop every LC_RPATH so a stale
    # /opt/homebrew rpath can't mask a missing dependency on a clean machine.
    for f in [BIN] + [d for d, _ in copied.values()]:
        for rp in rpaths(f):
            run(["install_name_tool", "-delete_rpath", rp, f], check=False)

    # Ad-hoc re-sign (mandatory on arm64 after editing a Mach-O): libs first.
    for f in [d for d, _ in copied.values()] + [BIN]:
        run(["codesign", "--remove-signature", f], check=False)
        run(["codesign", "-f", "-s", "-", f])

    if collisions:
        print("WARNING: basename collisions (distinct libs, same name):")
        for b, a, c in collisions:
            print(f"  {b}: {a} VS {c}")

    problems = verify()
    print("relocated dylibs:", len(copied))
    print("libs dir size:", run(["du", "-sh", LIB]).split()[0])
    if problems:
        print("SELF-CONTAINMENT CHECK FAILED:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print(
        "self-containment: OK (only @loader_path + system libs; no external "
        "refs or rpaths)"
    )


if __name__ == "__main__":
    main()
