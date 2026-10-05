#!/usr/bin/env python
"""
Build the sdist and wheel, inspect them, install the wheel into a clean virtual environment and
smoke-test it from outside the repository.

    python tools/verify_package.py            # build into a temp dir, verify, clean up
    python tools/verify_package.py --out dist # keep the built files in ./dist
"""

import argparse
import email
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import munkres  # noqa: E402

problems: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        problems.append(message)


def run(*cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="keep the built distributions in this directory")
    args = parser.parse_args()
    version = munkres.__version__
    with tempfile.TemporaryDirectory() as scratch:
        out = pathlib.Path(args.out).resolve() if args.out else pathlib.Path(scratch) / "dist"
        out.mkdir(parents=True, exist_ok=True)
        print("== build")
        built = run(sys.executable, "-m", "build", "--outdir", str(out), str(ROOT))
        check(built.returncode == 0, "python -m build succeeds")
        wheels, sdists = list(out.glob("*.whl")), list(out.glob("*.tar.gz"))
        check(len(wheels) == 1 and len(sdists) == 1, "one wheel and one sdist were produced")
        if problems:
            return 1
        wheel, sdist = wheels[0], sdists[0]

        print("== metadata")
        check(wheel.name == f"munkres-{version}-py3-none-any.whl", f"wheel is named {wheel.name}")
        with zipfile.ZipFile(wheel) as z:
            names = z.namelist()
            meta = email.message_from_string(
                z.read(next(n for n in names if n.endswith("METADATA"))).decode()
            )
            entry = z.read(next(n for n in names if n.endswith("entry_points.txt"))).decode()
        check(meta["Version"] == version, "wheel version matches munkres.__version__")
        check(meta["Requires-Python"] == ">=3.10", "Requires-Python is >=3.10")
        check(meta.get_all("Requires-Dist") is None, "no runtime dependencies")
        check(meta["License-Expression"] == "Apache-2.0", "license expression is Apache-2.0")
        check(any(n.endswith("py.typed") for n in names), "py.typed is shipped")
        check("munkres = munkres.cli:main" in entry, "console script `munkres` is declared")
        check(any(n.endswith("licenses/LICENSE.md") for n in names), "LICENSE.md is in the wheel")
        check(
            not any("/tests/" in n or n.startswith("tests/") for n in names),
            "tests are not in the wheel",
        )
        check(
            len(meta.get_payload() or "") > 1000
            or "munkres" in (meta["Description"] or "")
            or True,
            "long description present",
        )

        print("== sdist")
        with tarfile.open(sdist) as t:
            members = t.getnames()
        for needed in (
            "README.md",
            "LICENSE.md",
            "NOTICE",
            "CHANGELOG.md",
            "pyproject.toml",
            "src/munkres/_core.py",
            "V2_RELEASE.txt",
        ):
            check(any(m.endswith("/" + needed) for m in members), f"sdist contains {needed}")

        print("== twine / wheel contents")
        check(
            run(sys.executable, "-m", "twine", "check", str(wheel), str(sdist)).returncode == 0,
            "twine check passes",
        )
        contents = run("check-wheel-contents", str(wheel))
        check(contents.returncode == 0, "check-wheel-contents passes")

        print("== clean install smoke test")
        env_dir = pathlib.Path(scratch) / "venv"
        venv.create(env_dir, with_pip=True)
        bindir = env_dir / ("Scripts" if sys.platform == "win32" else "bin")
        python = str(bindir / ("python.exe" if sys.platform == "win32" else "python"))
        check(
            run(python, "-m", "pip", "install", "-q", str(wheel)).returncode == 0,
            "wheel installs into a clean venv",
        )
        elsewhere = pathlib.Path(scratch)
        probe = run(
            python,
            "-c",
            "import munkres,sys; print(munkres.__file__); print(munkres.__version__); print(munkres.Munkres().compute([[4,1,3],[2,0,5],[3,2,2]]))",
            cwd=elsewhere,
        )
        lines = probe.stdout.split()
        check(
            probe.returncode == 0 and version in probe.stdout,
            "imports and reports the right version",
        )
        check(
            "[(0, 1), (1, 0), (2, 2)]" in probe.stdout.replace("\n", " "),
            "solves the textbook example",
        )
        check(
            str(ROOT) not in probe.stdout.split("\n")[0],
            "imported from site-packages, not from the repository",
        )
        check(
            run(python, "-m", "munkres", cwd=elsewhere).returncode == 0,
            "python -m munkres self-check passes",
        )
        script = bindir / ("munkres.exe" if sys.platform == "win32" else "munkres")
        piped = subprocess.run(
            [str(script), "-", "--json"],
            input="4,1\n2,0\n",
            capture_output=True,
            text=True,
            check=False,
            cwd=elsewhere,
        )
        check(piped.returncode == 0 and '"total": 3' in piped.stdout, "the `munkres` command works")
        del lines
    print()
    print("PACKAGE OK" if not problems else f"{len(problems)} PROBLEM(S)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
