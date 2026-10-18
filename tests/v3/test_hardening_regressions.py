"""Bugs found by tests/hard (hostile-input fuzzing); each must stay fixed."""

import json
import os
import tempfile

import pytest

from munkres import Certificate, certify, cli, solve, solve_spec


def _cert_dict():
    a = [[4, 1], [2, 5]]
    return json.loads(certify(a, solve(a).pairs).to_json())


@pytest.mark.parametrize(
    "key,value",
    [
        ("u", ["1/0", "2"]),
        ("u", True),
        ("v", [None]),
        ("shape", [1]),
        ("pairs", [[0, "a"]]),
        ("objective", "NaN-ish"),
        ("maximize", "yes"),
        ("meta", []),
        ("problem_hash", 7),
    ],
)
def test_malformed_certificates_raise_value_error_only(key, value):
    d = _cert_dict()
    d[key] = value
    with pytest.raises(ValueError):
        Certificate.from_json(json.dumps(d))
    for text in ("[]", "null", "{", '{"version": 1}'):
        with pytest.raises(ValueError):
            Certificate.from_json(text)


@pytest.mark.parametrize(
    "spec",
    [
        {"cost": [[0]], "forbidden": True},
        {"cost": [[0]], "forbidden": [[0]]},
        {"cost": [[0]], "forbidden": [["a", 0]]},
        {"cost": [[0]], "verify": {"x": 1}},
        {"cost": [[0]], "limits": 5},
        {"cost": [[0]], "limits": {"max_rows": "x"}},
        {"cost": [[float("nan")]]},
        {"cost": [[0]], "limits": {"bogus": 1}},
    ],
)
def test_bad_specs_raise_value_error_not_type_error(spec):
    with pytest.raises(ValueError):
        solve_spec(spec)


def test_certify_cli_on_an_empty_file_is_a_clean_error(capsys):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "empty.csv")
        open(path, "w").close()
        assert cli.main(["certify", path]) in (1, 2)


def test_every_backend_descriptor_dtype_claim_is_selectable_and_true():
    from fractions import Fraction

    from munkres import _native, backends, select_backend

    for dt in ("int", "float", "int64", "float64", "Fraction", "Decimal"):
        assert select_backend([dt])["available"]
    assert solve(
        [[Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 5), Fraction(1, 7)]]
    ).total == Fraction(1, 3) + Fraction(1, 5)
    native = next(b for b in backends() if b["name"] == "native-kernel")
    if native["available"]:  # the native descriptor must not claim exact types it cannot handle
        assert (
            "Fraction" not in native["dtypes"]
            and _native.solve_dense(1, 1, [[(0, Fraction(1))]]) is None
        )
