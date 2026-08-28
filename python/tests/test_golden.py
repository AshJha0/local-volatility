"""Golden-value suite: the Python reference must reproduce data/golden/golden.json.

Case-name prefixes map to engines (the same dispatch every language port uses):
  flat_dupire_*      -> Dupire local vol on data/flat_surface.csv
  pde_flat_*         -> flat-vol European PDE price
  dupire_bundled_*   -> Dupire local vol on data/implied_surface.csv
  pde_localvol_*     -> local-vol European PDE on the bundled surface
  american_put_*     -> American put PSOR PDE (reference: CRR binomial 5000)
  mc_flat_*          -> flat-vol MC vanilla (statistical tolerance)
  mc_localvol_*      -> local-vol MC vanilla (statistical tolerance)
  barrier_upout_*    -> up-and-out call MC with Brownian bridge (statistical)
"""

from __future__ import annotations

import json

import pytest

from localvol import (
    DupireLocalVol,
    Market,
    price_american_put_pde,
    price_european_mc,
    price_european_pde,
    price_up_out_call_mc,
)

SEED = 20260827  # any fixed seed is allowed for the statistical cases


def _load_cases(data_dir):
    with open(data_dir / "golden" / "golden.json") as f:
        payload = json.load(f)
    return payload["cases"]


def test_golden_file_shape(data_dir):
    cases = _load_cases(data_dir)
    assert len(cases) == 15
    for c in cases:
        assert set(c) == {"name", "inputs", "expect", "tol"}
        for v in list(c["inputs"].values()) + list(c["expect"].values()):
            assert isinstance(v, (int, float)), f"{c['name']}: non-scalar golden value"


def test_all_golden_cases(data_dir, flat_surface, bundled_surface, bundled_localvol):
    checked = 0
    for c in _load_cases(data_dir):
        name, inp, exp, tol = c["name"], c["inputs"], c["expect"], c["tol"]
        if name.startswith("flat_dupire_"):
            got = float(DupireLocalVol(flat_surface).vol(inp["k"], inp["T"]))
            assert got == pytest.approx(exp["local_vol"], abs=tol), name
        elif name.startswith("dupire_bundled_"):
            got = float(bundled_localvol.vol(inp["k"], inp["T"]))
            assert got == pytest.approx(exp["local_vol"], abs=tol), name
        elif name.startswith("pde_flat_"):
            mkt = Market(inp["s"], inp["r"], inp["q"])
            got = price_european_pde(
                mkt, inp["k"], inp["T"], inp["sigma"], is_call=bool(inp["call"]),
                num_space=int(inp["num_space"]), num_time=int(inp["num_time"]),
            )
            assert got == pytest.approx(exp["price"], abs=tol), name
        elif name.startswith("pde_localvol_"):
            mkt = Market(inp["s"], inp["r"], inp["q"])
            got = price_european_pde(
                mkt, inp["k"], inp["T"], bundled_localvol.vol,
                num_space=int(inp["num_space"]), num_time=int(inp["num_time"]),
                sigma_ref=inp["sigma_ref"],
            )
            assert got == pytest.approx(exp["price"], abs=tol), name
        elif name.startswith("american_put_"):
            mkt = Market(inp["s"], inp["r"], inp["q"])
            got = price_american_put_pde(
                mkt, inp["k"], inp["T"], inp["sigma"],
                num_space=int(inp["num_space"]), num_time=int(inp["num_time"]),
            )
            assert got == pytest.approx(exp["price"], abs=tol), name
        elif name.startswith("mc_flat_"):
            mkt = Market(inp["s"], inp["r"], inp["q"])
            res = price_european_mc(
                mkt, inp["k"], inp["T"], inp["sigma"],
                n_paths=int(inp["n_paths"]), n_steps=int(inp["n_steps"]), seed=SEED,
            )
            assert res.price == pytest.approx(exp["price"], abs=tol), name
        elif name.startswith("mc_localvol_"):
            mkt = Market(inp["s"], inp["r"], inp["q"])
            res = price_european_mc(
                mkt, inp["k"], inp["T"], bundled_localvol.vol,
                n_paths=int(inp["n_paths"]), n_steps=int(inp["n_steps"]), seed=SEED,
            )
            assert res.price == pytest.approx(exp["price"], abs=tol), name
        elif name.startswith("barrier_upout_"):
            mkt = Market(inp["s"], inp["r"], inp["q"])
            res = price_up_out_call_mc(
                mkt, inp["k"], inp["b"], inp["T"], inp["sigma"],
                n_paths=int(inp["n_paths"]), n_steps=int(inp["n_steps"]),
                seed=SEED, brownian_bridge=True,
            )
            assert res.price == pytest.approx(exp["price"], abs=tol), name
        else:
            pytest.fail(f"unrecognised golden case name: {name}")
        checked += 1
    assert checked == 15
