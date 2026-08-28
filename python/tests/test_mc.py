"""Monte Carlo: agreement with BS/PDE, antithetic effect, barrier properties."""

from __future__ import annotations

import pytest

from localvol import (
    Market,
    bs_price,
    price_european_mc,
    price_european_pde,
    price_up_out_call_mc,
)

SEED = 987


def test_flat_mc_matches_bs_within_3se():
    mkt = Market(100.0, 0.03, 0.01)
    exact = bs_price(100, 105, 0.03, 0.01, 0.2, 1.0, True)
    res = price_european_mc(mkt, 105.0, 1.0, 0.2, n_paths=20000, n_steps=50, seed=SEED)
    assert res.stderr > 0.0
    assert res.within(exact, 3.0), f"{res.price}±{res.stderr} vs {exact}"


def test_flat_mc_put_and_fx():
    mkt = Market.fx(1.20, rd=0.03, rf=0.01)
    exact = bs_price(1.20, 1.25, 0.03, 0.01, 0.10, 0.5, False)
    res = price_european_mc(mkt, 1.25, 0.5, 0.10, is_call=False, n_paths=20000, n_steps=50, seed=SEED)
    assert res.within(exact, 3.0)


def test_mc_deterministic_given_seed():
    mkt = Market(100.0, 0.02, 0.0)
    a = price_european_mc(mkt, 100.0, 1.0, 0.2, n_paths=2000, n_steps=20, seed=11)
    b = price_european_mc(mkt, 100.0, 1.0, 0.2, n_paths=2000, n_steps=20, seed=11)
    c = price_european_mc(mkt, 100.0, 1.0, 0.2, n_paths=2000, n_steps=20, seed=12)
    assert a.price == b.price and a.stderr == b.stderr
    assert a.price != c.price


def test_antithetic_reduces_stderr():
    mkt = Market(100.0, 0.02, 0.0)
    anti = price_european_mc(mkt, 100.0, 1.0, 0.2, n_paths=20000, n_steps=25, seed=SEED, antithetic=True)
    plain = price_european_mc(mkt, 100.0, 1.0, 0.2, n_paths=20000, n_steps=25, seed=SEED, antithetic=False)
    assert anti.stderr < plain.stderr


def test_localvol_mc_matches_pde_within_3se(bundled_localvol):
    mkt = Market(100.0, 0.0, 0.0)
    pde = price_european_pde(
        mkt, 100.0, 1.0, bundled_localvol.vol, num_space=200, num_time=200, sigma_ref=0.2
    )
    res = price_european_mc(mkt, 100.0, 1.0, bundled_localvol.vol, n_paths=20000, n_steps=100, seed=SEED)
    assert res.within(pde, 3.0), f"MC {res.price}±{res.stderr} vs PDE {pde}"


def test_barrier_below_vanilla_and_bridge_lowers_price(bundled_localvol):
    mkt = Market(100.0, 0.02, 0.0)
    kwargs = dict(n_paths=10000, n_steps=50, seed=SEED)
    vanilla = price_european_mc(mkt, 100.0, 1.0, bundled_localvol.vol, **kwargs)
    plain = price_up_out_call_mc(mkt, 100.0, 125.0, 1.0, bundled_localvol.vol, brownian_bridge=False, **kwargs)
    bridged = price_up_out_call_mc(mkt, 100.0, 125.0, 1.0, bundled_localvol.vol, brownian_bridge=True, **kwargs)
    assert plain.price <= vanilla.price + 1e-12          # knock-out cannot add value
    assert bridged.price <= plain.price + 1e-12          # bridge sees more crossings
    assert bridged.price > 0.0


def test_barrier_born_knocked_out():
    mkt = Market(130.0, 0.02, 0.0)
    res = price_up_out_call_mc(mkt, 100.0, 125.0, 1.0, 0.2, n_paths=1000, n_steps=10, seed=1)
    assert res.price == 0.0 and res.stderr == 0.0


def test_validation():
    mkt = Market(100.0, 0.02, 0.0)
    with pytest.raises(ValueError, match="even"):
        price_european_mc(mkt, 100.0, 1.0, 0.2, n_paths=1001, antithetic=True)
    with pytest.raises(ValueError, match="n_steps"):
        price_european_mc(mkt, 100.0, 1.0, 0.2, n_steps=0)
    with pytest.raises(ValueError, match="expiry"):
        price_european_mc(mkt, 100.0, 0.0, 0.2)
    with pytest.raises(ValueError, match="flat vol"):
        price_european_mc(mkt, 100.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="barrier"):
        price_up_out_call_mc(mkt, 100.0, -5.0, 1.0, 0.2)
    with pytest.raises(ValueError, match="strike"):
        price_european_mc(mkt, -1.0, 1.0, 0.2)
