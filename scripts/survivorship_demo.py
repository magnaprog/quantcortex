"""Show S&P 500 membership changes and current-symbol price responses.

The PIT membership reconstruction (quantcortex/data/universe/sp500_wikipedia)
identifies names that were in the index on a past date but are absent today.
The script counts those names and checks whether current-symbol requests return
any usable prices for them. This does not diagnose missing prices or quantify
survivorship bias.

    python scripts/survivorship_demo.py --live-yfinance
    python scripts/survivorship_demo.py 2015-06-01 --live-yfinance

Requires network + lxml (Wikipedia) and yfinance (pricing the dropped names).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

# yfinance reports failed downloads via logging + stderr; quiet it so
# the summary of returned prices is readable.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
YFINANCE_NOTICE = (
    "Live Yahoo Finance data is fetched through yfinance. Review Yahoo's terms "
    "and yfinance's legal disclaimer at https://ranaroussi.github.io/yfinance/."
)


def main(argv) -> int:
    parser = argparse.ArgumentParser(
        description="show S&P 500 membership changes and current-symbol price availability"
    )
    parser.add_argument("as_of", nargs="?", default="2018-06-01")
    parser.add_argument(
        "--live-yfinance",
        action="store_true",
        required=True,
        help="explicitly permit this run to fetch live data through yfinance",
    )
    args = parser.parse_args(argv[1:])
    as_of = args.as_of
    print(YFINANCE_NOTICE, file=sys.stderr)
    try:
        from quantcortex.data.universe.sp500_universe import SP500Universe
    except Exception as exc:  # pragma: no cover
        print(f"import error: {exc}")
        return 1

    try:
        uni = SP500Universe.from_wikipedia()
    except Exception as exc:
        print(f"could not build the point-in-time universe (need network + lxml): {exc}")
        return 1

    past = set(uni.constituents(as_of))
    today = set(uni.constituents())
    dropped = sorted(past - today)
    added = sorted(today - past)

    print("S&P 500 point-in-time membership (Wikipedia reconstruction)")
    print("=" * 70)
    print(f"  members as of {as_of}: {len(past)}")
    print(f"  members today:          {len(today)}")
    print(f"  in {as_of[:4]} but gone today (dropped): {len(dropped)}")
    print(f"  added since {as_of[:4]}:                  {len(added)}")

    # Using today's members for all history omits the dropped names. Check
    # whether current-symbol requests return any usable prices for those names.
    try:
        from quantcortex.data.providers.yfinance_provider import YFinanceProvider

        provider = YFinanceProvider()
        provider_end = (date.today() + timedelta(days=1)).isoformat()
        px = provider.get_prices(dropped, start=as_of, end=provider_end)
        cols = list(getattr(px, "columns", []))
        # "Priceable" requires a returned column with at least one non-NaN value.
        # This describes the response, not the security's status.
        priceable = [s for s in dropped if s in cols and px[s].notna().any()]
        unpriceable = sorted(set(dropped) - set(priceable))
        print(f"\n  of the {len(dropped)} dropped names, {len(priceable)} returned")
        print(f"  at least one price and {len(unpriceable)} returned no usable prices.")
        print("  Missing prices do not identify delistings, mergers, or renamings.")
        print(f"  examples with no prices returned in this run: {unpriceable[:12]}")
    except Exception as exc:
        print(f"pricing step failed: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print("Takeaway: building a backtest universe from today's constituents")
    print(f"omits {len(dropped)} names that were members as of {as_of}.")
    print("Use SP500Universe.from_wikipedia() (or a licensed PIT feed) so the")
    print("universe is queried as-of each rebalance date instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
