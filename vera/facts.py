"""Turn raw context values into small, quotable facts.

A `Fact` is one sentence-sized piece of evidence that is already present in the contexts
the judge pushed. The composer may quote a fact; the validator (a later step) rejects any
number in a message that no context supports. Keeping every claim behind a fact is how the
bot stays grounded instead of inventing details.

Data conventions this module hides from the rest of the code:
  * every ratio is stored as a fraction (0.021 = 2.1%, -0.5 = down 50%)
  * only offers with status "active" may be claimed as the merchant's own
  * a percentage swing on a tiny base (2 calls -> 3 calls) is noise, so it is ignored
  * ISO timestamps are formatted as written; time zones are never shifted
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def numbers_in(text: str) -> frozenset[str]:
    """Every number written in `text`, as plain digits ('2,410' -> '2410')."""
    return frozenset(m.group().replace(",", "") for m in _NUMBER.finditer(text))


@dataclass(frozen=True)
class Fact:
    key: str   # stable id such as "peer.ctr"; lets tests and the validator point at a fact
    text: str  # a fragment ready to quote, e.g. "calls down 50% over the last 7 days"

    @property
    def numbers(self) -> frozenset[str]:
        return numbers_in(self.text)


# --------------------------------------------------------------------------- formatting

def num(n: float) -> str:
    """2410 -> '2,410'; 1.3 -> '1.3'; 12.0 -> '12'."""
    n = float(n)
    return f"{n:,.0f}" if n.is_integer() else f"{n:,.1f}".rstrip("0").rstrip(".")


def money(amount: float) -> str:
    return "₹" + num(amount)


def pct(fraction: float, signed: bool = False) -> str:
    """0.18 -> '18%'; 0.021 -> '2.1%'. With signed=True: '+18%' / '-5%'."""
    value = fraction * 100
    magnitude = abs(value)
    body = f"{magnitude:.0f}" if magnitude >= 10 else f"{magnitude:.1f}".rstrip("0").rstrip(".")
    if not signed:
        return f"{body or '0'}%"
    return ("-" if value < 0 else "+") + f"{body or '0'}%"


def words(snake: Any) -> str:
    """'delivery_late' -> 'delivery late'; '6_month_cleaning' -> '6-month cleaning'."""
    text = re.sub(r"[_\s]+", " ", str(snake)).strip()
    return re.sub(r"\b(\d+) (day|week|month|year)\b", r"\1-\2", text)


def when(iso: Any) -> str | None:
    """'2026-05-02T19:00:00+05:30' -> '2 May, 7pm'; '2026-05-02' -> '2 May'; junk -> None.

    A timestamp at exactly midnight is treated as a plain date ("28 Apr", not "28 Apr, 12am").
    """
    if not iso:
        return None
    text = str(iso)
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    day = f"{moment.day} {moment.strftime('%b')}"
    if "T" not in text or (moment.hour == 0 and moment.minute == 0):
        return day
    hour = moment.hour % 12 or 12
    minute = f":{moment.minute:02d}" if moment.minute else ""
    return f"{day}, {hour}{minute}{'am' if moment.hour < 12 else 'pm'}"


def months_in(label: str) -> set[str]:
    """'Oct-Dec' -> {'oct','nov','dec'}; 'Jan' -> {'jan'}; 'Feb 14' -> {'feb'}."""
    parts = [p.strip()[:3].lower() for p in re.split(r"\s*-\s*", label) if p.strip()]
    parts = [p for p in parts if p in _MONTHS]
    if len(parts) == 2:
        start, end = _MONTHS.index(parts[0]), _MONTHS.index(parts[1])
        span = (end - start) % 12
        return {_MONTHS[(start + i) % 12] for i in range(span + 1)}
    return set(parts)


# --------------------------------------------------------------------------- merchant

def peer_label(category: dict) -> str:
    """'metro_solo_practices_2026' -> 'metro solo practices'."""
    scope = str((category.get("peer_stats") or {}).get("scope") or "similar merchants")
    return words(re.sub(r"_?\d{4}$", "", scope))


def perf_snapshot(merchant: dict) -> Fact | None:
    p = merchant.get("performance") or {}
    parts = [
        f"{num(p[k])} {label}"
        for k, label in (("views", "views"), ("calls", "calls"), ("directions", "direction requests"))
        if p.get(k) is not None
    ]
    if not parts:
        return None
    return Fact("perf.snapshot", f"last {p.get('window_days') or 30} days: " + ", ".join(parts))


def peer_comparisons(merchant: dict, category: dict) -> list[tuple[float, Fact]]:
    """(relative gap vs peers, fact) for CTR and for calls. The gap is negative below peers."""
    perf, peer = merchant.get("performance") or {}, category.get("peer_stats") or {}
    label = peer_label(category)
    out = []
    for key, mine, theirs, text in (
        ("peer.ctr", perf.get("ctr"), peer.get("avg_ctr"), "CTR {mine} vs {theirs} for {label} ({side} peers)"),
        ("peer.calls", perf.get("calls"), peer.get("avg_calls_30d"), "{mine} calls in 30 days vs {theirs} for {label} ({side} peers)"),
    ):
        if mine is None or not theirs:
            continue
        side = "below" if mine < theirs else "above" if mine > theirs else "in line with"
        show = pct if key == "peer.ctr" else num
        out.append(((mine - theirs) / theirs, Fact(key, text.format(mine=show(mine), theirs=show(theirs), label=label, side=side))))
    return out


def _peer_fact(merchant: dict, category: dict, key: str) -> Fact | None:
    return next((fact for _, fact in peer_comparisons(merchant, category) if fact.key == key), None)


def ctr_vs_peer(merchant: dict, category: dict) -> Fact | None:
    return _peer_fact(merchant, category, "peer.ctr")


def calls_vs_peer(merchant: dict, category: dict) -> Fact | None:
    return _peer_fact(merchant, category, "peer.calls")


# A percentage swing only means something above a minimum base (2 calls -> 3 calls is +50%).
_MIN_BASE = {"views_pct": ("views", 100), "calls_pct": ("calls", 8)}
_DELTA_LABEL = {"views_pct": "views", "calls_pct": "calls", "ctr_pct": "CTR"}


def moves(merchant: dict) -> list[tuple[str, float]]:
    """(metric label, 7-day change as a fraction) for every change worth talking about."""
    perf = merchant.get("performance") or {}
    out = []
    for key, label in _DELTA_LABEL.items():
        value = (perf.get("delta_7d") or {}).get(key)
        if value is None:
            continue
        guard = _MIN_BASE.get(key)
        if guard and (perf.get(guard[0]) or 0) < guard[1]:
            continue
        out.append((label, float(value)))
    return out


def move_fact(merchant: dict, direction: str, minimum: float = 0.05) -> Fact | None:
    """The biggest 7-day move in one direction ('down' or 'up'), if it is at least `minimum`."""
    sign = -1 if direction == "down" else 1
    candidates = [(label, v) for label, v in moves(merchant) if v * sign > 0 and abs(v) >= minimum]
    if not candidates:
        return None
    label, value = max(candidates, key=lambda item: abs(item[1]))
    return Fact(f"perf.{label.lower()}_{direction}", f"{label} {direction} {pct(value)} over the last 7 days")


def active_offers(merchant: dict) -> list[dict]:
    return [o for o in merchant.get("offers") or [] if o.get("status") == "active" and o.get("title")]


def offer_fact(merchant: dict) -> Fact | None:
    offers = active_offers(merchant)
    if not offers:
        return None
    return Fact("offer.active", "active offer: " + "; ".join(o["title"] for o in offers[:2]))


def catalog_offer(
    category: dict, audience: str | None = None, match: Iterable[str] = (), require_match: bool = False,
    exclude: Iterable[str] = (),
) -> Fact | None:
    """A catalogue offer the merchant could run. A suggestion only, never the merchant's own claim.

    `match` ranks offers whose title shares a word with it; with `require_match` an offer that
    shares none is not returned at all (better no suggestion than an irrelevant one).
    `exclude` lists titles to leave out, normally the offers the merchant already runs.
    """
    wanted = {w.lower() for w in match}
    skip = {title.lower() for title in exclude}
    pool = [
        o for o in category.get("offer_catalog") or []
        if o.get("title") and o["title"].lower() not in skip and (audience is None or o.get("audience") in (audience, "all"))
    ]
    if not pool:
        return None
    overlap = lambda o: len(wanted & set(re.findall(r"[a-z]+", o["title"].lower())))  # noqa: E731
    pool.sort(key=lambda o: -overlap(o))
    if require_match and wanted and overlap(pool[0]) == 0:
        return None
    return Fact("offer.catalog", f"offer you could run (from the catalogue): {pool[0]['title']}")


def subscription_fact(merchant: dict) -> Fact | None:
    sub = merchant.get("subscription") or {}
    status, plan = sub.get("status"), sub.get("plan")
    left, since = sub.get("days_remaining"), sub.get("days_since_expiry")
    if status == "expired":
        when_text = f" {num(since)} days ago" if since is not None else ""
        return Fact("sub.status", f"{plan or 'plan'} subscription expired{when_text}")
    if left is None:
        return None
    if status == "trial":
        return Fact("sub.status", f"free trial: {num(left)} days left")
    return Fact("sub.status", f"{plan or 'plan'} subscription: {num(left)} days remaining")


_AGGREGATE_TEXT = [
    ("high_risk_adult_count", "{n} high-risk adult patients on your records", False),
    ("chronic_rx_count", "{n} chronic-prescription customers on your records", False),
    ("lapsed_180d_plus", "{n} customers who haven't visited in 180+ days", False),
    ("lapsed_90d_plus", "{n} customers who haven't visited in 90+ days", False),
    ("total_active_members", "{n} active members", False),
    ("retention_6mo_pct", "6-month customer retention {n}", True),
    ("retention_3mo_pct", "3-month customer retention {n}", True),
    ("repeat_customer_pct", "{n} of customers are repeat customers", True),
    ("monthly_churn_pct", "{n} monthly member churn", True),
    ("total_unique_ytd", "{n} unique customers this year", False),
]


def aggregate_facts(merchant: dict) -> list[Fact]:
    agg = merchant.get("customer_aggregate") or {}
    out = []
    for key, template, is_ratio in _AGGREGATE_TEXT:
        if agg.get(key) is not None:
            out.append(Fact(f"agg.{key}", template.format(n=pct(agg[key]) if is_ratio else num(agg[key]))))
    return out


def review_facts(merchant: dict) -> list[Fact]:
    """Review themes, positives first (strengths to build on), then complaints."""
    themes = sorted(merchant.get("review_themes") or [], key=lambda t: t.get("sentiment") != "pos")
    out = []
    for t in themes:
        quote = f' ("{t["common_quote"]}")' if t.get("common_quote") else ""
        kind = "praised for" if t.get("sentiment") == "pos" else "complaints about"
        out.append(Fact(f"review.{t.get('theme')}", f"{t.get('occurrences_30d', 'some')} reviews in 30 days: {kind} {words(t.get('theme'))}{quote}"))
    return out


_STALE_POSTS = re.compile(r"stale_posts:(\d+)d")
_DORMANT = re.compile(r"dormant_with_vera_(\d+)d")


def signal_facts(merchant: dict) -> list[Fact]:
    """The few profile signals worth quoting; the rest stay available in the raw context."""
    out = []
    for signal in merchant.get("signals") or []:
        stale, dormant = _STALE_POSTS.fullmatch(str(signal)), _DORMANT.fullmatch(str(signal))
        if stale:
            out.append(Fact("signal.stale_posts", f"last Google post was {stale.group(1)} days ago"))
        elif dormant:
            out.append(Fact("signal.dormant", f"{dormant.group(1)} days since the last conversation with Vera"))
        elif signal == "no_active_offers":
            out.append(Fact("signal.no_offers", "no active offer is running right now"))
    return out


# --------------------------------------------------------------------------- category

def find_digest(category: dict, item_id: Any) -> dict | None:
    for item in category.get("digest") or []:
        if item.get("id") == item_id:
            return item
    return None


def digest_of_kind(category: dict, kind: str) -> dict | None:
    """First digest item of a given kind ('compliance', 'cde', 'alert', 'seasonal', ...)."""
    return next((i for i in category.get("digest") or [] if i.get("kind") == kind), None)


def search_digest(category: dict, needle: str) -> dict | None:
    """First digest item whose title or summary mentions `needle` (case-insensitive)."""
    needle = needle.lower()
    for item in category.get("digest") or []:
        if needle in f"{item.get('title', '')} {item.get('summary', '')}".lower():
            return item
    return None


def pick_digest(category: dict, merchant: dict, avoid_kinds: Iterable[str] = ()) -> dict | None:
    """Most relevant digest item for this merchant: word overlap with their offers, signals,
    review themes and customer-aggregate keys. Ties go to the earlier (more recent) item."""
    bag: set[str] = set()
    for source in (
        [o.get("title", "") for o in merchant.get("offers") or []],
        [str(s) for s in merchant.get("signals") or []],
        [str(t.get("theme", "")) for t in merchant.get("review_themes") or []],
        list((merchant.get("customer_aggregate") or {}).keys()),
    ):
        for text in source:
            bag.update(w for w in re.findall(r"[a-z]{4,}", text.lower()))
    best, best_score = None, -1
    for item in category.get("digest") or []:
        if item.get("kind") in set(avoid_kinds):
            continue
        text = " ".join(str(item.get(k, "")) for k in ("title", "summary", "patient_segment", "actionable")).lower()
        score = len(bag & set(re.findall(r"[a-z]{4,}", text)))
        if score > best_score:
            best, best_score = item, score
    return best


def digest_fact(item: dict, key: str = "digest.item") -> Fact:
    detail = f". {item['summary']}" if item.get("summary") else ""
    return Fact(key, f"{item.get('title', 'update')} (source: {item.get('source', 'category digest')}){detail}")


def digest_action(item: dict) -> Fact | None:
    if not item.get("actionable"):
        return None
    return Fact("digest.action", f"suggested action: {item['actionable']}")


def beat_for(category: dict, months: Iterable[str] = (), tokens: Iterable[str] = ()) -> Fact | None:
    """A seasonal beat matching the given months (e.g. {'oct'}) or words in its note."""
    months, tokens = {m.lower() for m in months}, {t.lower() for t in tokens}
    best, best_score = None, 0
    for beat in category.get("seasonal_beats") or []:
        score = 2 * len(months & months_in(beat.get("month_range", "")))
        score += len(tokens & set(re.findall(r"[a-z]{3,}", f"{beat.get('month_range', '')} {beat.get('note', '')}".lower())))
        if score > best_score:
            best, best_score = beat, score
    if best is None:
        return None
    return Fact("category.season", f"{best['month_range']}: {best['note']}")


# --------------------------------------------------------------------------- customer

def customer_facts(customer: dict | None) -> list[Fact]:
    """What the merchant knows about this customer, most useful first."""
    if not customer:
        return []
    rel = customer.get("relationship") or {}
    prefs = customer.get("preferences") or {}
    out = []
    if when(rel.get("last_visit")):
        out.append(Fact("cust.last_visit", f"last visit: {when(rel['last_visit'])}"))
    if rel.get("visits_total"):
        visits = rel["visits_total"]
        out.append(Fact("cust.visits", f"{num(visits)} visit{'' if visits == 1 else 's'} so far"))
    services = [words(s) for s in rel.get("services_received") or [] if s and s != "..."]
    if services:
        out.append(Fact("cust.services", "past services: " + ", ".join(services[:3])))
    for key, label in (
        ("preferred_slots", "prefers {} slots"), ("preferred_stylist", "prefers stylist {}"),
        ("training_focus", "training focus: {}"), ("health_focus", "health focus: {}"),
        ("favourite_dish", "favourite dish: {}"),
    ):
        value = prefs.get(key) or rel.get(key)
        if value:
            out.append(Fact(f"cust.{key}", label.format(words(value))))
    return out
