# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""
Popular matching (Abraham, Irving, Kavitha, Mehlhorn 2007), strict preference lists.

Applicants rank posts strictly (a post may be missing = unacceptable); an unmatched applicant is worst. M is
popular if no other matching M' is preferred by more applicants than prefer M. A popular matching exists
iff the reduced graph (edges to each applicant's first choice f(a) and best non-f-post s(a), or a private
last-resort post) has an applicant-complete matching; then promoting a matched applicant onto every unmatched
f-post gives one.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

__all__ = ["is_popular", "popular_matching"]


def popular_matching(prefs: Sequence[Sequence[int]]) -> dict[int, int | None] | None:
    """Return a popular matching `{applicant: post or None}` or None if none exists. Posts are ints >= 0."""
    n = len(prefs)
    if any(len(set(p)) != len(p) for p in prefs):
        raise ValueError("preference lists must not repeat posts")
    f = {a: prefs[a][0] for a in range(n) if prefs[a]}
    fposts = set(f.values())
    big = 1 + max((x for p in prefs for x in p), default=-1)
    last = {a: big + a for a in range(n)}  # private last-resort post per applicant
    s = {a: next((x for x in prefs[a] if x not in fposts), last[a]) for a in range(n)}
    first = {a: (f[a] if a in f else last[a]) for a in range(n)}
    adj = {a: sorted({first[a], s[a]}) for a in range(n)}
    match_post: dict[int, int] = {}

    def aug(a: int, seen: set[int]) -> bool:
        for p in adj[a]:
            if p not in seen:
                seen.add(p)
                if p not in match_post or aug(match_post[p], seen):
                    match_post[p] = a
                    return True
        return False

    for a in range(n):
        if not aug(a, set()):
            return None
    for p in sorted(fposts):
        if p not in match_post:
            a = next(a for a in range(n) if f.get(a) == p)
            for q, b in list(match_post.items()):
                if b == a:
                    del match_post[q]
            match_post[p] = a
    out: dict[int, int | None] = dict.fromkeys(range(n))
    for p, a in match_post.items():
        out[a] = None if p >= big else p
    return out


def _rank(prefs: Sequence[Sequence[int]], a: int, post: int | None) -> float:
    if post is None or post not in prefs[a]:
        return float("inf")
    return prefs[a].index(post)


def is_popular(prefs: Sequence[Sequence[int]], matching: dict[int, int | None]) -> bool:
    """Brute force definition check over *all* matchings (exponential: small instances only, <= 6 applicants)."""
    n = len(prefs)
    if n > 6:
        raise ValueError("brute-force popularity check supports at most 6 applicants")
    options = [[None, *p] for p in prefs]
    for choice in itertools.product(*options):
        used = [c for c in choice if c is not None]
        if len(used) != len(set(used)):
            continue
        better = sum(
            _rank(prefs, a, choice[a]) < _rank(prefs, a, matching.get(a)) for a in range(n)
        )
        worse = sum(_rank(prefs, a, choice[a]) > _rank(prefs, a, matching.get(a)) for a in range(n))
        if better > worse:
            return False
    return True
