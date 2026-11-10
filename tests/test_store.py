import threading

import pytest

from vera.store import SCOPES, ContextStore


def test_put_then_get():
    store = ContextStore()
    result = store.put("merchant", "m_1", 1, {"name": "Dr. Meera"})
    assert (result.outcome, result.version) == ("stored", 1)
    assert store.get("merchant", "m_1") == {"name": "Dr. Meera"}
    assert store.version("merchant", "m_1") == 1


def test_missing_entries_return_none():
    store = ContextStore()
    assert store.get("merchant", "nope") is None
    assert store.version("merchant", "nope") is None
    assert store.get("not-a-scope", "x") is None


def test_same_version_is_a_noop_and_keeps_the_first_payload():
    store = ContextStore()
    store.put("merchant", "m_1", 2, {"v": "first"})
    result = store.put("merchant", "m_1", 2, {"v": "second"})
    assert (result.outcome, result.version) == ("noop", 2)
    assert store.get("merchant", "m_1") == {"v": "first"}


def test_higher_version_replaces():
    store = ContextStore()
    store.put("merchant", "m_1", 1, {"v": 1})
    result = store.put("merchant", "m_1", 3, {"v": 3})
    assert (result.outcome, result.version) == ("stored", 3)
    assert store.get("merchant", "m_1") == {"v": 3}


def test_lower_version_is_stale_and_changes_nothing():
    store = ContextStore()
    store.put("merchant", "m_1", 5, {"v": 5})
    result = store.put("merchant", "m_1", 4, {"v": 4})
    assert (result.outcome, result.version) == ("stale", 5)
    assert store.get("merchant", "m_1") == {"v": 5}


def test_same_id_in_different_scopes_does_not_collide():
    store = ContextStore()
    store.put("merchant", "x", 1, {"kind": "merchant"})
    store.put("customer", "x", 1, {"kind": "customer"})
    assert store.get("merchant", "x") == {"kind": "merchant"}
    assert store.get("customer", "x") == {"kind": "customer"}


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError):
        ContextStore().put("bogus", "x", 1, {})


def test_counts_and_sorted_ids():
    store = ContextStore()
    for cid in ("m_b", "m_c", "m_a"):
        store.put("merchant", cid, 1, {})
    store.put("category", "dentists", 1, {})
    assert store.counts() == {"category": 1, "merchant": 3, "customer": 0, "trigger": 0}
    assert store.ids("merchant") == ["m_a", "m_b", "m_c"]
    assert set(store.counts()) == set(SCOPES)


def test_clear_empties_every_scope():
    store = ContextStore()
    store.put("merchant", "m_1", 1, {})
    store.put("trigger", "t_1", 1, {})
    store.clear()
    assert sum(store.counts().values()) == 0


def test_concurrent_pushes_end_on_the_highest_version():
    store = ContextStore()

    def worker(offset: int) -> None:
        for version in range(1 + offset, 401, 8):
            store.put("merchant", "m_1", version, {"v": version})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.version("merchant", "m_1") == 400
    assert store.get("merchant", "m_1") == {"v": 400}
