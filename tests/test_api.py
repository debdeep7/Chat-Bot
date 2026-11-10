"""Contract tests for the five endpoints, written against the testing brief."""


def push(client, scope="merchant", context_id="m_1", version=1, payload=None):
    return client.post(
        "/v1/context",
        json={
            "scope": scope,
            "context_id": context_id,
            "version": version,
            "payload": {"id": context_id} if payload is None else payload,
            "delivered_at": "2026-04-29T10:00:00Z",
        },
    )


# ---------- healthz / metadata ----------

def test_healthz_starts_empty_with_every_scope_listed(client):
    response = client.get("/v1/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["uptime_seconds"], int)
    assert body["contexts_loaded"] == {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}


def test_healthz_answers_head_requests(client):
    assert client.head("/v1/healthz").status_code == 200


def test_metadata_has_the_fields_the_leaderboard_reads(client):
    body = client.get("/v1/metadata").json()
    required = {"team_name", "team_members", "model", "approach", "contact_email", "version", "submitted_at"}
    assert required <= set(body)
    assert isinstance(body["team_members"], list)


# ---------- /v1/context ----------

def test_context_accepts_every_scope_and_healthz_counts_them(client):
    for scope in ("category", "merchant", "customer", "trigger"):
        response = push(client, scope=scope, context_id=f"{scope}_1")
        assert response.status_code == 200
        body = response.json()
        assert body["accepted"] is True
        assert body["ack_id"] and body["stored_at"]
    counts = client.get("/v1/healthz").json()["contexts_loaded"]
    assert counts == {"category": 1, "merchant": 1, "customer": 1, "trigger": 1}


def test_reposting_the_same_version_is_accepted_but_changes_nothing(app, client):
    push(client, version=1, payload={"v": "first"})
    again = push(client, version=1, payload={"v": "second"})
    assert again.status_code == 200 and again.json()["accepted"] is True
    assert app.state.store.get("merchant", "m_1") == {"v": "first"}


def test_higher_version_replaces_the_payload(app, client):
    push(client, version=1, payload={"v": 1})
    assert push(client, version=2, payload={"v": 2}).status_code == 200
    assert app.state.store.get("merchant", "m_1") == {"v": 2}


def test_lower_version_gets_409_with_the_current_version(app, client):
    push(client, version=5, payload={"v": 5})
    response = push(client, version=3, payload={"v": 3})
    assert response.status_code == 409
    assert response.json() == {"accepted": False, "reason": "stale_version", "current_version": 5}
    assert app.state.store.get("merchant", "m_1") == {"v": 5}


def test_unknown_scope_gets_400_and_nothing_is_stored(app, client):
    response = push(client, scope="warehouse")
    assert response.status_code == 400
    body = response.json()
    assert body["accepted"] is False and body["reason"] == "invalid_scope"
    assert sum(app.state.store.counts().values()) == 0


def test_missing_payload_gets_400(client):
    response = client.post("/v1/context", json={"scope": "merchant", "context_id": "m_1", "version": 1})
    assert response.status_code == 400
    assert response.json()["accepted"] is False


def test_non_json_body_gets_400(client):
    response = client.post("/v1/context", content="not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["accepted"] is False


def test_rupee_sign_and_hindi_survive_the_round_trip(app, client):
    payload = {"title": "Dental Cleaning @ ₹299", "note": "aapke liye 2 slots ready hain"}
    push(client, payload=payload)
    assert app.state.store.get("merchant", "m_1") == payload


# ---------- /v1/tick ----------

def test_tick_with_nothing_loaded_returns_an_empty_action_list(client):
    response = client.post("/v1/tick", json={"now": "2026-04-29T10:30:00Z", "available_triggers": ["trg_unknown"]})
    assert response.status_code == 200
    assert response.json() == {"actions": []}


def test_tick_tolerates_missing_fields(client):
    response = client.post("/v1/tick", json={})
    assert response.status_code == 200
    assert isinstance(response.json()["actions"], list)


# ---------- /v1/reply ----------

def test_reply_returns_one_of_the_three_valid_actions(client):
    response = client.post(
        "/v1/reply",
        json={
            "conversation_id": "conv_1",
            "merchant_id": "m_1",
            "customer_id": None,
            "from_role": "merchant",
            "message": "Yes, send me the abstract",
            "received_at": "2026-04-29T10:45:00Z",
            "turn_number": 2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["action"] in {"send", "wait", "end"}
    if body["action"] == "wait":
        assert isinstance(body["wait_seconds"], int)
    if body["action"] == "send":
        assert body["body"].strip()
