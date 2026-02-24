-- ============================================
-- AgriConnect Protocol Hardening — DB Migration
-- PostgreSQL 16+
-- ============================================
-- Run AFTER init.sql and audit_trail.sql

SET search_path TO public;

-- ============================================
-- 1. PROTOCOL TRACE LOG
--    Stores structured decision traces for
--    every A2A / MCP / AG-UI pipeline run.
-- ============================================

CREATE TABLE IF NOT EXISTS protocol_trace_log (
    trace_id        VARCHAR(24)  PRIMARY KEY,
    correlation_id  VARCHAR(24)  NOT NULL,
    session_id      VARCHAR(100) DEFAULT '',
    user_id         VARCHAR(100) DEFAULT '',
    steps_json      JSONB        NOT NULL DEFAULT '[]'::jsonb,
    status          VARCHAR(20)  NOT NULL DEFAULT 'in_progress',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at    TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_trace_correlation ON protocol_trace_log(correlation_id);
CREATE INDEX IF NOT EXISTS idx_trace_user        ON protocol_trace_log(user_id);
CREATE INDEX IF NOT EXISTS idx_trace_status      ON protocol_trace_log(status);
CREATE INDEX IF NOT EXISTS idx_trace_created      ON protocol_trace_log(created_at DESC);


-- ============================================
-- 2. PROTOCOL MESSAGE LOG
--    Persistent A2A message log (replaces
--    the in-memory _message_log list).
-- ============================================

CREATE TABLE IF NOT EXISTS protocol_message_log (
    message_id      VARCHAR(24)  PRIMARY KEY,
    correlation_id  VARCHAR(24)  NOT NULL,
    message_type    VARCHAR(20)  NOT NULL,
    sender_id       VARCHAR(100) NOT NULL,
    receiver_id     VARCHAR(100) DEFAULT '',
    intent          VARCHAR(100) NOT NULL DEFAULT '',
    zone            VARCHAR(100) DEFAULT '',
    crop            VARCHAR(100) DEFAULT '',
    priority        INTEGER      DEFAULT 0,
    payload         JSONB        NOT NULL DEFAULT '{}'::jsonb,
    trace_id        VARCHAR(24),
    status          VARCHAR(20)  DEFAULT 'pending',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at    TIMESTAMP WITH TIME ZONE,

    CONSTRAINT fk_trace FOREIGN KEY (trace_id)
        REFERENCES protocol_trace_log(trace_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_msg_correlation ON protocol_message_log(correlation_id);
CREATE INDEX IF NOT EXISTS idx_msg_receiver    ON protocol_message_log(receiver_id, status);
CREATE INDEX IF NOT EXISTS idx_msg_intent      ON protocol_message_log(intent);
CREATE INDEX IF NOT EXISTS idx_msg_created     ON protocol_message_log(created_at DESC);


-- ============================================
-- 3. IDEMPOTENCY KEYS
--    Cross-instance duplicate detection
--    (replaces in-memory _seen_idempotency).
-- ============================================

CREATE TABLE IF NOT EXISTS protocol_idempotency_keys (
    idempotency_key VARCHAR(64)  PRIMARY KEY,
    message_id      VARCHAR(24)  NOT NULL,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Auto-expire old keys after 24h via pg_cron or app-level cleanup
    expires_at      TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '24 hours')
);

CREATE INDEX IF NOT EXISTS idx_idemp_expires ON protocol_idempotency_keys(expires_at);


-- ============================================
-- 4. HANDSHAKE STATES (FSM)
--    Persistent handshake lifecycle.
--    Replaces in-memory _handshakes dict.
-- ============================================

CREATE TABLE IF NOT EXISTS protocol_handshake_states (
    handshake_id    VARCHAR(24)  PRIMARY KEY,
    initiator_id    VARCHAR(100) NOT NULL,
    responder_id    VARCHAR(100) NOT NULL,
    intent          VARCHAR(100) NOT NULL DEFAULT '',
    current_state   VARCHAR(20)  NOT NULL DEFAULT 'proposed',
    turns           INTEGER      DEFAULT 0,
    max_turns       INTEGER      DEFAULT 5,
    payload         JSONB        NOT NULL DEFAULT '{}'::jsonb,
    history         JSONB        NOT NULL DEFAULT '[]'::jsonb,
    timeout_at      TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hs_initiator  ON protocol_handshake_states(initiator_id);
CREATE INDEX IF NOT EXISTS idx_hs_responder  ON protocol_handshake_states(responder_id);
CREATE INDEX IF NOT EXISTS idx_hs_state      ON protocol_handshake_states(current_state);


-- ============================================
-- 5. AGENT REGISTRY (persistent)
--    Replaces in-memory _agents dict.
-- ============================================

CREATE TABLE IF NOT EXISTS protocol_agent_registry (
    agent_id        VARCHAR(24)  PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    description     TEXT         DEFAULT '',
    domain          VARCHAR(50)  NOT NULL DEFAULT 'external',
    intents         JSONB        NOT NULL DEFAULT '[]'::jsonb,
    capabilities    JSONB        NOT NULL DEFAULT '[]'::jsonb,
    zones           JSONB        NOT NULL DEFAULT '[]'::jsonb,
    crops           JSONB        NOT NULL DEFAULT '[]'::jsonb,
    endpoint        VARCHAR(500) DEFAULT '',
    protocol        VARCHAR(20)  DEFAULT 'internal',
    version         VARCHAR(10)  DEFAULT '1.0',
    status          VARCHAR(20)  DEFAULT 'active',
    avg_response_ms INTEGER      DEFAULT 500,
    registered_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_heartbeat  TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_agent_status ON protocol_agent_registry(status);
CREATE INDEX IF NOT EXISTS idx_agent_domain ON protocol_agent_registry(domain);


-- ============================================
-- 6. CONTEXT CACHE METADATA
--    Tracks freshness & invalidation events.
-- ============================================

CREATE TABLE IF NOT EXISTS protocol_context_cache (
    cache_key       VARCHAR(255) PRIMARY KEY,
    user_id         VARCHAR(100) NOT NULL,
    context_data    JSONB        NOT NULL DEFAULT '{}'::jsonb,
    ttl_seconds     INTEGER      DEFAULT 300,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at      TIMESTAMP WITH TIME ZONE,
    invalidated_at  TIMESTAMP WITH TIME ZONE,
    invalidation_reason VARCHAR(255) DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_cache_user    ON protocol_context_cache(user_id);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON protocol_context_cache(expires_at);


-- ============================================
-- CLEANUP: Auto-expire idempotency keys (optional pg_cron)
-- ============================================
-- If pg_cron is available:
-- SELECT cron.schedule('cleanup_idempotency', '0 * * * *',
--   $$DELETE FROM protocol_idempotency_keys WHERE expires_at < NOW()$$
-- );
