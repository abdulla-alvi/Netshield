CREATE TABLE IF NOT EXISTS alerts (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    event_time DATETIME(6) NOT NULL,
    prediction VARCHAR(32) NOT NULL,
    confidence DECIMAL(6,5) NOT NULL,
    severity VARCHAR(16) NOT NULL,
    src_ip VARCHAR(45) NULL,
    dst_ip VARCHAR(45) NULL,
    src_port INT NULL,
    dst_port INT NULL,
    ip_proto INT NULL,
    flow_duration DOUBLE NULL,
    packet_rate DOUBLE NULL,
    model_version VARCHAR(64) NOT NULL,
    status VARCHAR(24) NOT NULL DEFAULT 'New',
    assigned_to VARCHAR(128) NULL,
    notes TEXT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    INDEX idx_alerts_event_time (event_time),
    INDEX idx_alerts_prediction (prediction),
    INDEX idx_alerts_severity (severity),
    INDEX idx_alerts_status (status),
    INDEX idx_alerts_confidence (confidence),
    INDEX idx_alerts_dst_port (dst_port),
    INDEX idx_alerts_ip_proto (ip_proto),
    INDEX idx_alerts_src_ip (src_ip),
    INDEX idx_alerts_dst_ip (dst_ip)
);

CREATE TABLE IF NOT EXISTS alert_xai (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    alert_id BIGINT UNSIGNED NOT NULL,
    attention_json JSON NOT NULL,
    top_features_json JSON NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    INDEX idx_alert_xai_alert_id (alert_id),
    CONSTRAINT fk_alert_xai_alert_id
        FOREIGN KEY (alert_id)
        REFERENCES alerts (id)
        ON DELETE CASCADE
        ON UPDATE RESTRICT
);

CREATE TABLE IF NOT EXISTS flow_rollups_1m (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    bucket_time DATETIME NOT NULL,
    src_ip VARCHAR(45) NULL,
    dst_port INT NULL,
    ip_proto INT NULL,
    flow_count INT NOT NULL DEFAULT 0,
    malicious_count INT NOT NULL DEFAULT 0,
    avg_confidence DECIMAL(6,5) NULL,
    max_confidence DECIMAL(6,5) NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    INDEX idx_rollups_bucket_time (bucket_time),
    INDEX idx_rollups_src_ip (src_ip),
    INDEX idx_rollups_dst_port (dst_port),
    INDEX idx_rollups_ip_proto (ip_proto),
    INDEX idx_rollups_bucket_dimensions (bucket_time, src_ip, dst_port, ip_proto)
);

CREATE TABLE IF NOT EXISTS model_registry (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    model_version VARCHAR(64) NOT NULL UNIQUE,
    artifact_path VARCHAR(255) NOT NULL,
    tokenizer_path VARCHAR(255) NULL,
    scaler_path VARCHAR(255) NULL,
    trained_on_range VARCHAR(128) NULL,
    metrics_json JSON NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    INDEX idx_model_registry_is_active (is_active),
    INDEX idx_model_registry_created_at (created_at)
);

CREATE TABLE IF NOT EXISTS analyst_actions (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    alert_id BIGINT UNSIGNED NOT NULL,
    action_type VARCHAR(32) NOT NULL,
    action_payload JSON NULL,
    actor VARCHAR(128) NOT NULL,
    action_time DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    INDEX idx_actions_alert_id (alert_id),
    INDEX idx_actions_actor (actor),
    INDEX idx_actions_action_time (action_time),
    CONSTRAINT fk_analyst_actions_alert_id
        FOREIGN KEY (alert_id)
        REFERENCES alerts (id)
        ON DELETE CASCADE
        ON UPDATE RESTRICT
);
