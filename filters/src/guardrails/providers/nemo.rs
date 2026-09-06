// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `NeMo` Guardrails provider: calls `/v1/checks` and maps the response
//! to [`GuardResult`].

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use http::{HeaderMap, HeaderValue, Method, Uri};
use praxis_ai_apis::subrequest::{self, SubRequest, SubRequestClient, SubRequestError, SubResponse};
use praxis_filter::FilterError;
use serde::{Deserialize, Serialize};

use super::{GuardPhase, GuardProvider, GuardResult};

/// Default timeout for `NeMo` HTTP calls (10 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// Maximum response body size accepted from `NeMo` (1 MiB).
const MAX_RESPONSE_SIZE: usize = 1024 * 1024;

/// `NeMo`-specific configuration fields.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NemoConfig {
    /// `NeMo` endpoint URL.
    endpoint: String,

    /// Allow the endpoint to resolve to non-public addresses.
    #[serde(default)]
    allow_private_endpoint: bool,

    /// Model name sent in each request. Defaults to `""` when omitted.
    #[serde(default)]
    model: String,

    /// Per-request timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
}

/// Returns the default timeout value for serde deserialization.
fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

/// Outgoing request payload for `NeMo`
#[derive(Serialize)]
struct NemoRequest {
    /// Model name
    model: String,
    /// List of messages to evaluate.
    messages: Vec<serde_json::Value>,
}

/// Incoming response payload for `/v1/checks`.
#[derive(Deserialize)]
struct NemoResponse {
    /// Overall verdict: `"passed"`, `"blocked"`, or `"modified"`.
    status: String,

    /// Content after rails processing.
    #[serde(default)]
    content: String,

    /// Name of the blocking rail, when `status` is `"blocked"`.
    rail: Option<String>,
}

/// `NeMo` Guardrails provider.
pub(in crate::guardrails) struct NemoProvider {
    /// Bounded HTTP client with admission control and circuit breaking.
    client: SubRequestClient,

    /// `NeMo` endpoint URL.
    endpoint: String,

    /// Model name included in every request. Empty string when not configured.
    model: String,

    /// Per-request deadline covering admission, connect, and I/O.
    timeout: Duration,

    /// Connect-time policy for the configured endpoint.
    address_policy: praxis_ai_apis::callout_target::AddressPolicy,
}

impl NemoProvider {
    /// Parse and validate `NeMo`-specific config from the provider settings.
    ///
    /// Uses the provided [`SubRequestClient`] so callouts inherit the
    /// runtime's admission control, circuit breaking, and deadline.
    ///
    /// # Errors
    ///
    /// Returns `FilterError` if the configuration is invalid.
    ///
    /// [`SubRequestClient`]: praxis_core::subrequest::SubRequestClient
    pub fn from_config(config: &serde_yaml::Value, client: SubRequestClient) -> Result<Self, FilterError> {
        let cfg: NemoConfig = serde_yaml::from_value(config.clone())
            .map_err(|e| -> FilterError { format!("ai_guardrails (nemo): {e}").into() })?;
        if cfg.endpoint.is_empty() {
            return Err("ai_guardrails (nemo): 'endpoint' must not be empty".into());
        }
        let address_policy =
            praxis_ai_apis::callout_target::AddressPolicy::from_allow_private(cfg.allow_private_endpoint);
        praxis_ai_apis::callout_target::validate_configured_http_target(
            "ai_guardrails (nemo)",
            &cfg.endpoint,
            address_policy,
        )?;
        if cfg.timeout_ms == 0 {
            return Err("ai_guardrails (nemo): 'timeout_ms' must be greater than zero".into());
        }

        Ok(Self {
            client,
            endpoint: cfg.endpoint,
            model: cfg.model,
            timeout: Duration::from_millis(cfg.timeout_ms),
            address_policy,
        })
    }

    /// POST one message slice to `/v1/checks` and map the response.
    async fn check_messages(&self, messages: Vec<serde_json::Value>) -> Result<GuardResult, FilterError> {
        let request = build_request(&self.model, messages)?;
        let response = subrequest::execute_url(
            &self.client,
            &self.endpoint,
            request,
            MAX_RESPONSE_SIZE,
            self.timeout,
            self.address_policy,
        )
        .await
        .map_err(|error| map_subrequest_error(&error))?;
        ensure_success_status(&response)?;
        let nemo_response: NemoResponse = serde_json::from_slice(&response.body)
            .map_err(|e| -> FilterError { format!("ai_guardrails (nemo): failed to parse response: {e}").into() })?;
        map_nemo_response(&nemo_response)
    }
}

#[async_trait]
impl GuardProvider for NemoProvider {
    async fn evaluate(&self, messages: Vec<serde_json::Value>, phase: GuardPhase) -> Result<GuardResult, FilterError> {
        let slices = message_slices_for_phase(&messages, phase);
        if slices.is_empty() {
            return Ok(GuardResult::Pass);
        }

        for slice in slices {
            let result = self.check_messages(slice).await?;
            if !matches!(result, GuardResult::Pass) {
                return Ok(result);
            }
        }

        Ok(GuardResult::Pass)
    }
}

// -----------------------------------------------------------------------------
// Private Utilities
// -----------------------------------------------------------------------------

/// Build cumulative message slices for the given phase.
///
/// `/v1/checks` evaluates only the last message of the relevant role per
/// call. To preserve full-conversation coverage, the provider issues one
/// HTTP request per target message, each carrying the prefix of the
/// conversation up to and including that message.
fn message_slices_for_phase(messages: &[serde_json::Value], phase: GuardPhase) -> Vec<Vec<serde_json::Value>> {
    let target_role = match phase {
        GuardPhase::Request => "user",
        GuardPhase::Response => "assistant",
    };

    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            (message.get("role").and_then(|role| role.as_str()) == Some(target_role))
                .then(|| messages[..=index].to_vec())
        })
        .collect()
}

/// Build the outbound `NeMo` JSON callout.
fn build_request(model: &str, messages: Vec<serde_json::Value>) -> Result<SubRequest, FilterError> {
    let payload = NemoRequest {
        model: model.to_owned(),
        messages,
    };
    let body =
        Bytes::from(serde_json::to_vec(&payload).map_err(|e| -> FilterError {
            format!("ai_guardrails (nemo): failed to serialize request: {e}").into()
        })?);

    let mut headers = HeaderMap::new();
    headers.insert(http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(http::header::ACCEPT, HeaderValue::from_static("application/json"));

    Ok(SubRequest {
        method: Method::POST,
        uri: Uri::default(),
        headers,
        body,
    })
}

/// Map a sub-request failure to a filter error, preserving the
/// distinct admission / circuit-open / I/O variants in the message.
fn map_subrequest_error(error: &SubRequestError) -> FilterError {
    format!("ai_guardrails (nemo): failed to send request: {error}").into()
}

/// Reject non-2xx HTTP responses from the provider.
fn ensure_success_status(response: &SubResponse) -> Result<(), FilterError> {
    if !(200..300).contains(&response.status) {
        return Err(format!(
            "ai_guardrails (nemo): provider returned HTTP status code {}",
            response.status
        )
        .into());
    }
    Ok(())
}

/// Map a deserialized [`NemoResponse`] to a [`GuardResult`].
///
/// The `/v1/checks` endpoint returns three statuses:
/// - `"passed"` - all rails passed
/// - `"blocked"` - at least one rail blocked the content
/// - `"modified"` - content was transformed (e.g. PII masked)
fn map_nemo_response(nemo: &NemoResponse) -> Result<GuardResult, FilterError> {
    match nemo.status.as_str() {
        "passed" => Ok(GuardResult::Pass),
        "blocked" => Ok(GuardResult::Block {
            reason: nemo.rail.clone().unwrap_or_default(),
        }),
        "modified" => Ok(GuardResult::Redact {
            modified_text: nemo.content.clone(),
            reason: nemo.rail.clone().unwrap_or_else(|| "modified".to_owned()),
        }),
        other => Err(format!("ai_guardrails (nemo): unknown status '{other}'").into()),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::str_to_string, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn message_slices_request_builds_cumulative_user_prefixes() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "first"}),
            serde_json::json!({"role": "assistant", "content": "reply"}),
            serde_json::json!({"role": "user", "content": "second"}),
        ];

        let slices = message_slices_for_phase(&messages, GuardPhase::Request);
        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0].len(), 2);
        assert_eq!(slices[1].len(), 4);
    }

    #[test]
    fn message_slices_response_checks_each_assistant_message() {
        let messages = vec![
            serde_json::json!({"role": "assistant", "content": "a"}),
            serde_json::json!({"role": "assistant", "content": "b"}),
        ];

        let slices = message_slices_for_phase(&messages, GuardPhase::Response);
        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0].len(), 1);
        assert_eq!(slices[1].len(), 2);
    }

    #[test]
    fn message_slices_empty_when_no_target_role() {
        let messages = vec![serde_json::json!({"role": "system", "content": "sys"})];
        assert!(message_slices_for_phase(&messages, GuardPhase::Request).is_empty());
    }

    #[test]
    fn map_nemo_response_passed_returns_pass() {
        let resp = NemoResponse {
            status: "passed".to_string(),
            content: "hello".to_string(),
            rail: None,
        };
        let result = map_nemo_response(&resp).unwrap();
        assert!(matches!(result, GuardResult::Pass));
    }

    #[test]
    fn map_nemo_response_blocked_returns_block_with_rail() {
        let resp = NemoResponse {
            status: "blocked".to_string(),
            content: "blocked text".to_string(),
            rail: Some("toxicity".to_string()),
        };
        let result = map_nemo_response(&resp).unwrap();
        assert!(
            matches!(result, GuardResult::Block { reason } if reason == "toxicity"),
            "blocked response should produce GuardResult::Block with rail name as reason"
        );
    }

    #[test]
    fn map_nemo_response_blocked_without_rail_returns_empty_reason() {
        let resp = NemoResponse {
            status: "blocked".to_string(),
            content: "blocked text".to_string(),
            rail: None,
        };
        let result = map_nemo_response(&resp).unwrap();
        assert!(matches!(result, GuardResult::Block { reason } if reason.is_empty()));
    }

    #[test]
    fn map_nemo_response_modified_returns_redact() {
        let resp = NemoResponse {
            status: "modified".to_string(),
            content: "masked text".to_string(),
            rail: None,
        };
        let result = map_nemo_response(&resp).unwrap();
        assert!(
            matches!(
                result,
                GuardResult::Redact {
                    modified_text,
                    reason
                } if modified_text == "masked text" && reason == "modified"
            ),
            "modified response should produce GuardResult::Redact"
        );
    }

    #[test]
    fn map_nemo_response_unknown_status_returns_error() {
        let resp = NemoResponse {
            status: "garbage".to_string(),
            content: String::new(),
            rail: None,
        };
        let err_msg = format!("{}", map_nemo_response(&resp).unwrap_err());
        assert!(
            err_msg.contains("unknown status 'garbage'"),
            "unknown status should produce a descriptive error: {err_msg}"
        );
    }
}
