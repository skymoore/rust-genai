use super::{DeepSeekAdapter, OllamaAdapter, OpenRouterAdapter};
use crate::adapter::{Adapter, AdapterKind, ServiceType};
use crate::chat::{ChatOptions, ChatOptionsSet, ChatRequest, ReasoningEffort};
use crate::resolver::{AuthData, Endpoint};
use crate::{Headers, ModelIden, ServiceTarget};
use serde_json::{Value, json};

type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>;

// region:    --- DeepSeek

#[test]
fn test_deepseek_managed_body_thinking_enables_non_zero_effort() -> Result<()> {
	// -- Setup & Fixtures
	let reasoning_effort = Some(ReasoningEffort::Max);

	// -- Exec
	let payload = support_deepseek_payload(reasoning_effort)?;

	// -- Check
	assert_eq!(payload["thinking"]["type"], "enabled");
	assert_eq!(payload["reasoning_effort"], "max");

	Ok(())
}

#[test]
fn test_deepseek_managed_body_thinking_disables_zero_effort() -> Result<()> {
	// -- Setup & Fixtures
	let reasoning_effort = Some(ReasoningEffort::Zero);

	// -- Exec
	let payload = support_deepseek_payload(reasoning_effort)?;

	// -- Check
	assert_eq!(payload["thinking"]["type"], "disabled");
	assert!(payload.get("reasoning_effort").is_none());

	Ok(())
}

#[test]
fn test_deepseek_managed_body_thinking_omits_fields_without_effort() -> Result<()> {
	// -- Setup & Fixtures
	let reasoning_effort = None;

	// -- Exec
	let payload = support_deepseek_payload(reasoning_effort)?;

	// -- Check
	assert!(payload.get("thinking").is_none());
	assert!(payload.get("reasoning_effort").is_none());

	Ok(())
}

// endregion: --- DeepSeek

// region:    --- Ollama

#[test]
fn test_ollama_think_omitted_without_effort() -> Result<()> {
	// -- Setup & Fixtures
	let reasoning_effort = None;

	// -- Exec
	let payload = support_ollama_payload(reasoning_effort)?;

	// -- Check
	assert!(payload.get("think").is_none());

	Ok(())
}

#[test]
fn test_ollama_think_disabled_for_zero_effort() -> Result<()> {
	// -- Setup & Fixtures
	let reasoning_effort = Some(ReasoningEffort::Zero);

	// -- Exec
	let payload = support_ollama_payload(reasoning_effort)?;

	// -- Check
	assert_eq!(payload["think"], false);

	Ok(())
}

#[test]
fn test_ollama_think_maps_efforts_to_string_levels() -> Result<()> {
	// Note: Ollama has no "minimal" level, so Minimal maps to "low",
	//       and XHigh/Max both map to "max" (Ollama's highest level).
	for (effort, expected) in [
		(ReasoningEffort::Minimal, "low"),
		(ReasoningEffort::Low, "low"),
		(ReasoningEffort::Medium, "medium"),
		(ReasoningEffort::High, "high"),
		(ReasoningEffort::XHigh, "max"),
		(ReasoningEffort::Max, "max"),
	] {
		// -- Exec
		let payload = support_ollama_payload(Some(effort.clone()))?;

		// -- Check
		assert_eq!(payload["think"], expected, "unexpected think level for {effort:?}");
	}

	Ok(())
}

#[test]
fn test_ollama_think_enabled_for_budget_effort() -> Result<()> {
	// Note: Ollama has no token-budget knob, so Budget enables thinking at the model's default level.
	// -- Setup & Fixtures
	let reasoning_effort = Some(ReasoningEffort::Budget(1024));

	// -- Exec
	let payload = support_ollama_payload(reasoning_effort)?;

	// -- Check
	assert_eq!(payload["think"], true);

	Ok(())
}

// endregion: --- Ollama

// region:    --- OpenRouter

#[test]
fn test_openrouter_default_attribution_headers() -> Result<()> {
	let (headers, payload) = support_openrouter_request(ChatOptions::default())?;
	assert_eq!(support_header(&headers, "HTTP-Referer"), Some("https://theuth.io"));
	assert_eq!(support_header(&headers, "X-OpenRouter-Title"), Some("theuth"));
	assert_eq!(support_header(&headers, "Authorization"), Some("Bearer test-key"));
	// No OpenRouter extension is emitted without the matching option.
	for key in [
		"reasoning",
		"reasoning_effort",
		"cache_control",
		"provider",
		"prompt_cache_options",
	] {
		assert!(payload.get(key).is_none(), "unexpected `{key}` in {payload}");
	}
	Ok(())
}

#[test]
fn test_openrouter_extra_headers_override_defaults_when_merged() -> Result<()> {
	// The client merges `ChatOptions::extra_headers` after the adapter headers (client_impl), so a
	// caller-provided value replaces the default; this pins that merge direction.
	let (mut headers, _) = support_openrouter_request(ChatOptions::default())?;
	let extra = Headers::from([("HTTP-Referer", "https://example.com"), ("X-OpenRouter-Title", "acme")]);
	headers.merge_with(&extra);
	assert_eq!(support_header(&headers, "HTTP-Referer"), Some("https://example.com"));
	assert_eq!(support_header(&headers, "X-OpenRouter-Title"), Some("acme"));
	Ok(())
}

#[test]
fn test_openrouter_reasoning_effort_uses_unified_reasoning_object() -> Result<()> {
	for (effort, keyword) in [
		(ReasoningEffort::Low, "low"),
		(ReasoningEffort::Medium, "medium"),
		(ReasoningEffort::High, "high"),
		(ReasoningEffort::XHigh, "xhigh"),
		(ReasoningEffort::Max, "max"),
		(ReasoningEffort::Minimal, "minimal"),
	] {
		let (_, payload) = support_openrouter_request(ChatOptions::default().with_reasoning_effort(effort))?;
		assert_eq!(payload["reasoning"], json!({"effort": keyword}), "payload: {payload}");
		assert!(payload.get("reasoning_effort").is_none());
	}
	Ok(())
}

#[test]
fn test_openrouter_reasoning_budget_maps_to_max_tokens() -> Result<()> {
	let (_, payload) =
		support_openrouter_request(ChatOptions::default().with_reasoning_effort(ReasoningEffort::Budget(2048)))?;
	assert_eq!(payload["reasoning"], json!({"max_tokens": 2048}));
	assert!(payload.get("reasoning_effort").is_none());
	Ok(())
}

#[test]
fn test_openrouter_cache_control_option_emits_top_level_ephemeral() -> Result<()> {
	let (_, payload) = support_openrouter_request(ChatOptions::default().with_openrouter_cache_control(true))?;
	assert_eq!(payload["cache_control"], json!({"type": "ephemeral"}));

	let (_, payload) = support_openrouter_request(ChatOptions::default().with_openrouter_cache_control(false))?;
	assert!(payload.get("cache_control").is_none());
	Ok(())
}

#[test]
fn test_openrouter_provider_option_is_emitted_verbatim() -> Result<()> {
	let provider =
		json!({"order": ["anthropic", "google-vertex"], "allow_fallbacks": false, "data_collection": "deny"});
	let (_, payload) = support_openrouter_request(ChatOptions::default().with_openrouter_provider(provider.clone()))?;
	assert_eq!(payload["provider"], provider);
	Ok(())
}

#[test]
fn test_openrouter_options_do_not_leak_into_plain_openai_requests() -> Result<()> {
	// Byte-level regression for every other OpenAI-compatible adapter: the new options and the
	// OpenRouter header/reasoning shaping leave the OpenAI chat request untouched.
	let baseline = support_openai_request(ChatOptions::default().with_reasoning_effort(ReasoningEffort::High))?;
	let with_options = support_openai_request(
		ChatOptions::default()
			.with_reasoning_effort(ReasoningEffort::High)
			.with_openrouter_cache_control(true)
			.with_openrouter_provider(json!({"order": ["x"]})),
	)?;
	assert_eq!(
		serde_json::to_string(&baseline.1)?,
		serde_json::to_string(&with_options.1)?
	);
	assert_eq!(baseline.1["reasoning_effort"], "high");
	assert!(baseline.1.get("reasoning").is_none());
	assert!(support_header(&baseline.0, "HTTP-Referer").is_none());
	assert!(support_header(&with_options.0, "HTTP-Referer").is_none());
	Ok(())
}

#[test]
fn test_openrouter_default_auth_env_fallback_order() {
	// `unsafe_code` is forbidden crate-wide, so the process env cannot be mutated here; the
	// selection logic is exercised through its lookup hook instead.
	use crate::adapter::adapters::support::pick_key_env_name;
	const NEW: &str = "OPENROUTER_API_KEY";
	const OLD: &str = "OPEN_ROUTER_API_KEY";
	assert_eq!(OpenRouterAdapter::DEFAULT_API_KEY_ENV_NAME, Some(OLD));
	// Neither set → the canonical default (also the name an ApiKeyEnvNotFound error reports).
	assert_eq!(pick_key_env_name(&[NEW], OLD, |_| false), OLD);
	// Only the new name set → the new name.
	assert_eq!(pick_key_env_name(&[NEW], OLD, |name| name == NEW), NEW);
	// Both set → first found wins (OPENROUTER_API_KEY before OPEN_ROUTER_API_KEY).
	assert_eq!(pick_key_env_name(&[NEW], OLD, |_| true), NEW);
	// Only the old name set → the default (the alternative is not set).
	assert_eq!(pick_key_env_name(&[NEW], OLD, |name| name == OLD), OLD);
	// The wired result is a FromEnv of whichever name won.
	let auth = OpenRouterAdapter::default_auth(AdapterKind::OpenRouter);
	assert!(
		matches!(&auth, AuthData::FromEnv(name) if name == NEW || name == OLD),
		"unexpected default auth"
	);
}

// endregion: --- OpenRouter

// region:    --- Support

fn support_deepseek_payload(reasoning_effort: Option<ReasoningEffort>) -> Result<Value> {
	let chat_options = reasoning_effort.map(|effort| ChatOptions::default().with_reasoning_effort(effort));
	let options_set = ChatOptionsSet::default().with_chat_options(chat_options.as_ref());
	let request = DeepSeekAdapter::to_web_request_data(
		ServiceTarget {
			model: ModelIden::new(AdapterKind::DeepSeek, "deepseek-v4-flash"),
			auth: AuthData::from_single("test-key"),
			endpoint: Endpoint::from_static("https://api.deepseek.com/v1/"),
		},
		ServiceType::Chat,
		ChatRequest::from_user("hello"),
		options_set,
	)?;

	Ok(request.payload)
}

fn support_ollama_payload(reasoning_effort: Option<ReasoningEffort>) -> Result<Value> {
	let chat_options = reasoning_effort.map(|effort| ChatOptions::default().with_reasoning_effort(effort));
	let options_set = ChatOptionsSet::default().with_chat_options(chat_options.as_ref());
	let request = OllamaAdapter::to_web_request_data(
		ServiceTarget {
			model: ModelIden::new(AdapterKind::Ollama, "qwen3"),
			auth: AuthData::from_single("test-key"),
			endpoint: Endpoint::from_static("http://localhost:11434/"),
		},
		ServiceType::Chat,
		ChatRequest::from_user("hello"),
		options_set,
	)?;

	Ok(request.payload)
}

fn support_openrouter_request(chat_options: ChatOptions) -> Result<(Headers, Value)> {
	let options_set = ChatOptionsSet::default().with_chat_options(Some(&chat_options));
	let request = OpenRouterAdapter::to_web_request_data(
		ServiceTarget {
			model: ModelIden::new(AdapterKind::OpenRouter, "anthropic/claude-opus-5.5"),
			auth: AuthData::from_single("test-key"),
			endpoint: OpenRouterAdapter::default_endpoint(AdapterKind::OpenRouter),
		},
		ServiceType::Chat,
		ChatRequest::from_user("hello"),
		options_set,
	)?;
	assert_eq!(request.url, "https://openrouter.ai/api/v1/chat/completions");
	Ok((request.headers, request.payload))
}

fn support_openai_request(chat_options: ChatOptions) -> Result<(Headers, Value)> {
	let options_set = ChatOptionsSet::default().with_chat_options(Some(&chat_options));
	let request = super::OpenAIAdapter::to_web_request_data(
		ServiceTarget {
			model: ModelIden::new(AdapterKind::OpenAI, "gpt-5.5"),
			auth: AuthData::from_single("test-key"),
			endpoint: super::OpenAIAdapter::default_endpoint(AdapterKind::OpenAI),
		},
		ServiceType::Chat,
		ChatRequest::from_user("hello"),
		options_set,
	)?;
	Ok((request.headers, request.payload))
}

fn support_header<'a>(headers: &'a Headers, name: &str) -> Option<&'a str> {
	headers.iter().find(|(k, _)| k.as_str() == name).map(|(_, v)| v.as_str())
}

// endregion: --- Support
