use super::parse_cache_creation_details;
use crate::adapter::adapters::support::{StreamerCapturedData, StreamerOptions, new_frame_tap};
use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent, InterStreamThoughtBlock};
use crate::chat::{ChatOptionsSet, PromptTokensDetails, StopReason, ThoughtOrigin, ThoughtSignature, ToolCall, Usage, UsageCost};
use crate::webc::{Event, EventSourceStream};
use crate::{Error, ModelIden, Result};
use serde_json::{Map, Value};
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};
use value_ext::JsonValueExt;

pub struct AnthropicStreamer {
	inner: EventSourceStream,
	options: StreamerOptions,

	// -- Set by the poll_next
	/// Flag to prevent polling the EventSource after a MessageStop event
	done: bool,

	captured_data: StreamerCapturedData,
	captured_thought_blocks: Vec<InterStreamThoughtBlock>,
	in_progress_block: InProgressBlock,
	pending_events: VecDeque<InterStreamEvent>,
	/// End built at `message_stop`, held until the connection closes so a trailing gateway cost
	/// frame (OpenCode Zen `event: ping` `{"type":"ping","cost":"…"}`) can still land in `captured_usage`.
	pending_end: Option<InterStreamEnd>,
	/// Ensures the unreplayable-thinking warning is emitted at most once per stream.
	warned_unreplayable_thinking: bool,
}

enum InProgressBlock {
	Text,
	ToolUse { id: String, name: String, input: String },
	Thinking(ThinkingBlock),
}

#[derive(Default)]
struct ThinkingBlock {
	/// `None` when the caller did not opt into reasoning content capture.
	reasoning: Option<String>,
	/// `None` when the caller opted out of every capture this block feeds.
	signature: Option<String>,
}

impl ThinkingBlock {
	fn new(reasoning: Option<String>, signature: Option<String>) -> Self {
		Self { reasoning, signature }
	}

	fn append_reasoning(&mut self, reasoning: &str) {
		if let Some(captured) = &mut self.reasoning {
			captured.push_str(reasoning);
		}
	}

	/// Append one `signature_delta` chunk.
	///
	/// Anthropic sends a block signature as `content_block_start.content_block.signature`
	/// (which seeds this buffer) followed by zero or more `signature_delta` chunks that
	/// concatenate. Signatures are opaque base64 blobs, so this must stay a plain append:
	/// any attempt to detect resends or overlapping chunks can match by coincidence and
	/// silently corrupt the signature, which Anthropic then rejects.
	fn append_signature_delta(&mut self, delta: &str) {
		if let Some(signature) = &mut self.signature {
			signature.push_str(delta);
		}
	}

	fn into_thought_block(self, origin: &ThoughtOrigin) -> Option<InterStreamThoughtBlock> {
		let signature = self.signature.filter(|signature| !signature.is_empty())?;
		Some(InterStreamThoughtBlock {
			reasoning_content: self.reasoning,
			signature: ThoughtSignature::new(signature).with_origin(origin.clone()),
		})
	}
}

impl AnthropicStreamer {
	pub fn new(inner: EventSourceStream, model_iden: ModelIden, options_set: ChatOptionsSet<'_, '_>) -> Self {
		let frame_tap = new_frame_tap(&model_iden, &options_set);

		Self {
			inner: inner.with_frame_tap(frame_tap),
			done: false,
			options: StreamerOptions::new(model_iden, options_set),
			captured_data: Default::default(),
			captured_thought_blocks: Vec::new(),
			in_progress_block: InProgressBlock::Text,
			pending_events: VecDeque::new(),
			pending_end: None,
			warned_unreplayable_thinking: false,
		}
	}

	/// Clones the frame tap (if any), so `ChatStream` can fire the terminal sink hooks.
	pub fn frame_tap(&self) -> Option<crate::webc::FrameTap> {
		self.inner.frame_tap()
	}
}

impl futures::Stream for AnthropicStreamer {
	type Item = Result<InterStreamEvent>;

	fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		if self.done {
			return Poll::Ready(None);
		}
		if let Some(event) = self.pending_events.pop_front() {
			return Poll::Ready(Some(Ok(event)));
		}

		while let Poll::Ready(event) = Pin::new(&mut self.inner).poll_next(cx) {
			// NOTE: At this point, we capture more events than needed for genai::StreamItem, but it serves as documentation.
			match event {
				Some(Ok(Event::Open)) => return Poll::Ready(Some(Ok(InterStreamEvent::Start))),
				Some(Ok(Event::Message(message))) => {
					let message_type = message.event.as_str();

					match message_type {
						"message_start" => {
							self.capture_usage(message_type, &message.data)?;
							continue;
						}
						"message_delta" => {
							self.capture_usage(message_type, &message.data)?;
							// Capture stop_reason from delta (e.g., "end_turn", "max_tokens", "tool_use")
							if let Ok(data) = self.parse_message_data(&message.data)
								&& let Ok(reason) = data.x_get::<String>("/delta/stop_reason")
							{
								self.captured_data.stop_reason = Some(reason);
							}
							continue;
						}
						"content_block_start" => {
							let mut data: Value =
								serde_json::from_str(&message.data).map_err(|serde_error| Error::StreamParse {
									model_iden: self.options.model_iden.clone(),
									serde_error,
								})?;

							match data.x_get_str("/content_block/type") {
								Ok("text") => self.in_progress_block = InProgressBlock::Text,
								Ok("thinking") => {
									let thinking = data.x_take::<String>("/content_block/thinking").unwrap_or_default();
									let signature =
										data.x_take::<String>("/content_block/signature").unwrap_or_default();

									if self.options.capture_reasoning_content && !thinking.is_empty() {
										match self.captured_data.reasoning_content {
											Some(ref mut reasoning) => reasoning.push_str(&thinking),
											None => self.captured_data.reasoning_content = Some(thinking.clone()),
										}
									}

									// The signature is the replayable half of a thinking block, so it
									// follows the same opt-in as the reasoning text it belongs to.
									// Callers replaying a signed assistant turn must set
									// `capture_reasoning_content`, otherwise Anthropic rejects the
									// continuation.
									// A signature is only replayable together with the reasoning it signs, so
									// capturing tool calls alone silently yields a continuation Anthropic
									// rejects. Surface that once rather than failing later at the provider.
									if self.options.capture_tool_calls
										&& !self.options.capture_reasoning_content
										&& !self.warned_unreplayable_thinking
									{
										self.warned_unreplayable_thinking = true;
										tracing::warn!(
											"anthropic - thinking block received with capture_tool_calls but not \
											 capture_reasoning_content; thought signatures will not be captured and \
											 the tool continuation will be missing its thinking blocks"
										);
									}

									let capture_block = self.options.capture_reasoning_content;
									let captured_reasoning = capture_block.then_some(thinking.clone());
									let captured_signature = capture_block.then(|| signature.clone());
									self.in_progress_block = InProgressBlock::Thinking(ThinkingBlock::new(
										captured_reasoning,
										captured_signature,
									));

									if !signature.is_empty() {
										self.pending_events
											.push_back(InterStreamEvent::ThoughtSignatureChunk(signature));
									}
									if !thinking.is_empty() {
										self.pending_events.push_back(InterStreamEvent::ReasoningChunk(thinking));
									}
									if let Some(event) = self.pending_events.pop_front() {
										return Poll::Ready(Some(Ok(event)));
									}
								}
								Ok("tool_use") => {
									let id: String = data.x_take("/content_block/id")?;
									let name: String = data.x_take("/content_block/name")?;

									// Emit an initial ToolCallChunk with name and empty args,
									// matching OpenAI's incremental streaming behaviour.
									let tc = ToolCall {
										call_id: id.clone(),
										fn_name: name.clone(),
										fn_arguments: Value::String(String::new()),
										thought_signatures: None,
									};

									self.in_progress_block = InProgressBlock::ToolUse {
										id,
										name,
										input: String::new(),
									};

									return Poll::Ready(Some(Ok(InterStreamEvent::ToolCallChunk(tc))));
								}
								Ok(txt) => {
									tracing::warn!("unhandled content type: {txt}");
								}
								Err(e) => {
									tracing::error!("{e:?}");
								}
							}

							continue;
						}
						"content_block_delta" => {
							let mut data: Value =
								serde_json::from_str(&message.data).map_err(|serde_error| Error::StreamParse {
									model_iden: self.options.model_iden.clone(),
									serde_error,
								})?;

							match &mut self.in_progress_block {
								InProgressBlock::Text => {
									let content: String = data.x_take("/delta/text")?;

									// Add to the captured_content if chat options say so
									if self.options.capture_content {
										match self.captured_data.content {
											Some(ref mut c) => c.push_str(&content),
											None => self.captured_data.content = Some(content.clone()),
										}
									}

									return Poll::Ready(Some(Ok(InterStreamEvent::Chunk(content))));
								}
								InProgressBlock::ToolUse { id, name, input } => {
									let partial = data.x_get_str("/delta/partial_json")?;
									input.push_str(partial);

									// Emit incremental ToolCallChunk with accumulated args
									// (as Value::String, same convention as OpenAI adapter).
									let tc = ToolCall {
										call_id: id.clone(),
										fn_name: name.clone(),
										fn_arguments: Value::String(input.clone()),
										thought_signatures: None,
									};

									return Poll::Ready(Some(Ok(InterStreamEvent::ToolCallChunk(tc))));
								}
								InProgressBlock::Thinking(thinking_block) => {
									if let Ok(thinking) = data.x_take::<String>("/delta/thinking") {
										thinking_block.append_reasoning(&thinking);
										// Add to the captured_thinking if chat options say so
										if self.options.capture_reasoning_content {
											match self.captured_data.reasoning_content {
												Some(ref mut r) => r.push_str(&thinking),
												None => self.captured_data.reasoning_content = Some(thinking.clone()),
											}
										}

										return Poll::Ready(Some(Ok(InterStreamEvent::ReasoningChunk(thinking))));
									} else if let Ok(signature) = data.x_take::<String>("/delta/signature") {
										thinking_block.append_signature_delta(&signature);
										return Poll::Ready(Some(Ok(InterStreamEvent::ThoughtSignatureChunk(
											signature,
										))));
									} else {
										// If it is thinking but no thinking or signature field, we log and skip.
										tracing::warn!(
											"content_block_delta for thinking block but no thinking or signature found: {data:?}"
										);
										continue;
									}
								}
							}
						}
						"content_block_stop" => {
							match std::mem::replace(&mut self.in_progress_block, InProgressBlock::Text) {
								InProgressBlock::ToolUse { id, name, input } if self.options.capture_tool_calls => {
									// ToolCallChunks were already emitted incrementally
									// during content_block_start and content_block_delta.
									// Here we only finalize capture with parsed arguments.
									let fn_arguments = if input.is_empty() {
										Value::Object(Map::new())
									} else {
										serde_json::from_str(&input).unwrap_or_else(|e| {
											tracing::warn!(
												"Anthropic streamer: failed to parse tool-call input JSON ({} bytes): {e}",
												input.len()
											);
											Value::String(input)
										})
									};

									let tc = ToolCall {
										call_id: id,
										fn_name: name,
										fn_arguments,
										thought_signatures: None,
									};

									match self.captured_data.tool_calls {
										Some(ref mut t) => t.push(tc),
										None => self.captured_data.tool_calls = Some(vec![tc]),
									}
								}
								InProgressBlock::Thinking(thinking_block) => {
									if let Some(block) = thinking_block.into_thought_block(&self.options.thought_origin) {
										self.captured_thought_blocks.push(block);
									}
								}
								_ => {
									// no-op for remaining block types
								}
							}

							continue;
						}
						// -- END MESSAGE
						"message_stop" => {
							// Capture the usage
							let captured_usage = if self.options.capture_usage {
								self.captured_data.usage.take().map(|mut usage| {
									// Compute the total if any of input/output are not null
									if usage.prompt_tokens.is_some() || usage.completion_tokens.is_some() {
										usage.total_tokens = Some(
											usage.prompt_tokens.unwrap_or(0) + usage.completion_tokens.unwrap_or(0),
										);
									}
									usage
								})
							} else {
								None
							};

							// Held until the connection closes (or a cost ping arrives), see `pending_end`.
							self.pending_end = Some(InterStreamEnd {
								captured_usage,
								captured_stop_reason: self.captured_data.stop_reason.take().map(StopReason::from),
								captured_text_content: self.captured_data.content.take(),
								captured_reasoning_content: self.captured_data.reasoning_content.take(),
								captured_tool_calls: self.captured_data.tool_calls.take(),
								captured_thought_signatures: None,
								captured_thought_blocks: (!self.captured_thought_blocks.is_empty())
									.then(|| std::mem::take(&mut self.captured_thought_blocks)),
								captured_response_id: None,
							});
							continue;
						}

						"ping" => {
							// OpenCode Zen injects the billed cost as a `ping` after `message_stop`;
							// a plain ping is a heartbeat.
							let cost = self
								.options
								.capture_usage
								.then(|| serde_json::from_str::<Value>(&message.data).ok())
								.flatten()
								.and_then(|data| data.get("cost").and_then(UsageCost::provider_reported));
							if let Some(cost) = cost {
								if let Some(mut end) = self.pending_end.take() {
									end.captured_usage.get_or_insert_default().cost = Some(cost);
									self.done = true;
									return Poll::Ready(Some(Ok(InterStreamEvent::End(end))));
								}
								self.captured_data.usage.get_or_insert_default().cost = Some(cost);
								continue;
							}
							if self.pending_end.is_some() {
								continue;
							}
							// Map ping events to Heartbeat events to indicate the stream is still active
							return Poll::Ready(Some(Ok(InterStreamEvent::Heartbeat)));
						}
						"error" => {
							// Anthropic may emit an `event: error` mid-stream (e.g. overloaded_error,
							// rate_limit, internal_server_error). Propagate it as a typed error so the
							// caller can surface the real cause instead of silently ending the stream.
							tracing::warn!("Anthropic stream error event, data: {}", message.data);
							let body: Value = serde_json::from_str(&message.data)
								.unwrap_or_else(|_| Value::String(message.data.clone()));
							self.done = true;
							return Poll::Ready(Some(Err(Error::ChatResponse {
								model_iden: self.options.model_iden.clone(),
								body,
							})));
						}
						other => tracing::warn!("UNKNOWN MESSAGE TYPE: {other}, data: {}", message.data),
					}
				}
				Some(Err(err)) => {
					tracing::error!("Error: {}", err);
					return Poll::Ready(Some(Err(Error::WebStream {
						model_iden: self.options.model_iden.clone(),
						cause: err.to_string(),
						error: err,
					})));
				}
				None => {
					if let Some(end) = self.pending_end.take() {
						self.done = true;
						return Poll::Ready(Some(Ok(InterStreamEvent::End(end))));
					}
					return Poll::Ready(None);
				}
			}
		}
		Poll::Pending
	}
}

// Support
impl AnthropicStreamer {
	fn capture_usage(&mut self, message_type: &str, message_data: &str) -> Result<()> {
		if self.options.capture_usage {
			let mut data = self.parse_message_data(message_data)?;

			let usage_path = if message_type == "message_start" {
				"/message/usage"
			} else if message_type == "message_delta" {
				"/usage"
			} else {
				// TODO: Use tracing
				tracing::debug!(
					"TRACING DEBUG - Anthropic message type not supported for input/output tokens: {message_type}"
				);
				return Ok(()); // For now permissive
			};

			// NOTE: Permissive on this one; if the usage object is absent, treat it as nonexistent (for now)
			let Ok(mut usage_value) = data.x_take::<Value>(usage_path) else {
				return Ok(());
			};

			// -- Capture the eventual input/output tokens
			let input_tokens = usage_value.x_take::<i32>("input_tokens").ok();
			let output_tokens = usage_value.x_take::<i32>("output_tokens").ok();

			// -- Capture cache tokens.
			// Standard Anthropic reports them in `message_start` and repeats them in `message_delta`;
			// some gateways (e.g. Alibaba DashScope / Qwen) report them only in `message_delta`.
			// NOTE: Anthropic's input_tokens does NOT include cached tokens, so we must add them.
			// See also: AnthropicAdapter::into_usage() for non-streaming equivalent.
			let cache_creation: i32 = usage_value.x_get("cache_creation_input_tokens").unwrap_or(0);
			let cache_read: i32 = usage_value.x_get("cache_read_input_tokens").unwrap_or(0);

			// Parse cache_creation breakdown if present (TTL-specific breakdown)
			let cache_creation_details = usage_value
				.x_get::<Value>("cache_creation")
				.ok()
				.as_ref()
				.and_then(parse_cache_creation_details);

			let has_cache = cache_creation > 0 || cache_read > 0 || cache_creation_details.is_some();

			// Nothing to capture from this snapshot, leave `usage` as-is (possibly None).
			if input_tokens.is_none() && output_tokens.is_none() && !has_cache {
				return Ok(());
			}

			let usage = self.captured_data.usage.get_or_insert(Usage::default());

			if let Some(input_tokens) = input_tokens {
				usage.prompt_tokens = Some(input_tokens + cache_creation + cache_read);
			}
			// NOTE: Cache tokens without `input_tokens` (gateway shape) are deliberately NOT added
			// to `prompt_tokens`, since gateway token accounting is provider-specific.

			if let Some(output_tokens) = output_tokens {
				usage.completion_tokens = Some(output_tokens);
			}

			if has_cache {
				// Anthropic sends the `cache_creation` breakdown only in `message_start`, while
				// `message_delta` repeats the totals without it. Keep the earlier breakdown so the
				// later snapshot does not erase it.
				let cache_creation_details = cache_creation_details.or_else(|| {
					usage
						.prompt_tokens_details
						.as_ref()
						.and_then(|details| details.cache_creation_details.clone())
				});

				// Set prompt_tokens_details (match into_usage behavior: always Some(value))
				usage.prompt_tokens_details = Some(PromptTokensDetails {
					cache_creation_tokens: Some(cache_creation),
					cache_creation_details,
					cached_tokens: Some(cache_read),
					audio_tokens: None,
				});
			}
		}

		Ok(())
	}

	/// Simple wrapper for now, with the corresponding map_err.
	/// Might have more logic later.
	fn parse_message_data(&self, payload: &str) -> Result<Value> {
		serde_json::from_str(payload).map_err(|serde_error| Error::StreamParse {
			model_iden: self.options.model_iden.clone(),
			serde_error,
		})
	}
}

#[cfg(test)]
mod tests {
	use super::{AnthropicStreamer, ThinkingBlock};
	use crate::ModelIden;
	use crate::adapter::AdapterKind;
	use crate::adapter::adapters::support::test_support::{capture_all, chunks_text, collect, options_set, sse_stream};
	use crate::adapter::inter_stream::InterStreamEvent;
	use crate::chat::{ChatOptions, ThoughtOrigin, UsageCostSource};

	const ANTHROPIC_STREAM: &str = concat!(
		"event: message_start\n",
		"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":25,\"output_tokens\":1,\"cache_read_input_tokens\":5}}}\n\n",
		"event: ping\n",
		"data: {\"type\":\"ping\"}\n\n",
		"event: content_block_start\n",
		"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"event: content_block_delta\n",
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
		"event: content_block_stop\n",
		"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: message_delta\n",
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n",
		"event: message_stop\n",
		"data: {\"type\":\"message_stop\"}\n\n",
	);

	async fn run(body: &str) -> Vec<crate::Result<InterStreamEvent>> {
		let options = capture_all();
		let model = ModelIden::new(AdapterKind::Anthropic, "m");
		collect(AnthropicStreamer::new(
			sse_stream(body).await,
			model,
			options_set(&options),
		))
		.await
	}

	#[tokio::test]
	async fn test_zen_anthropic_stream_cost_ping_after_message_stop_lands_in_usage() {
		let body = format!("{ANTHROPIC_STREAM}event: ping\ndata: {{\"type\":\"ping\",\"cost\":\"0.00123400\"}}\n\n");
		let events = run(&body).await;
		let Some(Ok(InterStreamEvent::End(end))) = events.last() else {
			panic!("End must be last: {events:?}");
		};
		let usage = end.captured_usage.as_ref().expect("usage");
		assert_eq!(usage.prompt_tokens, Some(30));
		assert_eq!(usage.completion_tokens, Some(7));
		assert_eq!(usage.total_tokens, Some(37));
		let cost = usage.cost.as_ref().expect("cost");
		assert_eq!(cost.amount, 0.001234);
		assert_eq!(cost.source, UsageCostSource::ProviderReported);
		assert_eq!(chunks_text(&events), "Hello");
		// The in-stream plain ping is still a heartbeat; the cost ping is not.
		let heartbeats = events
			.iter()
			.filter(|event| matches!(event, Ok(InterStreamEvent::Heartbeat)))
			.count();
		assert_eq!(heartbeats, 1);
	}

	#[tokio::test]
	async fn test_anthropic_plain_ping_changes_nothing() {
		let events = run(ANTHROPIC_STREAM).await;
		let Some(Ok(InterStreamEvent::End(end))) = events.last() else {
			panic!("End must be last: {events:?}");
		};
		let usage = end.captured_usage.as_ref().expect("usage");
		assert_eq!(usage.prompt_tokens, Some(30));
		assert_eq!(usage.completion_tokens, Some(7));
		assert!(usage.cost.is_none());
		assert_eq!(chunks_text(&events), "Hello");
		assert_eq!(
			events
				.iter()
				.filter(|event| matches!(event, Ok(InterStreamEvent::Heartbeat)))
				.count(),
			1
		);
	}

	fn signature(start: &str, deltas: &[&str]) -> Option<String> {
		let mut block = ThinkingBlock::new(None, Some(start.to_string()));
		for delta in deltas {
			block.append_signature_delta(delta);
		}
		let origin = ThoughtOrigin::new(&ModelIden::new(AdapterKind::Anthropic, "m"), None);
		block.into_thought_block(&origin).map(|block| block.signature.signature)
	}

	/// A streamed thinking block lands in `StreamEnd.captured_content` as a signature tagged
	/// with this adapter kind and the connection label the request was made with.
	#[tokio::test]
	async fn captured_thinking_signature_carries_origin_and_connection() {
		let body = concat!(
			"event: message_start\n",
			"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
			"event: content_block_start\n",
			"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
			"event: content_block_delta\n",
			"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Plan.\"}}\n\n",
			"event: content_block_delta\n",
			"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig-1\"}}\n\n",
			"event: content_block_stop\n",
			"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
			"event: content_block_start\n",
			"data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
			"event: content_block_delta\n",
			"data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
			"event: content_block_stop\n",
			"data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
			"event: message_delta\n",
			"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
			"event: message_stop\n",
			"data: {\"type\":\"message_stop\"}\n\n",
		);
		let options: ChatOptions = capture_all().with_thought_connection("work");
		let model = ModelIden::new(AdapterKind::Anthropic, "claude-x");
		let mut events = collect(AnthropicStreamer::new(sse_stream(body).await, model, options_set(&options))).await;
		let Some(Ok(InterStreamEvent::End(end))) = events.pop() else {
			panic!("End must be last: {events:?}");
		};
		let stream_end = crate::chat::StreamEnd::from(end);
		let content = stream_end.captured_content.expect("captured content");
		let sig = content.thought_signature_parts()[0];
		assert_eq!(sig.signature, "sig-1");
		let origin = sig.origin.as_ref().expect("origin");
		assert_eq!(origin.adapter_kind, AdapterKind::Anthropic);
		assert_eq!(origin.model_name, crate::ModelName::from("claude-x"));
		assert_eq!(origin.connection.as_deref(), Some("work"));
		assert!(sig.readable_by(AdapterKind::Anthropic, Some("work")));
		assert!(!sig.readable_by(AdapterKind::Anthropic, None));
		assert_eq!(content.reasoning_contents(), vec!["Plan."]);
	}

	#[test]
	fn split_signature_deltas_form_one_logical_block_signature() {
		assert_eq!(
			signature("", &["signed-", "fragmented"]).as_deref(),
			Some("signed-fragmented")
		);
	}

	#[test]
	fn start_and_delta_signature_variants_form_one_logical_signature() {
		// A start-only signature is kept as-is.
		assert_eq!(signature("start-only", &[]).as_deref(), Some("start-only"));
		// A non-empty start seeds the buffer and later deltas append to it.
		assert_eq!(signature("signed-", &["suffix"]).as_deref(), Some("signed-suffix"));
		// Chunks concatenate verbatim, including when a chunk repeats earlier bytes.
		// Signatures are opaque base64, so no resend/overlap collapsing is applied.
		assert_eq!(signature("ab", &["ab"]).as_deref(), Some("abab"));
		assert_eq!(signature("", &["QUJD", "QUJD"]).as_deref(), Some("QUJDQUJD"));
	}
}
