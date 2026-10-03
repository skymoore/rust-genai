//! This support module is for common constructs and utilities for all the adapter implementations.
//! It should be private to the `crate::adapter::adapters` module.

use crate::ModelIden;
use crate::chat::{ChatOptionsSet, FrameCtx, ThoughtOrigin, Usage};
use crate::resolver::AuthData;
use crate::webc::FrameTap;
use crate::{Error, Result};

pub fn get_api_key(auth: AuthData, model: &ModelIden) -> Result<String> {
	auth.single_key_value().map_err(|resolver_error| Error::Resolver {
		model_iden: model.clone(),
		resolver_error,
	})
}

/// Env var name an adapter reads its key from: the first of `alternatives` for which `is_set`
/// holds (non-empty in the process environment), else `default` (also the name a "not found"
/// error reports). Used by `impl_pass_through_adapter!`'s `key_env_alt`.
pub fn pick_key_env_name<'a>(alternatives: &[&'a str], default: &'a str, is_set: impl Fn(&str) -> bool) -> &'a str {
	alternatives.iter().copied().find(|name| is_set(name)).unwrap_or(default)
}

/// Builds the tap that feeds a user `ChatFrameSink`, when one is configured.
pub fn new_frame_tap(model_iden: &ModelIden, options_set: &ChatOptionsSet<'_, '_>) -> Option<FrameTap> {
	let sink = options_set.raw_frame_sink()?;
	Some(FrameTap::new(sink, FrameCtx::new(model_iden.clone())))
}

// region:    --- StreamerChatOptions

#[derive(Debug)]
pub struct StreamerOptions {
	pub capture_usage: bool,
	pub capture_reasoning_content: bool,
	pub capture_content: bool,
	pub capture_tool_calls: bool,
	pub model_iden: ModelIden,
	/// Stamped onto every thought signature this stream captures.
	pub thought_origin: ThoughtOrigin,
}

impl StreamerOptions {
	pub fn new(model_iden: ModelIden, options_set: ChatOptionsSet<'_, '_>) -> Self {
		Self {
			capture_usage: options_set.capture_usage().unwrap_or(false),
			capture_content: options_set.capture_content().unwrap_or(false),
			capture_reasoning_content: options_set.capture_reasoning_content().unwrap_or(false),
			capture_tool_calls: options_set.capture_tool_calls().unwrap_or(false),
			thought_origin: ThoughtOrigin::new(&model_iden, options_set.thought_connection()),
			model_iden,
		}
	}
}

// endregion: --- StreamerChatOptions

// region:    --- Streamer Captured Data

#[derive(Debug, Default)]
pub struct StreamerCapturedData {
	pub usage: Option<Usage>,
	pub stop_reason: Option<String>,
	pub content: Option<String>,
	pub reasoning_content: Option<String>,
	pub tool_calls: Option<Vec<crate::chat::ToolCall>>,
	pub thought_signatures: Option<Vec<crate::chat::ThoughtSignature>>,
}

// endregion: --- Streamer Captured Data

// region:    --- Test Support

/// Drives a streamer over handcrafted SSE bytes served by a one-shot local HTTP server.
#[cfg(test)]
pub(crate) mod test_support {
	use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
	use crate::chat::{ChatOptions, ChatOptionsSet};
	use crate::webc::EventSourceStream;
	use futures::{Stream, StreamExt};
	use tokio::io::{AsyncReadExt, AsyncWriteExt};
	use tokio::net::TcpListener;

	/// Serves `body` as a `text/event-stream` 200 response once, closing the connection afterwards.
	pub(crate) async fn sse_stream(body: &str) -> EventSourceStream {
		let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
		let url = format!("http://{}/", listener.local_addr().expect("addr"));
		let raw_response = format!(
			"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
			body.len()
		);
		tokio::spawn(async move {
			if let Ok((mut socket, _)) = listener.accept().await {
				let mut buf = [0u8; 4096];
				let _ = socket.read(&mut buf).await;
				let _ = socket.write_all(raw_response.as_bytes()).await;
				let _ = socket.shutdown().await;
			}
		});
		EventSourceStream::new(reqwest::Client::new().post(&url))
	}

	/// Chat options capturing everything, as `ChatOptionsSet` for a streamer constructor.
	pub(crate) fn capture_all() -> ChatOptions {
		ChatOptions::default()
			.with_capture_usage(true)
			.with_capture_content(true)
			.with_capture_reasoning_content(true)
			.with_capture_tool_calls(true)
	}

	pub(crate) fn options_set(options: &ChatOptions) -> ChatOptionsSet<'_, '_> {
		ChatOptionsSet::default().with_chat_options(Some(options))
	}

	/// Collects every event; errors are returned as `Err` at their position so tests can assert on them.
	pub(crate) async fn collect<S>(stream: S) -> Vec<crate::Result<InterStreamEvent>>
	where
		S: Stream<Item = crate::Result<InterStreamEvent>>,
	{
		stream.collect().await
	}

	/// The single `End` of a collected event list (panics with the events if there is not exactly one).
	pub(crate) fn single_end(events: &[crate::Result<InterStreamEvent>]) -> &InterStreamEnd {
		let ends: Vec<&InterStreamEnd> = events
			.iter()
			.filter_map(|event| match event {
				Ok(InterStreamEvent::End(end)) => Some(end),
				_ => None,
			})
			.collect();
		assert_eq!(ends.len(), 1, "expected exactly one End event, got: {events:?}");
		ends[0]
	}

	/// Concatenation of all text chunks.
	pub(crate) fn chunks_text(events: &[crate::Result<InterStreamEvent>]) -> String {
		events
			.iter()
			.filter_map(|event| match event {
				Ok(InterStreamEvent::Chunk(text)) => Some(text.as_str()),
				_ => None,
			})
			.collect()
	}
}

// endregion: --- Test Support
