use crate::adapter::AdapterKind;
use crate::{ModelIden, ModelName};
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Who issued a thought signature. Only the issuer can read it back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThoughtOrigin {
	pub adapter_kind: AdapterKind,
	pub model_name: ModelName,
	/// The caller's name for the account/gateway the request went to
	/// (`ChatOptions::with_thought_connection`); `None` when the caller set none.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub connection: Option<String>,
}

impl ThoughtOrigin {
	pub fn new(model_iden: &ModelIden, connection: Option<&str>) -> Self {
		Self {
			adapter_kind: model_iden.adapter_kind,
			model_name: model_iden.model_name.clone(),
			connection: connection.map(str::to_string),
		}
	}
}

/// A provider's opaque reasoning continuation token (Anthropic `thinking.signature`,
/// OpenAI Responses `reasoning.encrypted_content`, Gemini `thoughtSignature`, ...).
///
/// Serialises as an object; deserialises from the object **and** from a bare string
/// (legacy rows, `origin: None`). An untagged signature is replayed to nobody.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ThoughtSignature {
	pub signature: String,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub origin: Option<ThoughtOrigin>,
}

impl ThoughtSignature {
	pub fn new(signature: impl Into<String>) -> Self {
		Self {
			signature: signature.into(),
			origin: None,
		}
	}

	pub fn with_origin(mut self, origin: ThoughtOrigin) -> Self {
		self.origin = Some(origin);
		self
	}

	/// True iff this signature was issued by `kind` over the connection `connection`
	/// (both `None` is equal; one side `None` is not). Untagged → `false`.
	pub fn readable_by(&self, kind: AdapterKind, connection: Option<&str>) -> bool {
		self.origin
			.as_ref()
			.is_some_and(|o| o.adapter_kind == kind && o.connection.as_deref() == connection)
	}
}

impl<'de> Deserialize<'de> for ThoughtSignature {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		struct SigVisitor;

		impl<'de> Visitor<'de> for SigVisitor {
			type Value = ThoughtSignature;

			fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
				f.write_str("a thought signature string or object")
			}

			fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
				Ok(ThoughtSignature::new(v))
			}

			fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
				Ok(ThoughtSignature::new(v))
			}

			fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
				let mut signature: Option<String> = None;
				let mut origin: Option<ThoughtOrigin> = None;
				while let Some(key) = map.next_key::<String>()? {
					match key.as_str() {
						"signature" => signature = Some(map.next_value()?),
						"origin" => origin = map.next_value()?,
						_ => {
							map.next_value::<de::IgnoredAny>()?;
						}
					}
				}
				Ok(ThoughtSignature {
					signature: signature.ok_or_else(|| de::Error::missing_field("signature"))?,
					origin,
				})
			}
		}

		deserializer.deserialize_any(SigVisitor)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::chat::{ChatMessage, ContentPart};
	use serde_json::json;

	fn origin(kind: AdapterKind, connection: Option<&str>) -> ThoughtOrigin {
		ThoughtOrigin::new(&ModelIden::new(kind, "m"), connection)
	}

	#[test]
	fn thought_signature_deserializes_legacy_string_and_object() {
		let legacy: ThoughtSignature = serde_json::from_value(json!("abc")).unwrap();
		assert_eq!(legacy, ThoughtSignature::new("abc"));

		let object: ThoughtSignature = serde_json::from_value(json!({
			"signature": "abc",
			"origin": {"adapter_kind": "Anthropic", "model_name": "claude", "connection": "work"}
		}))
		.unwrap();
		assert_eq!(object.signature, "abc");
		assert_eq!(
			object.origin,
			Some(ThoughtOrigin::new(
				&ModelIden::new(AdapterKind::Anthropic, "claude"),
				Some("work")
			))
		);

		// Legacy assistant message as `ChatMessage` serialised it before this type existed.
		let msg: ChatMessage = serde_json::from_value(json!({
			"role": "Assistant",
			"content": [{"ThoughtSignature": "abc"}, {"ReasoningContent": "r"}, {"Text": "t"}],
			"options": null
		}))
		.unwrap();
		assert_eq!(msg.content.thought_signatures(), vec!["abc"]);
		assert!(msg.content.thought_signature_parts()[0].origin.is_none());
	}

	#[test]
	fn thought_signature_round_trips() {
		let sig = ThoughtSignature::new("abc").with_origin(origin(AdapterKind::OpenAIResp, Some("zen")));
		let msg = ChatMessage::assistant(vec![ContentPart::from(sig.clone()), ContentPart::from_text("t")]);
		let json = serde_json::to_value(&msg).unwrap();
		assert_eq!(json["content"][0]["ThoughtSignature"]["signature"], "abc");
		assert_eq!(json["content"][0]["ThoughtSignature"]["origin"]["connection"], "zen");
		let back: ChatMessage = serde_json::from_value(json).unwrap();
		assert_eq!(back.content.thought_signature_parts(), vec![&sig]);

		// No connection → key absent, and still round-trips.
		let sig = ThoughtSignature::new("abc").with_origin(origin(AdapterKind::Gemini, None));
		let json = serde_json::to_value(&sig).unwrap();
		assert!(json["origin"].get("connection").is_none());
		assert_eq!(serde_json::from_value::<ThoughtSignature>(json).unwrap(), sig);
	}

	#[test]
	fn readable_by_truth_table() {
		let untagged = ThoughtSignature::new("s");
		assert!(!untagged.readable_by(AdapterKind::Anthropic, None));
		assert!(!untagged.readable_by(AdapterKind::Anthropic, Some("a")));

		let no_conn = ThoughtSignature::new("s").with_origin(origin(AdapterKind::Anthropic, None));
		assert!(no_conn.readable_by(AdapterKind::Anthropic, None));
		assert!(!no_conn.readable_by(AdapterKind::Anthropic, Some("a")));
		assert!(!no_conn.readable_by(AdapterKind::OpenAIResp, None));

		let conn = ThoughtSignature::new("s").with_origin(origin(AdapterKind::Anthropic, Some("a")));
		assert!(conn.readable_by(AdapterKind::Anthropic, Some("a")));
		assert!(!conn.readable_by(AdapterKind::Anthropic, Some("b")));
		assert!(!conn.readable_by(AdapterKind::Anthropic, None));
		assert!(!conn.readable_by(AdapterKind::OpenAIResp, Some("a")));
	}
}
