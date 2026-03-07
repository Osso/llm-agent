mod common;

use common::*;
use llm_agent::*;

#[test]
fn chat_message_omits_nones() {
    let msg = ChatMessage {
        role: "user".into(),
        content: Some("hello".into()),
        tool_calls: None,
        tool_call_id: None,
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(!json.contains("tool_calls"));
    assert!(!json.contains("tool_call_id"));
}

#[test]
fn chat_message_roundtrip() {
    let msg = ChatMessage {
        role: "assistant".into(),
        content: Some("text".into()),
        tool_calls: Some(vec![make_tool_call("1", "Bash", r#"{"cmd":"ls"}"#)]),
        tool_call_id: None,
    };
    let json = serde_json::to_string(&msg).unwrap();
    let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.role, "assistant");
    assert_eq!(parsed.tool_calls.unwrap()[0].function.name, "Bash");
}

#[test]
fn tool_call_type_field_renamed() {
    let tc = make_tool_call("1", "Read", "{}");
    let json = serde_json::to_string(&tc).unwrap();
    assert!(json.contains(r#""type":"function"#));
    assert!(!json.contains("call_type"));
}
