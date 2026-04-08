mod common;

use common::*;
use llm_agent::*;

#[test]
fn text_concatenates_text_parts() {
    let r = Response {
        parts: vec![Part::Text("hello ".into()), Part::Text("world".into())],
        finish_reason: "stop".into(),
    };
    assert_eq!(r.text().unwrap(), "hello world");
}

#[test]
fn text_none_when_no_text_parts() {
    let r = Response {
        parts: vec![Part::Reasoning("thinking...".into())],
        finish_reason: "stop".into(),
    };
    assert!(r.text().is_none());
}

#[test]
fn reasoning_concatenates() {
    let r = Response {
        parts: vec![
            Part::Reasoning("step 1. ".into()),
            Part::Text("answer".into()),
            Part::Reasoning("step 2.".into()),
        ],
        finish_reason: "stop".into(),
    };
    assert_eq!(r.reasoning().unwrap(), "step 1. step 2.");
}

#[test]
fn reasoning_none_when_absent() {
    assert!(text_response("hello").reasoning().is_none());
}

#[test]
fn tool_calls_extracts_all() {
    let r = tool_response(vec![
        make_tool_call("1", "Read", "{}"),
        make_tool_call("2", "Write", "{}"),
    ]);
    let calls = r.tool_calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "Read");
    assert_eq!(calls[1].function.name, "Write");
}

#[test]
fn has_tool_calls_true() {
    assert!(tool_response(vec![make_tool_call("1", "Bash", "{}")]).has_tool_calls());
}

#[test]
fn has_tool_calls_false() {
    assert!(!text_response("done").has_tool_calls());
}

#[test]
fn mixed_parts() {
    let r = Response {
        parts: vec![
            Part::Reasoning("let me think".into()),
            Part::Text("I'll check".into()),
            Part::ToolUse(make_tool_call("1", "Read", r#"{"path":"f"}"#)),
        ],
        finish_reason: "tool_calls".into(),
    };
    assert_eq!(r.text().unwrap(), "I'll check");
    assert_eq!(r.reasoning().unwrap(), "let me think");
    assert_eq!(r.tool_calls().len(), 1);
    assert!(r.has_tool_calls());
}

#[test]
fn usage_accumulate() {
    let mut u = Usage {
        input_tokens: 10,
        output_tokens: 5,
        reasoning_tokens: 2,
    };
    u.accumulate(&Usage {
        input_tokens: 20,
        output_tokens: 10,
        reasoning_tokens: 3,
    });
    assert_eq!(u.input_tokens, 30);
    assert_eq!(u.output_tokens, 15);
    assert_eq!(u.reasoning_tokens, 5);
}

#[test]
fn usage_default_is_zero() {
    let u = Usage::default();
    assert_eq!(u.input_tokens, 0);
    assert_eq!(u.output_tokens, 0);
    assert_eq!(u.reasoning_tokens, 0);
}
