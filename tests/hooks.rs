mod common;
mod hook_helpers;

use common::*;
use hook_helpers::*;
use llm_agent::*;

#[tokio::test]
async fn blocked_tool_feeds_message_as_result() {
    let client = CapturingClient::new(vec![
        tool_response(vec![make_tool_call(
            "c1",
            "Bash",
            r#"{"command":"rm -rf /"}"#,
        )]),
        text_response("ok I won't do that"),
    ]);
    let executor = MockExecutor::new("should not run");
    let agent = AgentLoop::new(&client, &executor).with_hook(BlockBash);
    let output = agent.run("delete everything").await.unwrap();

    assert_eq!(output.text(), "ok I won't do that");
    assert!(executor.called().is_empty());
    let captured = client.messages.lock().unwrap();
    assert_eq!(
        captured[1][2].content.as_deref(),
        Some("Bash is not allowed")
    );
}

#[tokio::test]
async fn allowed_tool_executes_normally() {
    let client = CapturingClient::new(vec![
        tool_response(vec![make_tool_call("c1", "Read", "{}")]),
        text_response("done"),
    ]);
    let executor = MockExecutor::new("file content");
    let agent = AgentLoop::new(&client, &executor).with_hook(BlockBash);
    agent.run("read it").await.unwrap();

    assert_eq!(executor.called().len(), 1);
    assert_eq!(executor.called()[0].0, "Read");
}

#[tokio::test]
async fn ask_denied_in_headless_mode() {
    let client = CapturingClient::new(vec![
        tool_response(vec![make_tool_call("c1", "Write", "{}")]),
        text_response("understood"),
    ]);
    let executor = MockExecutor::new("should not run");
    let agent = AgentLoop::new(&client, &executor).with_hook(AskForWrite);
    agent.run("write file").await.unwrap();

    assert!(executor.called().is_empty());
    let captured = client.messages.lock().unwrap();
    let result = captured[1][2].content.as_deref().unwrap();
    assert!(result.contains("denied"));
    assert!(result.contains("Write requires confirmation"));
}

#[tokio::test]
async fn ask_with_suggestion() {
    let client = CapturingClient::new(vec![
        tool_response(vec![make_tool_call("c1", "Bash", "{}")]),
        text_response("ok"),
    ]);
    let agent = AgentLoop::new(&client, NoOpExecutor).with_hook(SuggestHook);
    agent.run("go").await.unwrap();

    let captured = client.messages.lock().unwrap();
    let result = captured[1][2].content.as_deref().unwrap();
    assert!(result.contains("dangerous command"));
}

#[tokio::test]
async fn hook_error_aborts_loop() {
    let client = MockClient::new(vec![tool_response(vec![make_tool_call(
        "c1", "Bash", "{}",
    )])]);
    let agent = AgentLoop::new(&client, NoOpExecutor).with_hook(FailingHook);
    let err = agent.run("go").await.unwrap_err();

    assert_eq!(err.to_string(), "hook crashed");
}

#[tokio::test]
async fn mixed_decisions_in_one_turn() {
    let client = CapturingClient::new(vec![
        tool_response(vec![
            make_tool_call("c1", "Read", "{}"),
            make_tool_call("c2", "Bash", "{}"),
        ]),
        text_response("done"),
    ]);
    let executor = MockExecutor::new("content");
    let agent = AgentLoop::new(&client, &executor).with_hook(BlockBash);
    agent.run("do both").await.unwrap();

    let calls = executor.called();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "Read");

    let captured = client.messages.lock().unwrap();
    let tool_msgs: Vec<_> = captured[1].iter().filter(|m| m.role == "tool").collect();
    assert_eq!(tool_msgs.len(), 2);
    assert_eq!(tool_msgs[0].content.as_deref(), Some("content"));
    assert_eq!(tool_msgs[1].content.as_deref(), Some("Bash is not allowed"));
}

#[tokio::test]
async fn no_hook_allows_everything() {
    let client = MockClient::new(vec![
        tool_response(vec![make_tool_call("c1", "Bash", "{}")]),
        text_response("done"),
    ]);
    let executor = MockExecutor::new("ok");
    let agent = AgentLoop::new(&client, &executor);
    agent.run("go").await.unwrap();

    assert_eq!(executor.called().len(), 1);
}

#[tokio::test]
async fn hook_receives_context() {
    let client = MockClient::new(vec![
        tool_response(vec![make_tool_call("c1", "Bash", r#"{"cmd":"ls"}"#)]),
        text_response("done"),
    ]);
    let hook = CapturingHook::new();
    let agent = AgentLoop::new(&client, NoOpExecutor).with_hook(&hook);
    agent.run("go").await.unwrap();

    let captured = hook.captured.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].0, "Bash");
    assert_eq!(captured[0].1, r#"{"cmd":"ls"}"#);
    assert_eq!(captured[0].2, 0);
}
