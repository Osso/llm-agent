mod common;

use common::*;
use llm_agent::*;

#[tokio::test]
async fn system_prompt_included() {
    let client = CapturingClient::new(vec![text_response("ok")]);
    let agent = AgentLoop::new(&client, NoOpExecutor).system_prompt("be helpful");
    agent.run("hello").await.unwrap();

    let captured = client.messages.lock().unwrap();
    assert_eq!(captured[0][0].role, "system");
    assert_eq!(captured[0][0].content.as_deref(), Some("be helpful"));
    assert_eq!(captured[0][1].role, "user");
    assert_eq!(captured[0][1].content.as_deref(), Some("hello"));
}

#[tokio::test]
async fn no_system_prompt_starts_with_user() {
    let client = CapturingClient::new(vec![text_response("ok")]);
    let agent = AgentLoop::new(&client, NoOpExecutor);
    agent.run("hello").await.unwrap();

    let captured = client.messages.lock().unwrap();
    assert_eq!(captured[0].len(), 1);
    assert_eq!(captured[0][0].role, "user");
}

#[tokio::test]
async fn tool_results_fed_back_correctly() {
    let client = CapturingClient::new(vec![
        tool_response(vec![make_tool_call("call_1", "Read", r#"{"p":"x"}"#)]),
        text_response("done"),
    ]);
    let executor = MockExecutor::new("file content here");
    let agent = AgentLoop::new(&client, &executor);
    agent.run("read x").await.unwrap();

    let captured = client.messages.lock().unwrap();
    let second_call = &captured[1];
    // user, assistant (with tool_calls), tool (with result)
    assert_eq!(second_call.len(), 3);
    assert_eq!(second_call[1].role, "assistant");
    assert!(second_call[1].tool_calls.is_some());
    assert_eq!(second_call[2].role, "tool");
    assert_eq!(second_call[2].content.as_deref(), Some("file content here"));
    assert_eq!(second_call[2].tool_call_id.as_deref(), Some("call_1"));
}
