mod common;

use common::*;
use llm_agent::*;
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn single_turn_text_response() {
    let client = MockClient::new(vec![text_response("hello world")]);
    let agent = AgentLoop::new(&client, NoOpExecutor);
    let output = agent.run("hi").await.unwrap();

    assert_eq!(output.text(), "hello world");
    assert_eq!(output.usage.input_tokens, 10);
    assert_eq!(client.calls(), 1);
}

#[tokio::test]
async fn tool_call_then_text() {
    let client = MockClient::new(vec![
        tool_response(vec![make_tool_call("c1", "Bash", r#"{"command":"ls"}"#)]),
        text_response("found 3 files"),
    ]);
    let executor = MockExecutor::new("file1\nfile2\nfile3");
    let agent = AgentLoop::new(&client, &executor);
    let output = agent.run("list files").await.unwrap();

    assert_eq!(output.text(), "found 3 files");
    assert_eq!(client.calls(), 2);
    assert_eq!(executor.called().len(), 1);
    assert_eq!(executor.called()[0].0, "Bash");
}

#[tokio::test]
async fn multiple_tool_calls_in_one_turn() {
    let client = MockClient::new(vec![
        tool_response(vec![
            make_tool_call("c1", "Read", r#"{"path":"a.txt"}"#),
            make_tool_call("c2", "Read", r#"{"path":"b.txt"}"#),
        ]),
        text_response("both read"),
    ]);
    let executor = MockExecutor::new("content");
    let agent = AgentLoop::new(&client, &executor);
    agent.run("read both").await.unwrap();

    let calls = executor.called();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].1, r#"{"path":"a.txt"}"#);
    assert_eq!(calls[1].1, r#"{"path":"b.txt"}"#);
}

#[tokio::test]
async fn parallel_safe_tool_calls_execute_in_one_turn() {
    struct ParallelExecutor {
        calls: Mutex<Vec<String>>,
    }

    #[async_trait::async_trait]
    impl ToolExecutor for ParallelExecutor {
        async fn execute(&self, name: &str, arguments: &str) -> String {
            self.calls
                .lock()
                .unwrap()
                .push(format!("{name}:{arguments}"));
            "ok".to_string()
        }

        fn supports_parallel(&self, _tool_name: &str) -> bool {
            true
        }
    }

    let client = MockClient::new(vec![
        tool_response(vec![
            make_tool_call("c1", "Read", r#"{"path":"a.txt"}"#),
            make_tool_call("c2", "Glob", r#"{"pattern":"*.rs"}"#),
        ]),
        text_response("done"),
    ]);
    let executor = ParallelExecutor {
        calls: Mutex::new(Vec::new()),
    };

    AgentLoop::new(&client, &executor)
        .run("read both")
        .await
        .unwrap();

    let calls = executor.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert!(calls.contains(&r#"Read:{"path":"a.txt"}"#.to_string()));
    assert!(calls.contains(&r#"Glob:{"pattern":"*.rs"}"#.to_string()));
}

#[tokio::test]
async fn multi_turn_tool_calls() {
    let client = MockClient::new(vec![
        tool_response(vec![make_tool_call("c1", "Read", "{}")]),
        tool_response(vec![make_tool_call("c2", "Write", "{}")]),
        text_response("done"),
    ]);
    let agent = AgentLoop::new(&client, MockExecutor::new("ok"));
    let output = agent.run("do stuff").await.unwrap();

    assert_eq!(output.text(), "done");
    assert_eq!(client.calls(), 3);
}

#[tokio::test]
async fn max_turns_exceeded() {
    let client = MockClient::new(vec![
        tool_response(vec![make_tool_call("c1", "Bash", "{}")]),
        tool_response(vec![make_tool_call("c2", "Bash", "{}")]),
        tool_response(vec![make_tool_call("c3", "Bash", "{}")]),
    ]);
    let agent = AgentLoop::new(&client, MockExecutor::new("ok")).max_turns(2);
    let err = agent.run("loop").await.unwrap_err();

    assert!(err.to_string().contains("exceeded max turns (2)"));
    assert_eq!(client.calls(), 2);
}

#[tokio::test]
async fn usage_accumulated_across_turns() {
    let client = MockClient::new(vec![
        tool_response(vec![make_tool_call("c1", "Bash", "{}")]),
        text_response("done"),
    ]);
    let agent = AgentLoop::new(&client, MockExecutor::new("ok"));
    let output = agent.run("go").await.unwrap();

    assert_eq!(output.usage.input_tokens, 20);
    assert_eq!(output.usage.output_tokens, 10);
}

#[tokio::test]
async fn observer_receives_each_turn_with_accumulated_usage() {
    struct RecordingObserver {
        turns: Arc<Mutex<Vec<(u32, String, u64)>>>,
    }

    impl TurnObserver for RecordingObserver {
        fn on_turn(&self, turn: u32, response: &Response, usage: &Usage) {
            self.turns.lock().unwrap().push((
                turn,
                response.finish_reason.clone(),
                usage.input_tokens,
            ));
        }
    }

    let observer = RecordingObserver {
        turns: Arc::new(Mutex::new(Vec::new())),
    };
    let turns = Arc::clone(&observer.turns);
    let client = MockClient::new(vec![
        tool_response(vec![make_tool_call("c1", "Read", "{}")]),
        text_response("done"),
    ]);

    AgentLoop::new(&client, MockExecutor::new("ok"))
        .with_observer(observer)
        .run("watch")
        .await
        .unwrap();

    let turns = turns.lock().unwrap();
    assert_eq!(
        turns.as_slice(),
        &[
            (0, "tool_calls".to_string(), 10),
            (1, "stop".to_string(), 20)
        ]
    );
}

#[tokio::test]
async fn tools_json_is_forwarded_to_client_each_turn() {
    let tools = serde_json::json!([
        {
            "name": "Read",
            "description": "read a file",
            "input_schema": {"type": "object"}
        }
    ]);
    let client = CapturingClient::new(vec![
        tool_response(vec![make_tool_call("c1", "Read", "{}")]),
        text_response("done"),
    ]);

    AgentLoop::new(&client, MockExecutor::new("ok"))
        .tools_json(tools.clone())
        .run("read")
        .await
        .unwrap();

    let captured = client.tools.lock().unwrap();
    assert_eq!(captured.as_slice(), &[Some(tools.clone()), Some(tools)]);
}

#[tokio::test]
async fn reasoning_parts_collected() {
    let client = MockClient::new(vec![reasoning_then_text(
        "let me think",
        "the answer is 42",
    )]);
    let agent = AgentLoop::new(&client, NoOpExecutor);
    let output = agent.run("meaning of life").await.unwrap();

    assert_eq!(output.text(), "the answer is 42");
    assert!(output.parts.iter().any(|p| matches!(p, Part::Reasoning(_))));
}

#[tokio::test]
async fn all_parts_collected_across_turns() {
    let client = MockClient::new(vec![
        tool_response_with_text("reading", vec![make_tool_call("c1", "Read", "{}")]),
        text_response("file says hello"),
    ]);
    let agent = AgentLoop::new(&client, MockExecutor::new("hello"));
    let output = agent.run("read it").await.unwrap();

    assert_eq!(output.parts.len(), 3);
    assert!(matches!(&output.parts[0], Part::Text(t) if t == "reading"));
    assert!(matches!(&output.parts[1], Part::ToolUse(_)));
    assert!(matches!(&output.parts[2], Part::Text(t) if t == "file says hello"));
}

#[tokio::test]
async fn agent_output_text_returns_last_text() {
    let client = MockClient::new(vec![
        tool_response_with_text("thinking", vec![make_tool_call("c1", "Bash", "{}")]),
        text_response("final answer"),
    ]);
    let agent = AgentLoop::new(&client, MockExecutor::new("ok"));
    let output = agent.run("go").await.unwrap();

    assert_eq!(output.text(), "final answer");
}

#[tokio::test]
async fn client_error_propagates() {
    struct FailClient;

    #[async_trait::async_trait]
    impl ChatClient for FailClient {
        async fn chat(
            &self,
            _: &[ChatMessage],
            _: Option<&serde_json::Value>,
        ) -> Result<(Response, Usage), Box<dyn std::error::Error + Send + Sync>> {
            Err("API error".into())
        }
    }

    let err = AgentLoop::new(&FailClient, NoOpExecutor)
        .run("hi")
        .await
        .unwrap_err();
    assert_eq!(err.to_string(), "API error");
}
