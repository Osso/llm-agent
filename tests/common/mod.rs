use llm_agent::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

pub fn make_tool_call(id: &str, name: &str, args: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        call_type: "function".into(),
        function: FunctionCall {
            name: name.into(),
            arguments: args.into(),
        },
    }
}

pub fn text_response(text: &str) -> Response {
    Response {
        parts: vec![Part::Text(text.into())],
        finish_reason: "stop".into(),
    }
}

pub fn tool_response(calls: Vec<ToolCall>) -> Response {
    let parts = calls.into_iter().map(Part::ToolUse).collect();
    Response {
        parts,
        finish_reason: "tool_calls".into(),
    }
}

pub fn tool_response_with_text(text: &str, calls: Vec<ToolCall>) -> Response {
    let mut parts: Vec<Part> = vec![Part::Text(text.into())];
    parts.extend(calls.into_iter().map(Part::ToolUse));
    Response {
        parts,
        finish_reason: "tool_calls".into(),
    }
}

pub fn reasoning_then_text(reasoning: &str, text: &str) -> Response {
    Response {
        parts: vec![
            Part::Reasoning(reasoning.into()),
            Part::Text(text.into()),
        ],
        finish_reason: "stop".into(),
    }
}

/// Mock client that returns a sequence of responses.
pub struct MockClient {
    responses: std::sync::Mutex<Vec<Response>>,
    call_count: AtomicU32,
}

impl MockClient {
    pub fn new(responses: Vec<Response>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
            call_count: AtomicU32::new(0),
        }
    }

    pub fn calls(&self) -> u32 {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl ChatClient for MockClient {
    async fn chat(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&serde_json::Value>,
    ) -> Result<(Response, Usage), Box<dyn std::error::Error + Send + Sync>> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let resp = self.responses.lock().unwrap().remove(0);
        Ok((resp, Usage { input_tokens: 10, output_tokens: 5, reasoning_tokens: 0 }))
    }
}

/// Mock executor that records calls and returns a fixed output.
pub struct MockExecutor {
    calls: std::sync::Mutex<Vec<(String, String)>>,
    output: String,
}

impl MockExecutor {
    pub fn new(output: &str) -> Self {
        Self {
            calls: std::sync::Mutex::new(Vec::new()),
            output: output.into(),
        }
    }

    pub fn called(&self) -> Vec<(String, String)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl ToolExecutor for MockExecutor {
    async fn execute(&self, name: &str, arguments: &str) -> String {
        self.calls.lock().unwrap().push((name.into(), arguments.into()));
        self.output.clone()
    }
}

pub struct NoOpExecutor;

#[async_trait::async_trait]
impl ToolExecutor for NoOpExecutor {
    async fn execute(&self, _name: &str, _arguments: &str) -> String {
        String::new()
    }
}

/// Client that captures all message arrays passed to chat().
pub struct CapturingClient {
    pub messages: Arc<std::sync::Mutex<Vec<Vec<ChatMessage>>>>,
    responses: std::sync::Mutex<Vec<Response>>,
}

impl CapturingClient {
    pub fn new(responses: Vec<Response>) -> Self {
        Self {
            messages: Arc::new(std::sync::Mutex::new(Vec::new())),
            responses: std::sync::Mutex::new(responses),
        }
    }
}

#[async_trait::async_trait]
impl ChatClient for CapturingClient {
    async fn chat(
        &self,
        msgs: &[ChatMessage],
        _tools: Option<&serde_json::Value>,
    ) -> Result<(Response, Usage), Box<dyn std::error::Error + Send + Sync>> {
        self.messages.lock().unwrap().push(msgs.to_vec());
        let resp = self.responses.lock().unwrap().remove(0);
        Ok((resp, Usage::default()))
    }
}
