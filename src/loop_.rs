use crate::client::ChatClient;
use crate::types::{ChatMessage, Part, Response, ToolResult, Usage};

/// Callback for executing tool calls.
#[async_trait::async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, name: &str, arguments: &str) -> String;
}

/// Output of a completed agent loop.
pub struct AgentOutput {
    /// All parts from every turn, in order.
    pub parts: Vec<Part>,
    pub usage: Usage,
}

impl AgentOutput {
    /// Final text output (last text parts concatenated).
    pub fn text(&self) -> String {
        // Collect text parts from the last response (after the last tool result).
        let mut last_text = String::new();
        for part in self.parts.iter().rev() {
            match part {
                Part::Text(t) => last_text = format!("{t}{last_text}"),
                Part::ToolUse(_) => break,
                _ => {}
            }
        }
        last_text
    }
}

/// Drives the multi-turn loop: send → parts → execute tools → feed back → repeat.
pub struct AgentLoop<C, T> {
    client: C,
    executor: T,
    tools_json: Option<serde_json::Value>,
    system_prompt: Option<String>,
    max_turns: u32,
}

impl<C: ChatClient, T: ToolExecutor> AgentLoop<C, T> {
    pub fn new(client: C, executor: T) -> Self {
        Self {
            client,
            executor,
            tools_json: None,
            system_prompt: None,
            max_turns: 20,
        }
    }

    pub fn tools_json(mut self, json: serde_json::Value) -> Self {
        self.tools_json = Some(json);
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn max_turns(mut self, n: u32) -> Self {
        self.max_turns = n;
        self
    }

    /// Run the agentic loop to completion.
    pub async fn run(
        &self,
        prompt: &str,
    ) -> Result<AgentOutput, Box<dyn std::error::Error + Send + Sync>> {
        let mut messages = build_messages(self.system_prompt.as_deref(), prompt);
        let mut total_usage = Usage::default();
        let mut all_parts = Vec::new();

        for turn in 0..self.max_turns {
            tracing::info!(turn, messages = messages.len(), "agent loop turn");
            let (response, usage) = self
                .client
                .chat(&messages, self.tools_json.as_ref())
                .await?;
            total_usage.accumulate(&usage);
            all_parts.extend(response.parts.clone());

            if !response.has_tool_calls() {
                return Ok(AgentOutput {
                    parts: all_parts,
                    usage: total_usage,
                });
            }

            let tool_calls: Vec<_> = response.tool_calls().into_iter().cloned().collect();
            let names: Vec<&str> = tool_calls.iter().map(|c| c.function.name.as_str()).collect();
            tracing::info!(turn, "tool calls: {:?}", names);

            let results = execute_tools(&self.executor, &tool_calls).await;
            append_turn(&mut messages, &response, results);
        }

        Err(format!("exceeded max turns ({})", self.max_turns).into())
    }
}

fn build_messages(system_prompt: Option<&str>, user_prompt: &str) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if let Some(sp) = system_prompt {
        messages.push(ChatMessage {
            role: "system".into(),
            content: Some(sp.into()),
            tool_calls: None,
            tool_call_id: None,
        });
    }
    messages.push(ChatMessage {
        role: "user".into(),
        content: Some(user_prompt.into()),
        tool_calls: None,
        tool_call_id: None,
    });
    messages
}

async fn execute_tools(
    executor: &dyn ToolExecutor,
    calls: &[crate::types::ToolCall],
) -> Vec<ToolResult> {
    let mut results = Vec::with_capacity(calls.len());
    for call in calls {
        let output = executor.execute(&call.function.name, &call.function.arguments).await;
        tracing::info!(tool = %call.function.name, "result: {} bytes", output.len());
        results.push(ToolResult {
            tool_call_id: call.id.clone(),
            output,
        });
    }
    results
}

fn append_turn(
    messages: &mut Vec<ChatMessage>,
    response: &Response,
    results: Vec<ToolResult>,
) {
    let text = response.text();
    let calls: Vec<_> = response.tool_calls().into_iter().cloned().collect();

    messages.push(ChatMessage {
        role: "assistant".into(),
        content: text,
        tool_calls: Some(calls),
        tool_call_id: None,
    });
    for result in results {
        messages.push(ChatMessage {
            role: "tool".into(),
            content: Some(result.output),
            tool_calls: None,
            tool_call_id: Some(result.tool_call_id),
        });
    }
}
