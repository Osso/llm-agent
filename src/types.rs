use serde::{Deserialize, Serialize};

/// A chat message in the conversation.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A tool call requested by the model.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Result of executing a tool, ready to feed back to the model.
pub struct ToolResult {
    pub tool_call_id: String,
    pub output: String,
}

/// What the model returned for a single turn.
pub enum Response {
    /// Final text response, no more tool calls.
    Text(String),
    /// Model wants to call tools before continuing.
    ToolCalls {
        text: Option<String>,
        calls: Vec<ToolCall>,
    },
}

/// Token usage for a single API call.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl Usage {
    pub fn accumulate(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}
