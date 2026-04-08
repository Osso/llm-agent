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

/// A typed part of a model response.
#[derive(Clone, Debug)]
pub enum Part {
    /// Text output from the model.
    Text(String),
    /// Extended thinking / reasoning.
    Reasoning(String),
    /// Tool call request.
    ToolUse(ToolCall),
}

/// What the model returned for a single turn — a list of parts.
#[derive(Clone, Debug)]
pub struct Response {
    pub parts: Vec<Part>,
    /// Why the model stopped: "stop", "tool_calls", "length", etc.
    pub finish_reason: String,
}

impl Response {
    /// Extract all text parts concatenated.
    pub fn text(&self) -> Option<String> {
        let texts: Vec<&str> = self
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join(""))
        }
    }

    /// Extract all reasoning parts concatenated.
    pub fn reasoning(&self) -> Option<String> {
        let parts: Vec<&str> = self
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::Reasoning(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(""))
        }
    }

    /// Extract all tool calls.
    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.parts
            .iter()
            .filter_map(|p| match p {
                Part::ToolUse(tc) => Some(tc),
                _ => None,
            })
            .collect()
    }

    /// Whether the model wants to continue (has tool calls).
    pub fn has_tool_calls(&self) -> bool {
        self.parts.iter().any(|p| matches!(p, Part::ToolUse(_)))
    }
}

/// Token usage for a single API call.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
}

impl Usage {
    pub fn accumulate(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
    }
}
