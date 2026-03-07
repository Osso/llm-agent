use crate::types::{ChatMessage, Response, Usage};

/// Trait for sending messages to an LLM and getting structured responses.
///
/// Implementors handle HTTP transport and response parsing.
/// The agentic loop is handled by `AgentLoop`, not the client.
#[async_trait::async_trait]
pub trait ChatClient: Send + Sync {
    /// Send a conversation and get back either text or tool call requests.
    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> Result<(Response, Usage), Box<dyn std::error::Error + Send + Sync>>;
}

#[async_trait::async_trait]
impl<T: ChatClient> ChatClient for &T {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> Result<(Response, Usage), Box<dyn std::error::Error + Send + Sync>> {
        (**self).chat(messages, tools).await
    }
}
