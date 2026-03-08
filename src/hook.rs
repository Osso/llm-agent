/// Decision from a pre-execution hook.
pub enum HookDecision {
    /// Allow the tool to execute.
    Allow,
    /// Block execution. The message is fed back as the tool result.
    Block(String),
    /// Ask for user confirmation with a reason.
    /// Includes an optional suggestion (e.g. a safer alternative command).
    /// If no confirmation handler is set, treated as Block.
    Ask {
        reason: String,
        suggestion: Option<String>,
    },
}

/// Context passed to hooks for each tool call.
pub struct HookContext<'a> {
    pub tool_name: &'a str,
    pub arguments: &'a str,
    pub turn: u32,
}

/// Hook called before each tool execution.
#[async_trait::async_trait]
pub trait ToolHook: Send + Sync {
    async fn pre_execute(
        &self,
        ctx: &HookContext<'_>,
    ) -> Result<HookDecision, Box<dyn std::error::Error + Send + Sync>>;
}

#[async_trait::async_trait]
impl<T: ToolHook> ToolHook for &T {
    async fn pre_execute(
        &self,
        ctx: &HookContext<'_>,
    ) -> Result<HookDecision, Box<dyn std::error::Error + Send + Sync>> {
        (**self).pre_execute(ctx).await
    }
}

/// Observer called after each turn in the agent loop.
pub trait TurnObserver: Send + Sync {
    fn on_turn(&self, turn: u32, response: &crate::types::Response, usage: &crate::types::Usage);
}

/// No-op observer that does nothing.
pub struct NoObserver;

impl TurnObserver for NoObserver {
    fn on_turn(&self, _turn: u32, _response: &crate::types::Response, _usage: &crate::types::Usage) {}
}

/// No-op hook that allows everything.
pub struct AllowAll;

#[async_trait::async_trait]
impl ToolHook for AllowAll {
    async fn pre_execute(
        &self,
        _ctx: &HookContext<'_>,
    ) -> Result<HookDecision, Box<dyn std::error::Error + Send + Sync>> {
        Ok(HookDecision::Allow)
    }
}
