use llm_agent::*;
use std::sync::Mutex;

pub struct BlockBash;

#[async_trait::async_trait]
impl ToolHook for BlockBash {
    async fn pre_execute(
        &self, ctx: &HookContext<'_>,
    ) -> Result<HookDecision, Box<dyn std::error::Error + Send + Sync>> {
        if ctx.tool_name == "Bash" {
            Ok(HookDecision::Block("Bash is not allowed".into()))
        } else {
            Ok(HookDecision::Allow)
        }
    }
}

pub struct AskForWrite;

#[async_trait::async_trait]
impl ToolHook for AskForWrite {
    async fn pre_execute(
        &self, ctx: &HookContext<'_>,
    ) -> Result<HookDecision, Box<dyn std::error::Error + Send + Sync>> {
        if ctx.tool_name == "Write" {
            Ok(HookDecision::Ask {
                reason: "Write requires confirmation".into(),
                suggestion: None,
            })
        } else {
            Ok(HookDecision::Allow)
        }
    }
}

pub struct SuggestHook;

#[async_trait::async_trait]
impl ToolHook for SuggestHook {
    async fn pre_execute(
        &self, _ctx: &HookContext<'_>,
    ) -> Result<HookDecision, Box<dyn std::error::Error + Send + Sync>> {
        Ok(HookDecision::Ask {
            reason: "dangerous command".into(),
            suggestion: Some("use ls instead".into()),
        })
    }
}

pub struct FailingHook;

#[async_trait::async_trait]
impl ToolHook for FailingHook {
    async fn pre_execute(
        &self, _ctx: &HookContext<'_>,
    ) -> Result<HookDecision, Box<dyn std::error::Error + Send + Sync>> {
        Err("hook crashed".into())
    }
}

pub struct CapturingHook {
    pub captured: Mutex<Vec<(String, String, u32)>>,
}

impl CapturingHook {
    pub fn new() -> Self {
        Self { captured: Mutex::new(Vec::new()) }
    }
}

#[async_trait::async_trait]
impl ToolHook for CapturingHook {
    async fn pre_execute(
        &self, ctx: &HookContext<'_>,
    ) -> Result<HookDecision, Box<dyn std::error::Error + Send + Sync>> {
        self.captured.lock().unwrap().push((
            ctx.tool_name.into(),
            ctx.arguments.into(),
            ctx.turn,
        ));
        Ok(HookDecision::Allow)
    }
}
