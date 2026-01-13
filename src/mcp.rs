//! MCP Server module for cargomap
//!
//! Provides an MCP (Model Context Protocol) server that exposes cargomap's
//! code analysis capabilities as tools for LLM clients.

use async_trait::async_trait;
use rust_mcp_sdk::McpServer;
use rust_mcp_sdk::macros::{JsonSchema, mcp_tool};
use rust_mcp_sdk::mcp_server::ServerHandler;
use rust_mcp_sdk::schema::{
    CallToolRequestParams, CallToolResult, CreateMessageContent, CreateMessageRequestParams,
    ListToolsResult, ModelPreferences, PaginatedRequestParams, Role, RpcError, SamplingMessage,
    SamplingMessageContent, TextContent, schema_utils::CallToolError,
};
use rust_mcp_sdk::tool_box;
use std::path::PathBuf;
use std::sync::Arc;

use crate::SemanticGravity;

/// MCP Server handler for cargomap analysis tools
pub struct CargomapServerHandler {
    project_root: PathBuf,
}

impl CargomapServerHandler {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }
}

#[async_trait]
impl ServerHandler for CargomapServerHandler {
    async fn handle_list_tools_request(
        &self,
        _params: Option<PaginatedRequestParams>,
        _runtime: Arc<dyn McpServer>,
    ) -> Result<ListToolsResult, RpcError> {
        Ok(ListToolsResult {
            meta: None,
            next_cursor: None,
            tools: CargomapTools::tools(),
        })
    }

    async fn handle_call_tool_request(
        &self,
        params: CallToolRequestParams,
        runtime: Arc<dyn McpServer>,
    ) -> Result<CallToolResult, CallToolError> {
        let tool_params: CargomapTools =
            CargomapTools::try_from(params).map_err(CallToolError::new)?;

        match tool_params {
            CargomapTools::AnalyzeStruct(tool) => tool.call_tool(&self.project_root),
            CargomapTools::SearchCode(tool) => tool.call_tool(&self.project_root),
            CargomapTools::GetSummary(tool) => tool.call_tool(&self.project_root),
            CargomapTools::FindCallers(tool) => tool.call_tool(&self.project_root),
            CargomapTools::GetExternalUsages(tool) => tool.call_tool(&self.project_root),
            CargomapTools::DiagnoseTraitBound(tool) => tool.call_tool(&self.project_root),
            // AuditImpact requires async + runtime for LLM sampling
            CargomapTools::AuditImpact(tool) => {
                tool.call_tool_async(&self.project_root, runtime).await
            }
            // Introspect requires async + runtime for dialectical LLM loop
            CargomapTools::Introspect(tool) => {
                tool.call_tool_async(&self.project_root, runtime).await
            }
        }
    }
}

// ==================== Tools ====================

/// Analyze a struct in the Rust project
#[mcp_tool(
    name = "analyze_struct",
    description = "Analyzes a struct in the Rust project and returns detailed information including implementations, trait impls, and usage patterns. Use this to understand a type's role in the codebase.",
    read_only_hint = true
)]
#[derive(Debug, serde::Deserialize, serde::Serialize, JsonSchema)]
pub struct AnalyzeStruct {
    /// The name of the struct to analyze
    struct_name: String,
}

impl AnalyzeStruct {
    pub fn call_tool(&self, project_root: &PathBuf) -> Result<CallToolResult, CallToolError> {
        let mut gravity = SemanticGravity::new();
        gravity
            .analyze_project(project_root)
            .map_err(|e| CallToolError::from_message(e.to_string()))?;

        let results = gravity.search(&self.struct_name);
        let struct_results: Vec<_> = results
            .iter()
            .filter(|r| {
                matches!(
                    r.item.kind,
                    crate::types::ItemKind::Struct { .. } | crate::types::ItemKind::Enum { .. }
                )
            })
            .collect();

        if struct_results.is_empty() {
            return Ok(CallToolResult::text_content(vec![TextContent::from(
                format!(
                    "No struct or enum named '{}' found in the project.",
                    self.struct_name
                ),
            )]));
        }

        let mut output = String::new();
        for result in struct_results.iter().take(3) {
            output.push_str(&format!("## {}\n\n", result.item.name));
            output.push_str(&format!(
                "**File:** {}:{}\n",
                result.item.file_path.display(),
                result.item.span.start_line
            ));
            output.push_str(&format!("**Path:** {}\n", result.context.breadcrumbs));
            output.push_str(&format!("**Score:** {:.1}\n\n", result.score));

            // Show fields for structs
            if let crate::types::ItemKind::Struct { fields, .. } = &result.item.kind {
                if !fields.is_empty() {
                    output.push_str("### Fields\n");
                    for field in fields {
                        let name = field.name.as_deref().unwrap_or("_");
                        output.push_str(&format!("- `{}`: `{}`\n", name, field.ty));
                    }
                    output.push_str("\n");
                }
            }

            // Show impl info
            if result.factors.impl_count > 0 {
                output.push_str(&format!("### Implementations\n"));
                output.push_str(&format!("- {} impl block(s)\n", result.factors.impl_count));
                if !result.factors.trait_impls.is_empty() {
                    output.push_str(&format!(
                        "- Traits: {}\n",
                        result.factors.trait_impls.join(", ")
                    ));
                }
                output.push_str("\n");
            }

            // Show generic bounds
            if !result.context.generic_bounds.is_empty() {
                output.push_str("### Generic Bounds\n");
                for bound in &result.context.generic_bounds {
                    if bound.bounds.is_empty() {
                        output.push_str(&format!("- `{}`\n", bound.param));
                    } else {
                        output.push_str(&format!(
                            "- `{}`: {}\n",
                            bound.param,
                            bound.bounds.join(" + ")
                        ));
                    }
                }
                output.push_str("\n");
            }

            // Show related items
            let related: Vec<_> = result
                .context
                .siblings
                .iter()
                .filter(|s| !s.shared_generics.is_empty())
                .take(5)
                .collect();
            if !related.is_empty() {
                output.push_str("### Related (shared generics)\n");
                for sib in related {
                    output.push_str(&format!(
                        "- {} `{}` (shares: {})\n",
                        sib.kind,
                        sib.name,
                        sib.shared_generics.join(", ")
                    ));
                }
                output.push_str("\n");
            }

            // Usage stats
            output.push_str("### Usage Stats\n");
            output.push_str(&format!(
                "- Cross-module usage: {}\n",
                result.factors.cross_module_count
            ));
            output.push_str(&format!("- Call count: {}\n", result.factors.call_count));
            output.push_str(&format!(
                "- Generic depth: {}\n",
                result.factors.generic_depth
            ));
        }

        Ok(CallToolResult::text_content(vec![TextContent::from(
            output,
        )]))
    }
}

/// Search for code items by name or pattern
#[mcp_tool(
    name = "search_code",
    description = "Search for functions, structs, enums, traits, and other items in the Rust codebase by name. Returns ranked results with semantic gravity scoring.",
    read_only_hint = true
)]
#[derive(Debug, serde::Deserialize, serde::Serialize, JsonSchema)]
pub struct SearchCode {
    /// Search query (matches against item names and doc comments)
    query: String,
    /// Maximum number of results to return (default: 10)
    #[serde(default = "default_limit")]
    limit: Option<u32>,
}

fn default_limit() -> Option<u32> {
    Some(10)
}

impl SearchCode {
    pub fn call_tool(&self, project_root: &PathBuf) -> Result<CallToolResult, CallToolError> {
        let mut gravity = SemanticGravity::new();
        gravity
            .analyze_project(project_root)
            .map_err(|e| CallToolError::from_message(e.to_string()))?;

        let results = gravity.search(&self.query);
        let limit = self.limit.unwrap_or(10) as usize;

        if results.is_empty() {
            return Ok(CallToolResult::text_content(vec![TextContent::from(
                format!("No results found for '{}'.", self.query),
            )]));
        }

        let mut output = format!("# Search Results for '{}'\n\n", self.query);
        output.push_str(&format!(
            "Found {} results (showing top {}):\n\n",
            results.len(),
            limit.min(results.len())
        ));

        for (i, result) in results.iter().take(limit).enumerate() {
            let kind = match &result.item.kind {
                crate::types::ItemKind::Function { .. } => "fn",
                crate::types::ItemKind::Struct { .. } => "struct",
                crate::types::ItemKind::Enum { .. } => "enum",
                crate::types::ItemKind::Trait { .. } => "trait",
                crate::types::ItemKind::Impl { .. } => "impl",
                _ => "item",
            };

            let test_marker = if result.factors.is_test {
                " [TEST]"
            } else {
                ""
            };

            output.push_str(&format!(
                "{}. **{}** `{}`{}\n",
                i + 1,
                kind,
                result.item.name,
                test_marker
            ));
            output.push_str(&format!("   - Path: {}\n", result.context.breadcrumbs));
            output.push_str(&format!(
                "   - File: {}:{}\n",
                result.item.file_path.display(),
                result.item.span.start_line
            ));
            output.push_str(&format!(
                "   - Score: {:.1} (x-mod: {}, generics: {})\n\n",
                result.score, result.factors.cross_module_count, result.factors.generic_depth
            ));
        }

        Ok(CallToolResult::text_content(vec![TextContent::from(
            output,
        )]))
    }
}

/// Get a summary of the project architecture
#[mcp_tool(
    name = "get_summary",
    description = "Get an overview of the Rust project's architecture including file count, item counts, top work sites, and hub functions.",
    read_only_hint = true
)]
#[derive(Debug, serde::Deserialize, serde::Serialize, JsonSchema)]
pub struct GetSummary {}

impl GetSummary {
    pub fn call_tool(&self, project_root: &PathBuf) -> Result<CallToolResult, CallToolError> {
        let mut gravity = SemanticGravity::new();
        gravity
            .analyze_project(project_root)
            .map_err(|e| CallToolError::from_message(e.to_string()))?;

        let summary = gravity.summarize();

        let mut output = String::new();
        output.push_str(&format!(
            "# Project Summary: {}\n\n",
            project_root.display()
        ));
        output.push_str("## Statistics\n\n");
        output.push_str(&format!("| Metric | Count |\n"));
        output.push_str(&format!("|--------|-------|\n"));
        output.push_str(&format!("| Files | {} |\n", summary.total_files));
        output.push_str(&format!("| Functions | {} |\n", summary.total_functions));
        output.push_str(&format!("| Structs | {} |\n", summary.total_structs));
        output.push_str(&format!("| Enums | {} |\n", summary.total_enums));
        output.push_str(&format!("| Traits | {} |\n", summary.total_traits));
        output.push_str(&format!("| Impl blocks | {} |\n", summary.total_impls));
        output.push_str(&format!("| Modules | {} |\n", summary.total_modules));
        output.push_str(&format!(
            "| Parse errors | {} |\n",
            summary.total_parse_errors
        ));
        output.push_str(&format!(
            "| External symbols | {} |\n\n",
            summary.external_usage_count
        ));

        if !summary.hotspots.is_empty() {
            output.push_str("## Top Work Sites\n\n");
            output.push_str("Items with highest semantic gravity scores:\n\n");
            for (i, hs) in summary.hotspots.iter().take(5).enumerate() {
                output.push_str(&format!(
                    "{}. **{}** (score: {:.1}, x-mod: {}, generics: {})\n",
                    i + 1,
                    hs.item.name,
                    hs.score,
                    hs.factors.cross_module_count,
                    hs.factors.generic_depth
                ));
            }
            output.push_str("\n");
        }

        if !summary.hub_functions.is_empty() {
            output.push_str("## Hub Functions\n\n");
            output.push_str("Functions called from multiple modules:\n\n");
            for (name, total, cross_mod) in summary.hub_functions.iter().take(5) {
                output.push_str(&format!(
                    "- **{}**: {} calls from {} modules\n",
                    name, total, cross_mod
                ));
            }
        }

        Ok(CallToolResult::text_content(vec![TextContent::from(
            output,
        )]))
    }
}

/// Find all callers of a function
#[mcp_tool(
    name = "find_callers",
    description = "Find all locations where a function is called in the codebase. Useful for understanding how a function is used.",
    read_only_hint = true
)]
#[derive(Debug, serde::Deserialize, serde::Serialize, JsonSchema)]
pub struct FindCallers {
    /// Name of the function to find callers for
    function_name: String,
}

impl FindCallers {
    pub fn call_tool(&self, project_root: &PathBuf) -> Result<CallToolResult, CallToolError> {
        let mut gravity = SemanticGravity::new();
        gravity
            .analyze_project(project_root)
            .map_err(|e| CallToolError::from_message(e.to_string()))?;

        let callers = gravity.find_call_sites(&self.function_name);

        if callers.is_empty() {
            return Ok(CallToolResult::text_content(vec![TextContent::from(
                format!("No callers found for function '{}'.", self.function_name),
            )]));
        }

        let mut output = format!("# Callers of `{}`\n\n", self.function_name);
        output.push_str(&format!("Found {} call site(s):\n\n", callers.len()));

        for (i, site) in callers.iter().enumerate() {
            output.push_str(&format!(
                "{}. In `{}()` at {}:{}\n",
                i + 1,
                site.caller,
                site.file.display(),
                site.line
            ));
        }

        Ok(CallToolResult::text_content(vec![TextContent::from(
            output,
        )]))
    }
}

/// Get usages of external crate symbols
#[mcp_tool(
    name = "get_external_usages",
    description = "Find where external crate symbols (like tokio::spawn, serde::Serialize) are used in the project. Helps understand external dependencies usage.",
    read_only_hint = true
)]
#[derive(Debug, serde::Deserialize, serde::Serialize, JsonSchema)]
pub struct GetExternalUsages {
    /// External path to search for (e.g., "tokio::spawn", "serde::Serialize")
    external_path: String,
}

impl GetExternalUsages {
    pub fn call_tool(&self, project_root: &PathBuf) -> Result<CallToolResult, CallToolError> {
        let mut gravity = SemanticGravity::new();
        gravity
            .analyze_project(project_root)
            .map_err(|e| CallToolError::from_message(e.to_string()))?;

        let usages = gravity.get_external_usages(&self.external_path);

        if usages.is_empty() {
            // Try to show available external symbols if exact match not found
            let all_externals = gravity.get_all_external_symbols();
            let suggestions: Vec<_> = all_externals
                .iter()
                .filter(|(path, _)| {
                    path.contains(&self.external_path) || self.external_path.contains(path.as_str())
                })
                .take(5)
                .collect();

            let mut output = format!("No usages found for '{}'.\n\n", self.external_path);
            if !suggestions.is_empty() {
                output.push_str("Did you mean one of these?\n");
                for (path, count) in suggestions {
                    output.push_str(&format!("- {} ({} usages)\n", path, count));
                }
            }
            return Ok(CallToolResult::text_content(vec![TextContent::from(
                output,
            )]));
        }

        let mut output = format!("# Usages of `{}`\n\n", self.external_path);
        output.push_str(&format!("Found {} usage(s):\n\n", usages.len()));

        // Sort by complexity for more interesting usages first
        let mut sorted_usages: Vec<_> = usages.iter().collect();
        sorted_usages.sort_by(|a, b| b.complexity.cmp(&a.complexity));

        for (i, usage) in sorted_usages.iter().take(10).enumerate() {
            let complexity_label = match usage.complexity {
                0..=2 => "simple",
                3..=5 => "moderate",
                _ => "complex",
            };
            output.push_str(&format!(
                "{}. In `{}()` at {}:{} [{}]\n",
                i + 1,
                usage.caller_context,
                usage.file.display(),
                usage.line,
                complexity_label
            ));
        }

        if usages.len() > 10 {
            output.push_str(&format!("\n... and {} more usages\n", usages.len() - 10));
        }

        Ok(CallToolResult::text_content(vec![TextContent::from(
            output,
        )]))
    }
}

/// Audit the impact of a proposed code change using LLM analysis
#[mcp_tool(
    name = "audit_impact",
    description = "Analyzes the semantic impact of a proposed change to a struct or function. Uses LLM sampling to reason about whether the change could break business logic at call sites. Example: 'Would changing field `count` from u32 to Option<u32> break callers?'",
    read_only_hint = true
)]
#[derive(Debug, serde::Deserialize, serde::Serialize, JsonSchema)]
pub struct AuditImpact {
    /// Name of the struct or function to audit
    target_name: String,
    /// Description of the proposed change (e.g., "change field `count` from u32 to Option<u32>")
    proposed_change: String,
    /// Maximum number of call sites to analyze (default: 5)
    #[serde(default = "default_max_sites")]
    max_sites: Option<u32>,
}

fn default_max_sites() -> Option<u32> {
    Some(5)
}

impl AuditImpact {
    /// Analyze impact using the MCP sampling feature to ask the LLM
    pub async fn call_tool_async(
        &self,
        project_root: &PathBuf,
        runtime: Arc<dyn McpServer>,
    ) -> Result<CallToolResult, CallToolError> {
        // Check if client supports sampling
        let supports_sampling = runtime.client_supports_sampling().unwrap_or(false);
        if !supports_sampling {
            return Ok(CallToolResult::text_content(vec![TextContent::from(
                "Error: The connected client does not support LLM sampling. \
                 The audit_impact tool requires a client with sampling capabilities \
                 to analyze the semantic impact of code changes."
                    .to_string(),
            )]));
        }

        // First, analyze the project to find dependencies
        let mut gravity = SemanticGravity::new();
        gravity
            .analyze_project(project_root)
            .map_err(|e| CallToolError::from_message(e.to_string()))?;

        // Find the target item
        let results = gravity.search(&self.target_name);
        if results.is_empty() {
            return Ok(CallToolResult::text_content(vec![TextContent::from(
                format!("No item named '{}' found in the project.", self.target_name),
            )]));
        }

        let target = &results[0];

        // Find all call sites for this item
        let call_sites = gravity.find_call_sites(&self.target_name);
        let max_sites = self.max_sites.unwrap_or(5) as usize;

        if call_sites.is_empty() {
            return Ok(CallToolResult::text_content(vec![TextContent::from(
                format!(
                    "No call sites found for '{}'. The change appears safe from a usage perspective, \
                     but manual review is still recommended.",
                    self.target_name
                ),
            )]));
        }

        // Collect context from call sites
        let mut call_site_contexts = Vec::new();
        for site in call_sites.iter().take(max_sites) {
            // Read a few lines around each call site for context
            let context = Self::read_call_site_context(&site.file, site.line);
            call_site_contexts.push(format!(
                "### Call site in `{}()` at {}:{}\n```rust\n{}\n```",
                site.caller,
                site.file.display(),
                site.line,
                context
            ));
        }

        // Build the target description
        let target_description = match &target.item.kind {
            crate::types::ItemKind::Struct { fields, .. } => {
                let field_list: Vec<String> = fields
                    .iter()
                    .map(|f| format!("  {}: {}", f.name.as_deref().unwrap_or("_"), f.ty))
                    .collect();
                format!(
                    "struct {} {{\n{}\n}}",
                    target.item.name,
                    field_list.join(",\n")
                )
            }
            crate::types::ItemKind::Function {
                parameters,
                return_type,
                is_async,
            } => {
                let params: Vec<String> = parameters
                    .iter()
                    .map(|p| format!("{}: {}", p.name, p.ty))
                    .collect();
                let ret = return_type
                    .as_ref()
                    .map(|r| format!(" -> {}", r))
                    .unwrap_or_default();
                let async_kw = if *is_async { "async " } else { "" };
                format!(
                    "{}fn {}({}){}",
                    async_kw,
                    target.item.name,
                    params.join(", "),
                    ret
                )
            }
            _ => format!("{} ({})", target.item.name, target.context.breadcrumbs),
        };

        // Build the prompt for the LLM
        let prompt = format!(
            r#"You are a Rust code safety auditor. Analyze whether the following proposed change could break existing code.

## Target Item
```rust
{target_description}
```
Located at: {file}:{line}

## Proposed Change
{proposed_change}

## Call Sites ({count} of {total} shown)
{call_sites}

## Your Task
1. For each call site, determine if the proposed change would:
   - Cause a compile error
   - Change runtime behavior in potentially breaking ways
   - Require updates to the calling code

2. Rate the overall risk: LOW / MEDIUM / HIGH / CRITICAL

3. Provide specific recommendations for each affected call site.

Respond in a structured format with clear assessments."#,
            target_description = target_description,
            file = target.item.file_path.display(),
            line = target.item.span.start_line,
            proposed_change = self.proposed_change,
            count = call_site_contexts.len(),
            total = call_sites.len(),
            call_sites = call_site_contexts.join("\n\n"),
        );

        // Create the sampling request
        let sampling_params = CreateMessageRequestParams {
            messages: vec![SamplingMessage {
                role: Role::User,
                content: SamplingMessageContent::TextContent(TextContent::from(prompt)),
                meta: None,
            }],
            model_preferences: Some(ModelPreferences {
                hints: vec![],
                cost_priority: Some(0.3),
                speed_priority: Some(0.5),
                intelligence_priority: Some(0.9), // We want good reasoning
            }),
            system_prompt: Some(
                "You are an expert Rust developer and code safety auditor. \
                 Analyze code changes for potential breaking impacts with precision and clarity."
                    .to_string(),
            ),
            max_tokens: 2000,
            include_context: None,
            meta: None,
            metadata: None,
            stop_sequences: vec![],
            task: None,
            temperature: Some(0.3), // Lower temperature for more focused analysis
            tool_choice: None,
            tools: vec![],
        };

        // Call the LLM via sampling
        let result = runtime
            .request_message_creation(sampling_params)
            .await
            .map_err(|e| CallToolError::from_message(format!("Sampling request failed: {}", e)))?;

        // Extract the response text
        let response_text = match &result.content {
            CreateMessageContent::TextContent(text) => text.text.clone(),
            CreateMessageContent::ImageContent(_) => {
                "Error: Received image response instead of text".to_string()
            }
            CreateMessageContent::AudioContent(_) => {
                "Error: Received audio response instead of text".to_string()
            }
            _ => "Error: Received unexpected content type".to_string(),
        };

        // Build the final output
        let mut output = format!("# Impact Audit: `{}`\n\n", self.target_name);
        output.push_str(&format!(
            "**Proposed Change:** {}\n\n",
            self.proposed_change
        ));
        output.push_str(&format!(
            "**Call Sites Analyzed:** {} of {}\n\n",
            call_site_contexts.len(),
            call_sites.len()
        ));
        output.push_str("---\n\n");
        output.push_str("## LLM Analysis\n\n");
        output.push_str(&response_text);

        if call_sites.len() > max_sites {
            output.push_str(&format!(
                "\n\n---\n*Note: {} additional call sites were not analyzed. \
                 Consider increasing `max_sites` for a more comprehensive audit.*",
                call_sites.len() - max_sites
            ));
        }

        Ok(CallToolResult::text_content(vec![TextContent::from(
            output,
        )]))
    }

    /// Read context lines around a call site
    fn read_call_site_context(file: &PathBuf, line: usize) -> String {
        let content = std::fs::read_to_string(file).unwrap_or_default();
        let lines: Vec<&str> = content.lines().collect();

        let start = line.saturating_sub(3);
        let end = (line + 3).min(lines.len());

        lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let line_num = start + i + 1;
                let marker = if line_num == line { ">>>" } else { "   " };
                format!("{} {:4} | {}", marker, line_num, l)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Diagnose why a trait isn't implemented for a struct
#[mcp_tool(
    name = "diagnose_trait_bound",
    description = "Diagnoses why a trait (like Send, Serialize, Clone) isn't implemented for a struct. Analyzes each field to find the 'blocker' - the field type preventing the trait from being derived. Returns specific suggestions for fixing the issue.",
    read_only_hint = true
)]
#[derive(Debug, serde::Deserialize, serde::Serialize, JsonSchema)]
pub struct DiagnoseTraitBound {
    /// The name of the struct to analyze
    struct_name: String,
    /// The trait that's failing (e.g., "Send", "Serialize", "Clone", "Debug")
    trait_name: String,
}

/// Known types that block specific traits
mod trait_blockers {

    /// Types that don't implement Send
    pub const NON_SEND: &[&str] = &[
        "Rc",
        "std::rc::Rc",
        "Weak", // std::rc::Weak
        "*const",
        "*mut",
        "Cell",
        "RefCell",
        "UnsafeCell",
        "MutexGuard",
        "RwLockReadGuard",
        "RwLockWriteGuard",
        "Ref",      // std::cell::Ref
        "RefMut",   // std::cell::RefMut
        "LocalKey", // thread_local
    ];

    /// Types that don't implement Sync
    pub const NON_SYNC: &[&str] = &[
        "Rc",
        "std::rc::Rc",
        "Cell",
        "RefCell",
        "UnsafeCell",
        "*const",
        "*mut",
    ];

    /// Types that don't implement Clone
    pub const NON_CLONE: &[&str] = &[
        "MutexGuard",
        "RwLockReadGuard",
        "RwLockWriteGuard",
        "File",
        "TcpStream",
        "TcpListener",
        "UdpSocket",
        "UnixStream",
        "UnixListener",
        "Stdin",
        "Stdout",
        "Stderr",
        "JoinHandle",
        "Receiver", // mpsc
        "Sender",   // mpsc (SyncSender is Clone though)
    ];

    /// Types that don't implement Serialize (serde)
    pub const NON_SERIALIZE: &[&str] = &[
        "SystemTime",
        "Instant",
        "Duration", // Actually implements, but often problematic
        "File",
        "TcpStream",
        "TcpListener",
        "UdpSocket",
        "Mutex",
        "RwLock",
        "Arc",     // needs T: Serialize
        "Rc",      // needs T: Serialize
        "Box<dyn", // trait objects
        "&dyn",    // trait object refs
        "fn(",     // function pointers
        "impl ",   // impl Trait
        "MutexGuard",
        "RwLockReadGuard",
        "RwLockWriteGuard",
        "JoinHandle",
        "Receiver",
        "Sender",
        "OsString",
        "OsStr",
        "PathBuf", // implements with feature, but often forgotten
        "Path",
        "CString",
        "CStr",
    ];

    /// Types that don't implement Copy
    pub const NON_COPY: &[&str] = &[
        "String",
        "Vec",
        "Box",
        "Rc",
        "Arc",
        "HashMap",
        "HashSet",
        "BTreeMap",
        "BTreeSet",
        "VecDeque",
        "LinkedList",
        "BinaryHeap",
        "PathBuf",
        "OsString",
        "CString",
        "Mutex",
        "RwLock",
        "File",
        "TcpStream",
        "TcpListener",
    ];

    /// Types that don't implement Default
    pub const NON_DEFAULT: &[&str] = &[
        "File",
        "TcpStream",
        "TcpListener",
        "UdpSocket",
        "NonZeroU8",
        "NonZeroU16",
        "NonZeroU32",
        "NonZeroU64",
        "NonZeroU128",
        "NonZeroUsize",
        "NonZeroI8",
        "NonZeroI16",
        "NonZeroI32",
        "NonZeroI64",
        "NonZeroI128",
        "NonZeroIsize",
        "&str",
        "&[",
    ];

    /// Get suggestions for fixing a trait bound issue
    pub fn get_suggestion(trait_name: &str, field_name: &str, field_type: &str) -> String {
        // Normalize the type (parser may add spaces around angle brackets)
        let normalized = field_type.replace(" < ", "<").replace(" > ", ">");

        match trait_name.to_lowercase().as_str() {
            "send" => {
                if normalized.contains("Rc<") {
                    format!(
                        "Field `{}` uses `Rc<T>`. Replace with `Arc<T>` for thread-safe reference counting.",
                        field_name
                    )
                } else if normalized.contains("RefCell") {
                    format!(
                        "Field `{}` uses `RefCell<T>`. For thread-safe interior mutability, use `Mutex<T>` or `RwLock<T>`.",
                        field_name
                    )
                } else if normalized.contains("Cell<") {
                    format!(
                        "Field `{}` uses `Cell<T>`. For thread-safe alternatives, use `AtomicXxx` types or `Mutex<T>`.",
                        field_name
                    )
                } else if normalized.contains("*const") || normalized.contains("*mut") {
                    format!(
                        "Field `{}` contains a raw pointer. Either wrap in a `Send` newtype with unsafe impl, or redesign to avoid raw pointers.",
                        field_name
                    )
                } else {
                    format!(
                        "Field `{}` (type: `{}`) is not Send. Check if there's a thread-safe alternative.",
                        field_name, field_type
                    )
                }
            }
            "sync" => {
                if normalized.contains("Rc<") {
                    format!(
                        "Field `{}` uses `Rc<T>`. Replace with `Arc<T>` for Sync.",
                        field_name
                    )
                } else if field_type.contains("RefCell") || field_type.contains("Cell<") {
                    format!(
                        "Field `{}` uses interior mutability without synchronization. Use `Mutex<T>`, `RwLock<T>`, or atomic types.",
                        field_name
                    )
                } else {
                    format!(
                        "Field `{}` (type: `{}`) is not Sync. Consider thread-safe alternatives.",
                        field_name, field_type
                    )
                }
            }
            "serialize" | "deserialize" => {
                if normalized.contains("SystemTime") {
                    format!(
                        "Field `{}` uses `SystemTime`. Add `#[serde(with = \"humantime_serde\")]` or use `chrono::DateTime` instead.",
                        field_name
                    )
                } else if normalized.contains("Instant") {
                    format!(
                        "Field `{}` uses `Instant` which cannot be serialized (it's relative to process start). Store as Duration or timestamp instead.",
                        field_name
                    )
                } else if normalized.contains("PathBuf") || normalized.contains("Path") {
                    format!(
                        "Field `{}` uses `PathBuf`. Enable serde's `std` feature, or use `#[serde(serialize_with = ...)]` for custom handling.",
                        field_name
                    )
                } else if normalized.contains("Arc<") || normalized.contains("Rc<") {
                    format!(
                        "Field `{}` uses `{}`. Enable the `rc` feature in serde: `serde = {{ version = \"1\", features = [\"rc\"] }}`.",
                        field_name,
                        if normalized.contains("Arc") {
                            "Arc"
                        } else {
                            "Rc"
                        }
                    )
                } else if normalized.contains("Mutex<") || normalized.contains("RwLock<") {
                    format!(
                        "Field `{}` uses a lock type. Mark with `#[serde(skip)]` or extract the inner value for serialization.",
                        field_name
                    )
                } else if normalized.contains("dyn ") || normalized.contains("Box<dyn") {
                    format!(
                        "Field `{}` contains a trait object. Use `#[serde(skip)]`, or implement custom serialization with `typetag` crate.",
                        field_name
                    )
                } else if normalized.contains("fn(") {
                    format!(
                        "Field `{}` is a function pointer. Function pointers cannot be serialized. Use `#[serde(skip)]`.",
                        field_name
                    )
                } else {
                    format!(
                        "Field `{}` (type: `{}`) doesn't implement Serialize. Add `#[serde(skip)]` or implement Serialize for the type.",
                        field_name, field_type
                    )
                }
            }
            "clone" => {
                if normalized.contains("MutexGuard")
                    || normalized.contains("RwLockReadGuard")
                    || normalized.contains("RwLockWriteGuard")
                {
                    format!(
                        "Field `{}` holds a lock guard. Guards cannot be cloned. Restructure to not store guards in structs.",
                        field_name
                    )
                } else if normalized.contains("File")
                    || normalized.contains("TcpStream")
                    || normalized.contains("UdpSocket")
                {
                    format!(
                        "Field `{}` holds an I/O resource. Use `try_clone()` method or wrap in `Arc<Mutex<...>>` for sharing.",
                        field_name
                    )
                } else if normalized.contains("JoinHandle") {
                    format!(
                        "Field `{}` holds a JoinHandle. Handles represent unique ownership of a thread and cannot be cloned.",
                        field_name
                    )
                } else {
                    format!(
                        "Field `{}` (type: `{}`) doesn't implement Clone. Derive Clone for it or use `Arc<T>` for shared ownership.",
                        field_name, field_type
                    )
                }
            }
            "copy" => {
                if normalized.contains("String")
                    || normalized.contains("Vec<")
                    || normalized.contains("Box<")
                {
                    format!(
                        "Field `{}` uses heap-allocated type `{}`. Copy cannot be derived for types with heap allocation. Use Clone instead.",
                        field_name, field_type
                    )
                } else {
                    format!(
                        "Field `{}` (type: `{}`) doesn't implement Copy. Copy requires all fields to be Copy. Consider using Clone.",
                        field_name, field_type
                    )
                }
            }
            "default" => {
                if normalized.contains("NonZero") {
                    format!(
                        "Field `{}` uses a NonZero type which has no default (zero is not valid). Provide a manual Default impl.",
                        field_name
                    )
                } else if normalized.contains("File")
                    || normalized.contains("TcpStream")
                    || normalized.contains("TcpListener")
                {
                    format!(
                        "Field `{}` holds an I/O resource with no default. Use `Option<{}>` with None as default.",
                        field_name, field_type
                    )
                } else {
                    format!(
                        "Field `{}` (type: `{}`) doesn't implement Default. Implement Default manually or wrap in Option.",
                        field_name, field_type
                    )
                }
            }
            "debug" => format!(
                "Field `{}` (type: `{}`) doesn't implement Debug. Derive Debug for it or use `#[derive(Debug)]` on the type.",
                field_name, field_type
            ),
            "eq" | "partialeq" => format!(
                "Field `{}` (type: `{}`) doesn't implement Eq/PartialEq. Derive it or implement manually.",
                field_name, field_type
            ),
            "hash" => {
                if normalized.contains("f32") || normalized.contains("f64") {
                    format!(
                        "Field `{}` is a floating-point type. Floats don't implement Hash due to NaN. Use `ordered_float` crate.",
                        field_name
                    )
                } else if normalized.contains("HashMap") || normalized.contains("HashSet") {
                    format!(
                        "Field `{}` uses a hash-based collection which doesn't implement Hash. Use BTreeMap/BTreeSet or a sorted Vec.",
                        field_name
                    )
                } else {
                    format!(
                        "Field `{}` (type: `{}`) doesn't implement Hash. Derive Hash or implement manually.",
                        field_name, field_type
                    )
                }
            }
            "ord" | "partialord" => {
                if normalized.contains("HashMap") || normalized.contains("HashSet") {
                    format!(
                        "Field `{}` uses an unordered collection. Use BTreeMap/BTreeSet for ordering support.",
                        field_name
                    )
                } else if normalized.contains("f32") || normalized.contains("f64") {
                    format!(
                        "Field `{}` is a float which only implements PartialOrd (NaN is unordered). Use `ordered_float` crate for Ord.",
                        field_name
                    )
                } else {
                    format!(
                        "Field `{}` (type: `{}`) doesn't implement Ord/PartialOrd.",
                        field_name, field_type
                    )
                }
            }
            _ => format!(
                "Field `{}` (type: `{}`) may not implement `{}`.",
                field_name, field_type, trait_name
            ),
        }
    }

    /// Check if a type blocks a specific trait
    pub fn blocks_trait(field_type: &str, trait_name: &str) -> bool {
        // Normalize the type (parser may add spaces around angle brackets)
        let normalized = field_type.replace(" < ", "<").replace(" > ", ">");

        let blockers = match trait_name.to_lowercase().as_str() {
            "send" => NON_SEND,
            "sync" => NON_SYNC,
            "clone" => NON_CLONE,
            "serialize" | "deserialize" => NON_SERIALIZE,
            "copy" => NON_COPY,
            "default" => NON_DEFAULT,
            "hash" => &["f32", "f64", "HashMap", "HashSet"][..],
            "ord" => &["f32", "f64", "HashMap", "HashSet"][..],
            _ => return false,
        };

        for blocker in blockers {
            if normalized.contains(blocker) {
                return true;
            }
        }
        false
    }
}

impl DiagnoseTraitBound {
    pub fn call_tool(&self, project_root: &PathBuf) -> Result<CallToolResult, CallToolError> {
        let mut gravity = SemanticGravity::new();
        gravity
            .analyze_project(project_root)
            .map_err(|e| CallToolError::from_message(e.to_string()))?;

        // Search for the struct
        let results = gravity.search(&self.struct_name);
        let struct_result = results.iter().find(|r| {
            matches!(r.item.kind, crate::types::ItemKind::Struct { .. })
                && r.item.name == self.struct_name
        });

        let Some(result) = struct_result else {
            // Try fuzzy match
            let struct_results: Vec<_> = results
                .iter()
                .filter(|r| matches!(r.item.kind, crate::types::ItemKind::Struct { .. }))
                .take(5)
                .collect();

            let mut output = format!(
                "No struct named '{}' found in the project.\n",
                self.struct_name
            );
            if !struct_results.is_empty() {
                output.push_str("\nDid you mean one of these?\n");
                for r in struct_results {
                    output.push_str(&format!("- {}\n", r.item.name));
                }
            }
            return Ok(CallToolResult::text_content(vec![TextContent::from(
                output,
            )]));
        };

        let crate::types::ItemKind::Struct { fields, .. } = &result.item.kind else {
            return Ok(CallToolResult::text_content(vec![TextContent::from(
                format!("'{}' is not a struct.", self.struct_name),
            )]));
        };

        // Analyze each field for trait blockers
        let mut blockers: Vec<(String, String, String)> = Vec::new();
        let mut warnings: Vec<(String, String, String)> = Vec::new();

        for field in fields {
            let field_name = field.name.as_deref().unwrap_or("_unnamed");
            let field_type = &field.ty;

            if trait_blockers::blocks_trait(field_type, &self.trait_name) {
                let suggestion =
                    trait_blockers::get_suggestion(&self.trait_name, field_name, field_type);
                blockers.push((field_name.to_string(), field_type.clone(), suggestion));
            } else if self.might_be_problematic(field_type) {
                // Check for generic types that might not implement the trait
                warnings.push((
                    field_name.to_string(),
                    field_type.clone(),
                    format!(
                        "Generic type `{}` - ensure the type parameter implements `{}`.",
                        field_type, self.trait_name
                    ),
                ));
            }
        }

        // Build output
        let mut output = format!(
            "# Trait Bound Diagnosis: `{}` for `{}`\n\n",
            self.trait_name, self.struct_name
        );
        output.push_str(&format!(
            "**File:** {}:{}\n\n",
            result.item.file_path.display(),
            result.item.span.start_line
        ));

        if blockers.is_empty() && warnings.is_empty() {
            output.push_str(&format!(
                "✅ **No obvious blockers found** for `{}`.\n\n",
                self.trait_name
            ));
            output.push_str("The struct's fields appear compatible with this trait. ");
            output.push_str("If you're still seeing errors, check:\n");
            output.push_str("1. Generic type parameters may need bounds\n");
            output.push_str("2. Nested types within collections may be the issue\n");
            output.push_str("3. The trait may need to be in scope (`use` statement)\n");
            output
                .push_str("4. For derive macros, ensure the trait's derive macro is available\n\n");

            // Show the struct fields for reference
            output.push_str("### Struct Fields\n");
            for field in fields {
                let name = field.name.as_deref().unwrap_or("_");
                output.push_str(&format!("- `{}`: `{}`\n", name, field.ty));
            }
        } else {
            if !blockers.is_empty() {
                output.push_str(&format!(
                    "❌ **Found {} blocking field(s)**\n\n",
                    blockers.len()
                ));

                for (i, (field_name, field_type, suggestion)) in blockers.iter().enumerate() {
                    output.push_str(&format!("### {}. Field `{}`\n", i + 1, field_name));
                    output.push_str(&format!("**Type:** `{}`\n\n", field_type));
                    output.push_str(&format!("**Issue:** {}\n\n", suggestion));
                }
            }

            if !warnings.is_empty() {
                output.push_str(&format!(
                    "⚠️  **{} field(s) to verify**\n\n",
                    warnings.len()
                ));

                for (field_name, field_type, note) in &warnings {
                    output.push_str(&format!(
                        "- `{}` (`{}`): {}\n",
                        field_name, field_type, note
                    ));
                }
                output.push_str("\n");
            }

            // Summary with actionable next steps
            output.push_str("---\n\n### Summary\n\n");
            if !blockers.is_empty() {
                let main_blocker = &blockers[0];
                output.push_str(&format!(
                    "**Primary blocker:** Field `{}` (type: `{}`) is the main reason `{}` fails.\n\n",
                    main_blocker.0, main_blocker.1, self.trait_name
                ));
            }

            // Trait-specific general advice
            output.push_str(&self.get_trait_advice());
        }

        Ok(CallToolResult::text_content(vec![TextContent::from(
            output,
        )]))
    }

    /// Check if a type might be problematic (generic types, trait objects, etc.)
    fn might_be_problematic(&self, ty: &str) -> bool {
        // Single uppercase letter often indicates a generic
        let has_generic = ty
            .chars()
            .any(|c| c.is_uppercase() && ty.matches(c).count() == 1);
        let has_angle_brackets = ty.contains('<');

        // Check for potentially problematic patterns
        (has_generic && has_angle_brackets) || ty.contains("impl ") || ty.contains("dyn ")
    }

    /// Get general advice for specific traits
    fn get_trait_advice(&self) -> String {
        match self.trait_name.to_lowercase().as_str() {
            "send" => "**General advice for Send:**\n\
                - `Send` means the type can be transferred across thread boundaries\n\
                - Types with thread-local or non-atomic interior mutability fail Send\n\
                - Raw pointers are not Send by default (use `unsafe impl Send` carefully)\n\
                - Consider `Arc<T>` instead of `Rc<T>`, `Mutex<T>` instead of `RefCell<T>`\n"
                .to_string(),
            "sync" => "**General advice for Sync:**\n\
                - `Sync` means `&T` can be shared between threads safely\n\
                - Types with unsynchronized interior mutability fail Sync\n\
                - `T: Sync` is equivalent to `&T: Send`\n\
                - Use synchronized primitives (`Mutex`, `RwLock`, `Atomic*`) for Sync\n"
                .to_string(),
            "serialize" | "deserialize" => "**General advice for Serialize/Deserialize:**\n\
                - Ensure `serde` feature flags are enabled for std types\n\
                - Use `#[serde(skip)]` for fields that can't/shouldn't be serialized\n\
                - Use `#[serde(with = \"...\")]` for custom serialization logic\n\
                - Consider `#[serde(default)]` for optional fields\n\
                - For `Arc`/`Rc`, enable the `rc` feature: `serde = { features = [\"rc\"] }`\n"
                .to_string(),
            "clone" => "**General advice for Clone:**\n\
                - Types holding unique resources (files, sockets, threads) can't be cloned\n\
                - Consider `Arc<T>` for shared ownership instead of cloning\n\
                - For `File`/`TcpStream`, use `try_clone()` method if cloning is needed\n"
                .to_string(),
            "copy" => "**General advice for Copy:**\n\
                - Copy is for types that can be duplicated via memcpy\n\
                - Heap-allocated types (String, Vec, Box) cannot be Copy\n\
                - Types with Drop impl cannot be Copy\n\
                - Consider using Clone if Copy isn't possible\n"
                .to_string(),
            "default" => "**General advice for Default:**\n\
                - Not all types have a sensible default value\n\
                - Wrap non-Default fields in `Option<T>` with `#[derive(Default)]`\n\
                - Implement Default manually for complex initialization\n"
                .to_string(),
            "debug" => "**General advice for Debug:**\n\
                - Most std types implement Debug\n\
                - For custom types, `#[derive(Debug)]` usually works\n\
                - For types you don't control, implement Debug manually\n"
                .to_string(),
            _ => format!(
                "**General advice:** Ensure all fields implement `{}`.\n",
                self.trait_name
            ),
        }
    }
}

// ==================== Introspect (Strange Loop) ====================

/// Introspect a symbol to discover its "soul" through dialectical reasoning
#[mcp_tool(
    name = "introspect",
    description = "Discovers the essential nature ('soul') of a struct or function through dialectical reasoning. Uses iterative LLM sampling to generate a thesis about what the symbol IS, then probes the AST for contradictions, forcing re-evaluation until convergence. Returns a 'realization' - a stable, grounded understanding of the symbol's role.",
    read_only_hint = true
)]
#[derive(Debug, serde::Deserialize, serde::Serialize, JsonSchema)]
pub struct Introspect {
    /// The name of the struct or function to introspect
    symbol: String,
    /// Maximum iterations before forcing convergence (default: 5)
    #[serde(default = "default_max_loops")]
    max_loops: Option<u8>,
    /// Similarity threshold for convergence (0.0-1.0, default: 0.85)
    #[serde(default = "default_convergence_threshold")]
    convergence_threshold: Option<f32>,
}

fn default_max_loops() -> Option<u8> {
    Some(5)
}

fn default_convergence_threshold() -> Option<f32> {
    Some(0.85)
}

/// A contradiction found between the thesis and the code
#[derive(Debug)]
struct Contradiction {
    /// What the thesis claimed
    claim: String,
    /// What the code actually shows
    evidence: String,
    /// Location in codebase
    location: String,
}

/// Patterns to probe for in the AST based on thesis claims
#[derive(Debug, Clone)]
enum AstProbe {
    /// Check if struct has mutable state (fields that suggest state)
    HasMutableState,
    /// Check if struct has stateful fields (HashMap, Vec, etc.)
    HasStatefulFields,
    /// Check if any method mutates self
    HasMutatingMethods,
    /// Check if there are side effects (calls to external systems)
    HasSideEffects,
    /// Check for static/global state
    HasStaticState,
    /// Check for async operations
    HasAsyncOps,
    /// Check for error handling patterns
    HasErrorHandling,
    /// Custom pattern to search for
    Custom(String),
}

impl Introspect {
    pub async fn call_tool_async(
        &self,
        project_root: &PathBuf,
        runtime: Arc<dyn McpServer>,
    ) -> Result<CallToolResult, CallToolError> {
        // Check if client supports sampling
        let supports_sampling = runtime.client_supports_sampling().unwrap_or(false);
        if !supports_sampling {
            return Ok(CallToolResult::text_content(vec![TextContent::from(
                "Error: The connected client does not support LLM sampling. \
                 The introspect tool requires sampling capabilities for dialectical reasoning."
                    .to_string(),
            )]));
        }

        // Analyze the project
        let mut gravity = SemanticGravity::new();
        gravity
            .analyze_project(project_root)
            .map_err(|e| CallToolError::from_message(e.to_string()))?;

        // Find the target symbol
        let results = gravity.search(&self.symbol);
        if results.is_empty() {
            return Ok(CallToolResult::text_content(vec![TextContent::from(
                format!("No symbol named '{}' found in the project.", self.symbol),
            )]));
        }

        let target = &results[0];
        let max_loops = self.max_loops.unwrap_or(5) as usize;
        let threshold = self.convergence_threshold.unwrap_or(0.85);

        // Gather context about the symbol
        let symbol_context = self.gather_symbol_context(&gravity, target);

        // The Strange Loop begins
        let mut loop_trace = Vec::new();
        let mut current_thesis = String::new();
        let mut previous_thesis = String::new();
        let mut iteration = 0;

        // Initial thesis generation
        current_thesis = self
            .generate_initial_thesis(&symbol_context, &runtime)
            .await?;
        loop_trace.push(format!(
            "## Iteration 0: Initial Thesis\n{}",
            current_thesis
        ));

        for i in 0..max_loops {
            iteration = i + 1;

            // Generate falsifiable predicates from the thesis
            let probes = self.generate_probes(&current_thesis, &runtime).await?;

            // Probe the AST for contradictions
            let contradictions = self.find_contradictions(&probes, &gravity, target);

            if contradictions.is_empty() {
                // No contradictions found - thesis is consistent with code
                loop_trace.push(format!(
                    "## Iteration {}: No Contradictions\nThesis is consistent with codebase.",
                    iteration
                ));
                break;
            }

            // Format contradictions for the LLM
            let contradiction_text: String = contradictions
                .iter()
                .map(|c| {
                    format!(
                        "- **Claim:** {}\n  **Evidence:** {}\n  **Location:** {}",
                        c.claim, c.evidence, c.location
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n");

            loop_trace.push(format!(
                "## Iteration {}: Contradictions Found\n{}",
                iteration, contradiction_text
            ));

            // Store previous thesis for convergence check
            previous_thesis = current_thesis.clone();

            // Synthesize new thesis with temperature decay
            let temperature = 0.7 - (i as f32 * 0.12); // Decay towards determinism
            current_thesis = self
                .synthesize_thesis(
                    &current_thesis,
                    &contradiction_text,
                    &symbol_context,
                    temperature,
                    &runtime,
                )
                .await?;

            loop_trace.push(format!(
                "## Iteration {}: Revised Thesis (temp={:.2})\n{}",
                iteration, temperature, current_thesis
            ));

            // Check for convergence using embedding similarity
            if !previous_thesis.is_empty() {
                match crate::embedding::has_converged(&current_thesis, &previous_thesis, threshold)
                {
                    Ok(true) => {
                        loop_trace.push(format!(
                            "## Convergence Reached\nSemantic similarity exceeded {} threshold.",
                            threshold
                        ));
                        break;
                    }
                    Ok(false) => {
                        // Check for divergence (thesis growing without substance)
                        if current_thesis.len() > previous_thesis.len() * 2 {
                            loop_trace.push(
                                "## Divergence Detected\nThesis growing without adding substance. Forcing simplification."
                                    .to_string(),
                            );
                            // Force a simplification in next iteration by including this in context
                        }
                    }
                    Err(e) => {
                        // Embedding failed, fall back to simple length check
                        loop_trace.push(format!(
                            "## Convergence Check Failed\nEmbedding error: {}. Using fallback.",
                            e
                        ));
                    }
                }
            }
        }

        // Build the final realization
        let mut output = format!("# Introspection: `{}`\n\n", self.symbol);
        output.push_str(&format!(
            "**Symbol:** {} at {}:{}\n",
            target.item.name,
            target.item.file_path.display(),
            target.item.span.start_line
        ));
        output.push_str(&format!("**Iterations:** {} / {}\n", iteration, max_loops));
        output.push_str(&format!("**Convergence Threshold:** {}\n\n", threshold));

        output.push_str("---\n\n");
        output.push_str("## The Realization\n\n");
        output.push_str(&current_thesis);
        output.push_str("\n\n---\n\n");

        output.push_str("## Dialectical Trace\n\n");
        output.push_str(&loop_trace.join("\n\n---\n\n"));

        Ok(CallToolResult::text_content(vec![TextContent::from(
            output,
        )]))
    }

    /// Gather comprehensive context about a symbol
    fn gather_symbol_context(
        &self,
        gravity: &SemanticGravity,
        target: &crate::types::WorkSiteScore,
    ) -> String {
        let mut context = String::new();

        // Basic info
        context.push_str(&format!("**Name:** {}\n", target.item.name));
        context.push_str(&format!("**Path:** {}\n", target.context.breadcrumbs));
        context.push_str(&format!(
            "**File:** {}:{}\n",
            target.item.file_path.display(),
            target.item.span.start_line
        ));

        // Kind-specific info
        match &target.item.kind {
            crate::types::ItemKind::Struct { fields, .. } => {
                context.push_str("\n**Fields:**\n");
                for field in fields {
                    context.push_str(&format!(
                        "  - {}: {}\n",
                        field.name.as_deref().unwrap_or("_"),
                        field.ty
                    ));
                }
            }
            crate::types::ItemKind::Function {
                parameters,
                return_type,
                is_async,
            } => {
                if *is_async {
                    context.push_str("**Async:** yes\n");
                }
                context.push_str("\n**Parameters:**\n");
                for param in parameters {
                    context.push_str(&format!("  - {}: {}\n", param.name, param.ty));
                }
                if let Some(ret) = return_type {
                    context.push_str(&format!("**Returns:** {}\n", ret));
                }
            }
            crate::types::ItemKind::Enum { variants } => {
                context.push_str("\n**Variants:**\n");
                for variant in variants {
                    context.push_str(&format!("  - {}\n", variant.name));
                }
            }
            _ => {}
        }

        // Doc comment if available
        if let Some(doc) = &target.item.doc_comment {
            context.push_str(&format!("\n**Documentation:**\n{}\n", doc));
        }

        // Find call sites
        let call_sites = gravity.find_call_sites(&self.symbol);
        if !call_sites.is_empty() {
            context.push_str(&format!("\n**Call Sites:** {} found\n", call_sites.len()));
            for site in call_sites.iter().take(5) {
                context.push_str(&format!(
                    "  - {}() at {}:{}\n",
                    site.caller,
                    site.file.display(),
                    site.line
                ));
            }
        }

        // Read the actual source code
        if let Ok(source) = std::fs::read_to_string(&target.item.file_path) {
            let lines: Vec<&str> = source.lines().collect();
            let start = target.item.span.start_line.saturating_sub(1);
            let end = (target.item.span.end_line).min(lines.len());
            if start < end {
                context.push_str("\n**Source Code:**\n```rust\n");
                for line in &lines[start..end] {
                    context.push_str(line);
                    context.push('\n');
                }
                context.push_str("```\n");
            }
        }

        context
    }

    /// Generate the initial thesis about what the symbol IS
    async fn generate_initial_thesis(
        &self,
        context: &str,
        runtime: &Arc<dyn McpServer>,
    ) -> Result<String, CallToolError> {
        let prompt = format!(
            r#"You are analyzing a Rust symbol to understand its essential nature - its "soul".

## Symbol Context
{context}

## Your Task
Describe what this symbol IS at its core. Not what it does mechanically, but its essential PURPOSE and NATURE in the system.

Consider:
1. What role does it play in the architecture?
2. What invariants does it maintain?
3. What would break if this symbol didn't exist?
4. Is it stateful or stateless? A coordinator or a worker?

Be concise but precise. Use 2-4 sentences maximum."#
        );

        self.sample_llm(&prompt, 0.7, 500, runtime).await
    }

    /// Generate AST probes based on the thesis
    async fn generate_probes(
        &self,
        thesis: &str,
        runtime: &Arc<dyn McpServer>,
    ) -> Result<Vec<AstProbe>, CallToolError> {
        let prompt = format!(
            r#"Given this thesis about a Rust symbol:

"{thesis}"

What falsifiable claims does this thesis make? For each claim, specify what code pattern would CONTRADICT it.

Respond with a JSON array of probe types. Valid types:
- "HasMutableState" - contradicts claims of immutability/statelessness
- "HasStatefulFields" - contradicts claims of being a pure function/stateless
- "HasMutatingMethods" - contradicts claims of not modifying state
- "HasSideEffects" - contradicts claims of being pure/side-effect-free
- "HasStaticState" - contradicts claims of no global state
- "HasAsyncOps" - contradicts claims of synchronous operation
- "HasErrorHandling" - contradicts claims of infallibility
- {{"Custom": "pattern"}} - search for specific pattern in source

Example response:
["HasMutableState", "HasStatefulFields", {{"Custom": "unsafe"}}]

Return ONLY the JSON array, no explanation."#
        );

        let response = self.sample_llm(&prompt, 0.3, 200, runtime).await?;

        // Parse the JSON response
        let probes = self.parse_probes(&response);
        Ok(probes)
    }

    /// Parse probe response from LLM
    fn parse_probes(&self, response: &str) -> Vec<AstProbe> {
        // Try to extract JSON array from response
        let json_start = response.find('[');
        let json_end = response.rfind(']');

        if let (Some(start), Some(end)) = (json_start, json_end) {
            let json_str = &response[start..=end];
            if let Ok(parsed) = serde_json::from_str::<Vec<serde_json::Value>>(json_str) {
                return parsed
                    .iter()
                    .filter_map(|v| match v {
                        serde_json::Value::String(s) => match s.as_str() {
                            "HasMutableState" => Some(AstProbe::HasMutableState),
                            "HasStatefulFields" => Some(AstProbe::HasStatefulFields),
                            "HasMutatingMethods" => Some(AstProbe::HasMutatingMethods),
                            "HasSideEffects" => Some(AstProbe::HasSideEffects),
                            "HasStaticState" => Some(AstProbe::HasStaticState),
                            "HasAsyncOps" => Some(AstProbe::HasAsyncOps),
                            "HasErrorHandling" => Some(AstProbe::HasErrorHandling),
                            _ => None,
                        },
                        serde_json::Value::Object(obj) => obj
                            .get("Custom")
                            .and_then(|v| v.as_str())
                            .map(|s| AstProbe::Custom(s.to_string())),
                        _ => None,
                    })
                    .collect();
            }
        }

        // Default probes if parsing fails
        vec![
            AstProbe::HasMutableState,
            AstProbe::HasStatefulFields,
            AstProbe::HasMutatingMethods,
        ]
    }

    /// Find contradictions between probes and actual code
    fn find_contradictions(
        &self,
        probes: &[AstProbe],
        gravity: &SemanticGravity,
        target: &crate::types::WorkSiteScore,
    ) -> Vec<Contradiction> {
        let mut contradictions = Vec::new();

        // Read source code for pattern matching
        let source = std::fs::read_to_string(&target.item.file_path).unwrap_or_default();

        for probe in probes {
            match probe {
                AstProbe::HasMutableState => {
                    // Check for &mut self methods or mutable fields
                    if source.contains("&mut self") {
                        contradictions.push(Contradiction {
                            claim: "stateless/immutable".to_string(),
                            evidence: "Found `&mut self` method - this type mutates its own state"
                                .to_string(),
                            location: target.item.file_path.display().to_string(),
                        });
                    }
                }
                AstProbe::HasStatefulFields => {
                    // Check struct fields for stateful types
                    if let crate::types::ItemKind::Struct { fields, .. } = &target.item.kind {
                        for field in fields {
                            let ty = &field.ty;
                            if ty.contains("HashMap")
                                || ty.contains("Vec")
                                || ty.contains("BTreeMap")
                                || ty.contains("HashSet")
                            {
                                contradictions.push(Contradiction {
                                    claim: "stateless".to_string(),
                                    evidence: format!(
                                        "Field `{}` has type `{}` - a stateful collection",
                                        field.name.as_deref().unwrap_or("_"),
                                        ty
                                    ),
                                    location: format!(
                                        "{}:{}",
                                        target.item.file_path.display(),
                                        target.item.span.start_line
                                    ),
                                });
                            }
                            if ty.contains("Mutex") || ty.contains("RwLock") || ty.contains("Cell")
                            {
                                contradictions.push(Contradiction {
                                    claim: "simple state".to_string(),
                                    evidence: format!(
                                        "Field `{}` uses interior mutability (`{}`)",
                                        field.name.as_deref().unwrap_or("_"),
                                        ty
                                    ),
                                    location: format!(
                                        "{}:{}",
                                        target.item.file_path.display(),
                                        target.item.span.start_line
                                    ),
                                });
                            }
                        }
                    }
                }
                AstProbe::HasMutatingMethods => {
                    // Search for methods that modify state
                    let related_items = gravity.search(&self.symbol);
                    for item in &related_items {
                        if let crate::types::ItemKind::Function { parameters, .. } = &item.item.kind
                        {
                            for param in parameters {
                                if param.name == "self" && param.ty.contains("&mut") {
                                    contradictions.push(Contradiction {
                                        claim: "non-mutating".to_string(),
                                        evidence: format!(
                                            "Method `{}` takes `&mut self`",
                                            item.item.name
                                        ),
                                        location: format!(
                                            "{}:{}",
                                            item.item.file_path.display(),
                                            item.item.span.start_line
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
                AstProbe::HasSideEffects => {
                    // Check for side-effect patterns
                    let patterns = [
                        ("std::fs::", "filesystem operations"),
                        ("std::io::", "I/O operations"),
                        ("std::net::", "network operations"),
                        ("println!", "console output"),
                        ("eprintln!", "error output"),
                        ("log::", "logging"),
                        ("tracing::", "tracing"),
                    ];
                    for (pattern, description) in patterns {
                        if source.contains(pattern) {
                            contradictions.push(Contradiction {
                                claim: "pure/no side effects".to_string(),
                                evidence: format!("Found {} ({})", pattern, description),
                                location: target.item.file_path.display().to_string(),
                            });
                            break; // One is enough
                        }
                    }
                }
                AstProbe::HasStaticState => {
                    // Check for static/global state
                    if source.contains("static mut")
                        || source.contains("lazy_static")
                        || source.contains("OnceLock")
                    {
                        contradictions.push(Contradiction {
                            claim: "no global state".to_string(),
                            evidence: "Found static/global state pattern".to_string(),
                            location: target.item.file_path.display().to_string(),
                        });
                    }
                }
                AstProbe::HasAsyncOps => {
                    // Check for async patterns
                    if source.contains("async fn") || source.contains(".await") {
                        contradictions.push(Contradiction {
                            claim: "synchronous".to_string(),
                            evidence: "Found async operations".to_string(),
                            location: target.item.file_path.display().to_string(),
                        });
                    }
                }
                AstProbe::HasErrorHandling => {
                    // Check for Result/Option patterns suggesting fallibility
                    if source.contains("-> Result<") || source.contains("?;") {
                        contradictions.push(Contradiction {
                            claim: "infallible".to_string(),
                            evidence: "Returns Result type - operation can fail".to_string(),
                            location: target.item.file_path.display().to_string(),
                        });
                    }
                }
                AstProbe::Custom(pattern) => {
                    if source.contains(pattern.as_str()) {
                        contradictions.push(Contradiction {
                            claim: format!("does not use '{}'", pattern),
                            evidence: format!("Found '{}' in source", pattern),
                            location: target.item.file_path.display().to_string(),
                        });
                    }
                }
            }
        }

        contradictions
    }

    /// Synthesize a new thesis incorporating contradictions
    async fn synthesize_thesis(
        &self,
        current_thesis: &str,
        contradictions: &str,
        context: &str,
        temperature: f32,
        runtime: &Arc<dyn McpServer>,
    ) -> Result<String, CallToolError> {
        let prompt = format!(
            r#"You are refining your understanding of a Rust symbol through dialectical reasoning.

## Your Previous Thesis
{current_thesis}

## Contradictions Found
{contradictions}

## Symbol Context (for reference)
{context}

## Your Task
Synthesize a NEW thesis that reconciles your previous understanding with the evidence.
- If the evidence invalidates your claim, revise it
- If the evidence reveals nuance, incorporate it
- If the evidence is about a special case, note the exception

Be precise and concise. 2-4 sentences. Do not simply repeat the previous thesis with minor changes."#
        );

        self.sample_llm(&prompt, temperature, 500, runtime).await
    }

    /// Sample the LLM with given parameters
    async fn sample_llm(
        &self,
        prompt: &str,
        temperature: f32,
        max_tokens: i64,
        runtime: &Arc<dyn McpServer>,
    ) -> Result<String, CallToolError> {
        let sampling_params = CreateMessageRequestParams {
            messages: vec![SamplingMessage {
                role: Role::User,
                content: SamplingMessageContent::TextContent(TextContent::from(prompt.to_string())),
                meta: None,
            }],
            model_preferences: Some(ModelPreferences {
                hints: vec![],
                cost_priority: Some(0.3),
                speed_priority: Some(0.4),
                intelligence_priority: Some(0.8),
            }),
            system_prompt: Some(
                "You are an expert Rust developer performing deep code analysis. \
                 Be precise, concise, and insightful."
                    .to_string(),
            ),
            max_tokens,
            include_context: None,
            meta: None,
            metadata: None,
            stop_sequences: vec![],
            task: None,
            temperature: Some(temperature as f64),
            tool_choice: None,
            tools: vec![],
        };

        let result = runtime
            .request_message_creation(sampling_params)
            .await
            .map_err(|e| CallToolError::from_message(format!("Sampling failed: {}", e)))?;

        match &result.content {
            CreateMessageContent::TextContent(text) => Ok(text.text.clone()),
            _ => Err(CallToolError::from_message(
                "Unexpected response type".to_string(),
            )),
        }
    }
}

// Generate the tool_box enum
tool_box!(
    CargomapTools,
    [
        AnalyzeStruct,
        SearchCode,
        GetSummary,
        FindCallers,
        GetExternalUsages,
        AuditImpact,
        DiagnoseTraitBound,
        Introspect
    ]
);

/// Run the MCP server over stdio
pub async fn run_mcp_server(project_root: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    use rust_mcp_sdk::mcp_server::{McpServerOptions, ServerRuntime, server_runtime};
    use rust_mcp_sdk::schema::{
        Implementation, InitializeResult, ProtocolVersion, ServerCapabilities,
        ServerCapabilitiesTools,
    };
    use rust_mcp_sdk::{StdioTransport, ToMcpServerHandler, TransportOptions};

    let server_details = InitializeResult {
        server_info: Implementation {
            name: "cargomap".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            title: Some("cargomap - Rust Architecture Analysis".into()),
            description: Some("MCP server for analyzing Rust project architecture with semantic gravity ranking".into()),
            icons: vec![],
            website_url: None,
        },
        capabilities: ServerCapabilities {
            tools: Some(ServerCapabilitiesTools { list_changed: None }),
            ..Default::default()
        },
        meta: None,
        instructions: Some("Use the available tools to analyze Rust codebases. Tools include searching for code items, analyzing structs, finding callers, and getting project summaries.".into()),
        protocol_version: ProtocolVersion::V2025_11_25.into(),
    };

    let transport = StdioTransport::new(TransportOptions::default())?;
    let handler = CargomapServerHandler::new(project_root);

    let server: Arc<ServerRuntime> = server_runtime::create_server(McpServerOptions {
        server_details,
        transport,
        handler: handler.to_mcp_server_handler(),
        task_store: None,
        client_task_store: None,
    });

    server.start().await?;
    Ok(())
}
