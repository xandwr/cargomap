//! MCP Server module for cargomap
//!
//! Provides an MCP (Model Context Protocol) server that exposes cargomap's
//! code analysis capabilities as tools for LLM clients.

use async_trait::async_trait;
use rust_mcp_sdk::McpServer;
use rust_mcp_sdk::macros::{JsonSchema, mcp_tool};
use rust_mcp_sdk::mcp_server::ServerHandler;
use rust_mcp_sdk::schema::{
    CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams, RpcError,
    TextContent, schema_utils::CallToolError,
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
        _runtime: Arc<dyn McpServer>,
    ) -> Result<CallToolResult, CallToolError> {
        let tool_params: CargomapTools =
            CargomapTools::try_from(params).map_err(CallToolError::new)?;

        match tool_params {
            CargomapTools::AnalyzeStruct(tool) => tool.call_tool(&self.project_root),
            CargomapTools::SearchCode(tool) => tool.call_tool(&self.project_root),
            CargomapTools::GetSummary(tool) => tool.call_tool(&self.project_root),
            CargomapTools::FindCallers(tool) => tool.call_tool(&self.project_root),
            CargomapTools::GetExternalUsages(tool) => tool.call_tool(&self.project_root),
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

// Generate the tool_box enum
tool_box!(
    CargomapTools,
    [
        AnalyzeStruct,
        SearchCode,
        GetSummary,
        FindCallers,
        GetExternalUsages
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
