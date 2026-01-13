//! cargomap - Rust Architecture Analysis Tool
//!
//! A resilient code analysis tool that provides:
//! - Partial parsing (LSP-Lite) that handles broken code gracefully
//! - Dependency bridge mapping Cargo.lock to registry sources
//! - Semantic gravity ranking for intelligent result ordering
//! - Call-site teleportation (local usage mapping for external symbols)
//! - MCP server for LLM tool integration

pub mod dependency;
pub mod embedding;
pub mod gravity;
pub mod mcp;
pub mod parser;
pub mod types;

pub use dependency::DependencyBridge;
pub use gravity::SemanticGravity;
pub use parser::PartialParser;
pub use types::*;
