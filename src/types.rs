//! Core types for the architecture analysis tool

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Represents a parsed item from source code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedItem {
    pub kind: ItemKind,
    pub name: String,
    pub visibility: Visibility,
    pub span: Span,
    pub file_path: PathBuf,
    pub attributes: Vec<String>,
    pub doc_comment: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ItemKind {
    Function {
        is_async: bool,
        parameters: Vec<Parameter>,
        return_type: Option<String>,
    },
    Struct {
        fields: Vec<StructField>,
        is_tuple: bool,
    },
    Enum {
        variants: Vec<EnumVariant>,
    },
    Trait {
        methods: Vec<String>,
        supertraits: Vec<String>,
    },
    Impl {
        self_type: String,
        trait_name: Option<String>,
        methods: Vec<String>,
    },
    Mod {
        inline: bool,
    },
    Use {
        path: String,
    },
    Const {
        ty: String,
    },
    Static {
        ty: String,
        is_mut: bool,
    },
    TypeAlias {
        ty: String,
    },
    Macro {
        is_declarative: bool,
    },
    /// Represents an item that failed to parse
    Unknown {
        raw_text: String,
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Parameter {
    pub name: String,
    pub ty: String,
    pub is_self: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StructField {
    pub name: Option<String>,
    pub ty: String,
    pub visibility: Visibility,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnumVariant {
    pub name: String,
    pub fields: Vec<StructField>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Crate,
    Super,
    Private,
    Restricted,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Span {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl Default for Span {
    fn default() -> Self {
        Self {
            start_line: 0,
            start_col: 0,
            end_line: 0,
            end_col: 0,
        }
    }
}

/// Result of parsing a file, includes both successful and failed items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedFile {
    pub path: PathBuf,
    pub items: Vec<ParsedItem>,
    pub parse_errors: Vec<ParseError>,
    pub module_path: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseError {
    pub message: String,
    pub span: Option<Span>,
    pub raw_text: String,
}

/// Dependency information from Cargo.lock
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateDependency {
    pub name: String,
    pub version: String,
    pub source: Option<String>,
    pub registry_path: Option<PathBuf>,
    pub public_api: Vec<ParsedItem>,
}

/// Mapping from crate names to their resolved locations
#[derive(Debug, Default)]
pub struct DependencyMap {
    pub crates: HashMap<String, CrateDependency>,
    pub registry_path: PathBuf,
}

/// Work-site score for semantic gravity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkSiteScore {
    pub item: ParsedItem,
    pub score: f64,
    pub factors: ScoreFactors,
    /// Recursive context window for richer search results
    pub context: ContextEnvelope,
}

/// Recursive Context Window - enriched context for search results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContextEnvelope {
    /// Full module path breadcrumb (e.g., "crate::gravity::scoring")
    pub breadcrumbs: String,
    /// Other types/functions in the same file that share generics or are closely related
    pub siblings: Vec<SiblingInfo>,
    /// Expanded generic bounds without opening the file
    pub generic_bounds: Vec<GenericBound>,
    /// Parent item if this is nested (e.g., method inside impl block)
    pub parent_context: Option<String>,
}

/// Information about a sibling item in the same file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiblingInfo {
    pub name: String,
    pub kind: String,
    pub line: usize,
    /// Shared generic parameters with the main item
    pub shared_generics: Vec<String>,
}

/// A generic bound extracted from a type signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericBound {
    /// The generic parameter name (e.g., "T")
    pub param: String,
    /// The bound constraints (e.g., ["Serialize", "Clone"])
    pub bounds: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScoreFactors {
    /// Distance from entry point (main.rs or lib.rs)
    pub entry_distance: usize,
    /// Number of times this item is referenced
    pub call_count: usize,
    /// Whether this is a "site" (few callers) or "utility" (many callers)
    pub is_site: bool,
    /// Related impl blocks found
    pub impl_count: usize,
    /// Trait implementations
    pub trait_impls: Vec<String>,
    /// Number of unique modules that call this item (cross-module usage)
    pub cross_module_count: usize,
    /// Generic complexity depth (e.g., Vec<HashMap<K, V>> = 2)
    pub generic_depth: usize,
    /// Whether this item is a test function
    pub is_test: bool,
}

/// Reference to an external dependency usage in local code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalReference {
    /// The external symbol being used (e.g., "tokio::spawn")
    pub external_path: String,
    /// File where it's used
    pub file: PathBuf,
    /// Line number
    pub line: usize,
    /// The function/context where it's called
    pub caller_context: String,
    /// Complexity score (based on surrounding code)
    pub complexity: usize,
}

/// Map of external symbols to their local usages
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReferenceMap {
    /// Maps "crate::path" -> list of local usages
    pub references: HashMap<String, Vec<ExternalReference>>,
}

/// Project-wide analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectAnalysis {
    pub files: Vec<ParsedFile>,
    pub dependencies: Vec<CrateDependency>,
    pub call_graph: CallGraph,
    pub module_tree: ModuleTree,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CallGraph {
    /// Maps function names to list of call sites
    pub callers: HashMap<String, Vec<CallSite>>,
    /// Maps function names to what they call
    pub callees: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallSite {
    pub caller: String,
    pub file: PathBuf,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModuleTree {
    pub root: ModuleNode,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModuleNode {
    pub name: String,
    pub path: PathBuf,
    pub children: Vec<ModuleNode>,
    pub depth: usize,
}
