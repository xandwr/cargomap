//! Semantic Gravity - Intelligent ranking of search results
//!
//! Instead of returning a flat list of search results, this module
//! computes a "Work-Site Score" based on:
//! - Distance to entry point (main.rs/lib.rs)
//! - Cross-module usage (more valuable than same-file calls)
//! - Generic complexity depth
//! - Test function detection (deprioritized)
//! - Trait implementations for structs

use crate::parser::PartialParser;
use crate::types::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GravityError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
}

/// Scoring weights based on the factor table
pub mod weights {
    pub const CROSS_MODULE_USAGE: f64 = 50.0;
    pub const PUB_VISIBILITY: f64 = 20.0;
    pub const GENERIC_DEPTH: f64 = 15.0;
    pub const IS_TEST_PENALTY: f64 = -80.0;
    pub const SITE_BONUS: f64 = 30.0;
    pub const UTILITY_PENALTY: f64 = -20.0;
    pub const ENTRY_DISTANCE_PENALTY: f64 = -5.0;
    pub const IMPL_RICHNESS: f64 = 5.0;
    pub const TRAIT_IMPL: f64 = 3.0;
}

/// Standard library / prelude methods to filter out
const PRELUDE_METHODS: &[&str] = &[
    // Iterator methods
    "iter",
    "into_iter",
    "map",
    "filter",
    "collect",
    "fold",
    "find",
    "any",
    "all",
    "take",
    "skip",
    "enumerate",
    "zip",
    "chain",
    "flatten",
    "flat_map",
    "cloned",
    "copied",
    // Option/Result methods
    "unwrap",
    "unwrap_or",
    "unwrap_or_else",
    "unwrap_or_default",
    "expect",
    "ok",
    "err",
    "is_some",
    "is_none",
    "is_ok",
    "is_err",
    "map_err",
    "and_then",
    "or_else",
    "ok_or",
    "ok_or_else",
    // Common trait methods
    "clone",
    "to_string",
    "to_owned",
    "into",
    "from",
    "default",
    "new",
    "len",
    "is_empty",
    "push",
    "pop",
    "get",
    "get_mut",
    "insert",
    "remove",
    "contains",
    "clear",
    "extend",
    "as_ref",
    "as_mut",
    "borrow",
    "borrow_mut",
    "deref",
    "deref_mut",
    // String methods
    "trim",
    "split",
    "starts_with",
    "ends_with",
    "contains",
    "replace",
    "to_lowercase",
    "to_uppercase",
    "chars",
    "bytes",
    "lines",
    // Display/Debug
    "fmt",
    "write",
    "writeln",
    "format",
    "print",
    "println",
    "eprint",
    "eprintln",
    // Comparison
    "eq",
    "ne",
    "cmp",
    "partial_cmp",
    "lt",
    "le",
    "gt",
    "ge",
    "min",
    "max",
    // Memory
    "drop",
    "take",
    "replace",
    "swap",
    "mem",
];

/// Semantic gravity analyzer for ranking code elements
pub struct SemanticGravity {
    parser: PartialParser,
    /// Module tree built from the project
    module_tree: ModuleTree,
    /// Call graph built from analysis
    call_graph: CallGraph,
    /// All parsed files
    files: Vec<ParsedFile>,
    /// Map from type names to their impl blocks
    impl_map: HashMap<String, Vec<ParsedItem>>,
    /// Distance cache from entry point
    distance_cache: HashMap<PathBuf, usize>,
    /// External reference map (crate::path -> local usages)
    reference_map: ReferenceMap,
    /// Module membership for cross-module analysis
    file_to_module: HashMap<PathBuf, String>,
}

impl SemanticGravity {
    pub fn new() -> Self {
        Self {
            parser: PartialParser::new(),
            module_tree: ModuleTree::default(),
            call_graph: CallGraph::default(),
            files: Vec::new(),
            impl_map: HashMap::new(),
            distance_cache: HashMap::new(),
            reference_map: ReferenceMap::default(),
            file_to_module: HashMap::new(),
        }
    }

    /// Analyze a project and build the gravity model
    pub fn analyze_project(&mut self, root: &Path) -> Result<(), GravityError> {
        // Parse all files
        self.files = self
            .parser
            .parse_project(root)
            .map_err(|e| GravityError::Parse(e.to_string()))?;

        // Build file -> module mapping
        self.build_file_module_map();

        // Build module tree
        self.build_module_tree(root);

        // Build impl map
        self.build_impl_map();

        // Build call graph with cross-module tracking
        self.build_call_graph()?;

        // Build external reference map
        self.build_reference_map()?;

        // Compute distances from entry point
        self.compute_distances(root);

        Ok(())
    }

    /// Build mapping from file paths to their module names
    fn build_file_module_map(&mut self) {
        self.file_to_module.clear();

        for file in &self.files {
            let module_name = file.module_path.join("::");
            let module_name = if module_name.is_empty() {
                "crate".to_string()
            } else {
                format!("crate::{}", module_name)
            };
            self.file_to_module.insert(file.path.clone(), module_name);
        }
    }

    /// Build the module tree from parsed files
    fn build_module_tree(&mut self, root: &Path) {
        let mut tree = ModuleTree::default();
        tree.root.name = "crate".to_string();
        tree.root.path = root.to_path_buf();
        tree.root.depth = 0;

        let entry = if root.join("src/lib.rs").exists() {
            root.join("src/lib.rs")
        } else {
            root.join("src/main.rs")
        };

        tree.root.path = entry;

        for file in &self.files {
            for item in &file.items {
                if let ItemKind::Mod { inline } = &item.kind {
                    let depth = file.module_path.len() + 1;
                    let node = ModuleNode {
                        name: item.name.clone(),
                        path: if *inline {
                            file.path.clone()
                        } else {
                            self.resolve_mod_path(&file.path, &item.name)
                        },
                        children: Vec::new(),
                        depth,
                    };
                    tree.root.children.push(node);
                }
            }
        }

        self.module_tree = tree;
    }

    /// Resolve a mod declaration to its file path
    fn resolve_mod_path(&self, parent: &Path, mod_name: &str) -> PathBuf {
        let parent_dir = parent.parent().unwrap_or(Path::new("."));

        let direct = parent_dir.join(format!("{}.rs", mod_name));
        if direct.exists() {
            return direct;
        }

        let nested = parent_dir.join(mod_name).join("mod.rs");
        if nested.exists() {
            return nested;
        }

        direct
    }

    /// Build map from type names to impl blocks
    fn build_impl_map(&mut self) {
        self.impl_map.clear();

        for file in &self.files {
            for item in &file.items {
                if let ItemKind::Impl { self_type, .. } = &item.kind {
                    let type_name = self.normalize_type_name(self_type);
                    self.impl_map
                        .entry(type_name)
                        .or_default()
                        .push(item.clone());
                }
            }
        }
    }

    /// Normalize a type name for lookup
    fn normalize_type_name(&self, ty: &str) -> String {
        let mut name = ty.to_string();
        name = name.trim_start_matches('&').to_string();
        name = name.trim_start_matches("mut ").to_string();

        if let Some(idx) = name.find('<') {
            name = name[..idx].to_string();
        }

        name.trim().to_string()
    }

    /// Build call graph with cross-module tracking
    fn build_call_graph(&mut self) -> Result<(), GravityError> {
        self.call_graph = CallGraph::default();

        let call_pattern = regex::Regex::new(r"(\w+)\s*\(").expect("Invalid regex");
        let method_pattern = regex::Regex::new(r"\.(\w+)\s*\(").expect("Invalid regex");

        for file in &self.files {
            let content = std::fs::read_to_string(&file.path).unwrap_or_default();
            let mut current_fn: Option<String> = None;

            for (line_num, line) in content.lines().enumerate() {
                if line.contains("fn ") {
                    if let Some(name) = self.extract_fn_name(line) {
                        current_fn = Some(name);
                    }
                }

                if let Some(caller) = &current_fn {
                    for cap in call_pattern.captures_iter(line) {
                        if let Some(callee) = cap.get(1) {
                            let callee_name = callee.as_str().to_string();

                            if !self.is_keyword(&callee_name)
                                && !self.is_prelude_method(&callee_name)
                            {
                                let call_site = CallSite {
                                    caller: caller.clone(),
                                    file: file.path.clone(),
                                    line: line_num + 1,
                                };

                                self.call_graph
                                    .callers
                                    .entry(callee_name.clone())
                                    .or_default()
                                    .push(call_site);

                                self.call_graph
                                    .callees
                                    .entry(caller.clone())
                                    .or_default()
                                    .push(callee_name);
                            }
                        }
                    }

                    for cap in method_pattern.captures_iter(line) {
                        if let Some(method) = cap.get(1) {
                            let method_name = method.as_str().to_string();
                            if !self.is_keyword(&method_name)
                                && !self.is_prelude_method(&method_name)
                            {
                                self.call_graph
                                    .callers
                                    .entry(method_name.clone())
                                    .or_default()
                                    .push(CallSite {
                                        caller: caller.clone(),
                                        file: file.path.clone(),
                                        line: line_num + 1,
                                    });
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Build the external reference map
    fn build_reference_map(&mut self) -> Result<(), GravityError> {
        self.reference_map = ReferenceMap::default();

        // Pattern to match qualified paths like tokio::spawn, std::fs::read
        let qualified_pattern =
            regex::Regex::new(r"(\w+(?:::\w+)+)\s*[(\[{<]?").expect("Invalid regex");

        for file in &self.files {
            let content = std::fs::read_to_string(&file.path).unwrap_or_default();
            let mut current_fn = String::from("<module>");
            let mut brace_depth = 0;

            for (line_num, line) in content.lines().enumerate() {
                // Track function context
                if line.contains("fn ") {
                    if let Some(name) = self.extract_fn_name(line) {
                        current_fn = name;
                        brace_depth = 0;
                    }
                }

                // Track brace depth for complexity estimation
                brace_depth += line.matches('{').count();
                brace_depth = brace_depth.saturating_sub(line.matches('}').count());

                // Find qualified paths
                for cap in qualified_pattern.captures_iter(line) {
                    if let Some(path_match) = cap.get(1) {
                        let path = path_match.as_str();

                        // Skip local crate paths
                        if path.starts_with("crate::") || path.starts_with("self::") {
                            continue;
                        }

                        // Check if first segment is an external crate
                        let first_segment = path.split("::").next().unwrap_or("");
                        if self.is_likely_external_crate(first_segment) {
                            let reference = ExternalReference {
                                external_path: path.to_string(),
                                file: file.path.clone(),
                                line: line_num + 1,
                                caller_context: current_fn.clone(),
                                complexity: brace_depth + self.estimate_line_complexity(line),
                            };

                            self.reference_map
                                .references
                                .entry(path.to_string())
                                .or_default()
                                .push(reference);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if a name is likely an external crate
    fn is_likely_external_crate(&self, name: &str) -> bool {
        // Common external crates and standard library modules
        let external_indicators = [
            "std",
            "core",
            "alloc",
            "tokio",
            "async_std",
            "serde",
            "regex",
            "syn",
            "quote",
            "proc_macro",
            "proc_macro2",
            "thiserror",
            "anyhow",
            "log",
            "tracing",
            "futures",
            "hyper",
            "reqwest",
            "actix",
            "axum",
            "rocket",
            "diesel",
            "sqlx",
            "chrono",
            "rand",
            "clap",
            "structopt",
            "env_logger",
            "parking_lot",
            "crossbeam",
            "rayon",
            "itertools",
            "bytes",
            "http",
            "url",
            "walkdir",
            "cargo_metadata",
            "indexmap",
            "hashbrown",
        ];

        external_indicators.contains(&name)
            || (name.chars().next().is_some_and(|c| c.is_lowercase())
                && !self.is_keyword(name)
                && name.len() > 2)
    }

    /// Estimate complexity of a line
    fn estimate_line_complexity(&self, line: &str) -> usize {
        let mut complexity = 0;

        // Generic parameters add complexity
        complexity += line.matches('<').count();
        complexity += line.matches("where").count() * 2;
        complexity += line.matches("impl").count();
        complexity += line.matches("dyn").count();
        complexity += line.matches("async").count();
        complexity += line.matches("await").count();
        complexity += line.matches("unsafe").count() * 2;

        complexity
    }

    /// Extract function name from a line containing fn
    fn extract_fn_name(&self, line: &str) -> Option<String> {
        let fn_pattern = regex::Regex::new(r"fn\s+(\w+)").ok()?;
        fn_pattern
            .captures(line)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_string())
    }

    /// Check if a string is a Rust keyword
    fn is_keyword(&self, s: &str) -> bool {
        matches!(
            s,
            "if" | "else"
                | "match"
                | "for"
                | "while"
                | "loop"
                | "return"
                | "break"
                | "continue"
                | "let"
                | "mut"
                | "ref"
                | "fn"
                | "struct"
                | "enum"
                | "impl"
                | "trait"
                | "type"
                | "where"
                | "use"
                | "mod"
                | "pub"
                | "const"
                | "static"
                | "unsafe"
                | "async"
                | "await"
                | "move"
                | "dyn"
                | "Some"
                | "None"
                | "Ok"
                | "Err"
                | "Self"
                | "self"
                | "super"
                | "crate"
                | "as"
                | "in"
                | "true"
                | "false"
        )
    }

    /// Check if a method is a prelude/std method (noise filter)
    fn is_prelude_method(&self, s: &str) -> bool {
        PRELUDE_METHODS.contains(&s)
    }

    /// Compute distances from entry point
    fn compute_distances(&mut self, root: &Path) {
        self.distance_cache.clear();

        let entry = if root.join("src/lib.rs").exists() {
            root.join("src/lib.rs")
        } else {
            root.join("src/main.rs")
        };

        let mut visited: HashSet<PathBuf> = HashSet::new();
        let mut queue: Vec<(PathBuf, usize)> = vec![(entry.clone(), 0)];

        while let Some((path, dist)) = queue.pop() {
            if visited.contains(&path) {
                continue;
            }
            visited.insert(path.clone());
            self.distance_cache.insert(path.clone(), dist);

            if let Some(file) = self.files.iter().find(|f| f.path == path) {
                for item in &file.items {
                    if let ItemKind::Mod { .. } = &item.kind {
                        let mod_path = self.resolve_mod_path(&path, &item.name);
                        if !visited.contains(&mod_path) {
                            queue.push((mod_path, dist + 1));
                        }
                    }
                }
            }
        }

        let max_dist = self.distance_cache.values().max().copied().unwrap_or(0) + 1;
        for file in &self.files {
            self.distance_cache
                .entry(file.path.clone())
                .or_insert(max_dist);
        }
    }

    /// Count how many unique modules call this item
    fn count_cross_module_callers(&self, item_name: &str) -> usize {
        let call_sites = match self.call_graph.callers.get(item_name) {
            Some(sites) => sites,
            None => return 0,
        };

        let unique_modules: HashSet<&String> = call_sites
            .iter()
            .filter_map(|site| self.file_to_module.get(&site.file))
            .collect();

        unique_modules.len()
    }

    /// Estimate generic depth from item signature
    fn estimate_generic_depth(&self, item: &ParsedItem) -> usize {
        let text = match &item.kind {
            ItemKind::Function {
                return_type,
                parameters,
                ..
            } => {
                let mut text = parameters
                    .iter()
                    .map(|p| p.ty.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                if let Some(ret) = return_type {
                    text.push_str(ret);
                }
                text
            }
            ItemKind::Struct { fields, .. } => fields
                .iter()
                .map(|f| f.ty.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            ItemKind::Impl { self_type, .. } => self_type.clone(),
            _ => String::new(),
        };

        // Count nested generic depth
        let mut max_depth: usize = 0;
        let mut current_depth: usize = 0;
        for c in text.chars() {
            if c == '<' {
                current_depth += 1;
                max_depth = max_depth.max(current_depth);
            } else if c == '>' {
                current_depth = current_depth.saturating_sub(1);
            }
        }

        max_depth
    }

    /// Check if an item is a test function
    fn is_test_item(&self, item: &ParsedItem) -> bool {
        // Check for #[test] attribute
        item.attributes.iter().any(|attr| attr.contains("test"))
            // Check if in a tests module
            || item.file_path.to_string_lossy().contains("/tests/")
            || item.name.starts_with("test_")
    }

    /// Score a single item with the new weighting system
    pub fn score_item(&self, item: &ParsedItem) -> WorkSiteScore {
        let entry_distance = self
            .distance_cache
            .get(&item.file_path)
            .copied()
            .unwrap_or(usize::MAX);

        let call_count = self
            .call_graph
            .callers
            .get(&item.name)
            .map(|v| v.len())
            .unwrap_or(0);

        let cross_module_count = self.count_cross_module_callers(&item.name);
        let generic_depth = self.estimate_generic_depth(item);
        let is_test = self.is_test_item(item);

        // "Site" = called in 1-3 places, "Utility" = called in many places
        let is_site = call_count > 0 && call_count <= 3;

        let (impl_count, trait_impls) = self.get_impl_info(&item.name);

        // Base score
        let mut score = 100.0;

        // Apply weights from the factor table
        score += (cross_module_count as f64) * weights::CROSS_MODULE_USAGE;

        if matches!(item.visibility, Visibility::Public) {
            score += weights::PUB_VISIBILITY;
        }

        score += (generic_depth as f64) * weights::GENERIC_DEPTH;

        if is_test {
            score += weights::IS_TEST_PENALTY;
        }

        // Legacy factors (kept for continuity)
        score += (entry_distance as f64) * weights::ENTRY_DISTANCE_PENALTY;

        if is_site {
            score += weights::SITE_BONUS;
        } else if call_count > 10 {
            score += weights::UTILITY_PENALTY;
        }

        score += (impl_count as f64) * weights::IMPL_RICHNESS;
        score += (trait_impls.len() as f64) * weights::TRAIT_IMPL;

        let factors = ScoreFactors {
            entry_distance,
            call_count,
            is_site,
            impl_count,
            trait_impls,
            cross_module_count,
            generic_depth,
            is_test,
        };

        // Build the context envelope
        let context = self.build_context_envelope(item);

        WorkSiteScore {
            item: item.clone(),
            score: score.max(0.0),
            factors,
            context,
        }
    }

    /// Build a ContextEnvelope for an item (Recursive Context Window)
    fn build_context_envelope(&self, item: &ParsedItem) -> ContextEnvelope {
        let breadcrumbs = self.get_breadcrumbs(item);
        let siblings = self.get_siblings(item);
        let generic_bounds = self.extract_generic_bounds(item);
        let parent_context = self.get_parent_context(item);

        ContextEnvelope {
            breadcrumbs,
            siblings,
            generic_bounds,
            parent_context,
        }
    }

    /// Get the full module path breadcrumb for an item
    fn get_breadcrumbs(&self, item: &ParsedItem) -> String {
        let module = self
            .file_to_module
            .get(&item.file_path)
            .cloned()
            .unwrap_or_else(|| "crate".to_string());

        format!("{}::{}", module, item.name)
    }

    /// Find sibling items in the same file that share generics or are related
    fn get_siblings(&self, item: &ParsedItem) -> Vec<SiblingInfo> {
        let item_generics = self.extract_generic_params(item);

        let Some(file) = self.files.iter().find(|f| f.path == item.file_path) else {
            return Vec::new();
        };

        file.items
            .iter()
            .filter(|sibling| {
                sibling.name != item.name
                    && matches!(
                        sibling.kind,
                        ItemKind::Struct { .. }
                            | ItemKind::Enum { .. }
                            | ItemKind::Function { .. }
                            | ItemKind::Trait { .. }
                    )
            })
            .take(5) // Limit siblings to avoid noise
            .map(|sibling| {
                let sibling_generics = self.extract_generic_params(sibling);
                let shared: Vec<String> = item_generics
                    .iter()
                    .filter(|g| sibling_generics.contains(g))
                    .cloned()
                    .collect();

                SiblingInfo {
                    name: sibling.name.clone(),
                    kind: self.item_kind_name(&sibling.kind),
                    line: sibling.span.start_line,
                    shared_generics: shared,
                }
            })
            .collect()
    }

    /// Get the kind name as a string
    fn item_kind_name(&self, kind: &ItemKind) -> String {
        match kind {
            ItemKind::Function { .. } => "fn",
            ItemKind::Struct { .. } => "struct",
            ItemKind::Enum { .. } => "enum",
            ItemKind::Trait { .. } => "trait",
            ItemKind::Impl { .. } => "impl",
            ItemKind::Mod { .. } => "mod",
            ItemKind::Const { .. } => "const",
            ItemKind::Static { .. } => "static",
            ItemKind::TypeAlias { .. } => "type",
            ItemKind::Macro { .. } => "macro",
            ItemKind::Use { .. } => "use",
            ItemKind::Unknown { .. } => "unknown",
        }
        .to_string()
    }

    /// Extract generic parameter names from an item
    fn extract_generic_params(&self, item: &ParsedItem) -> Vec<String> {
        let text = self.get_item_signature_text(item);
        self.parse_generic_params(&text)
    }

    /// Get the signature text for generic extraction
    fn get_item_signature_text(&self, item: &ParsedItem) -> String {
        match &item.kind {
            ItemKind::Function {
                parameters,
                return_type,
                ..
            } => {
                let params: String = parameters.iter().map(|p| p.ty.as_str()).collect();
                let ret = return_type.as_deref().unwrap_or("");
                format!("{} {}", params, ret)
            }
            ItemKind::Struct { fields, .. } => fields.iter().map(|f| f.ty.as_str()).collect(),
            ItemKind::Impl { self_type, .. } => self_type.clone(),
            ItemKind::Trait { supertraits, .. } => supertraits.join(" + "),
            _ => String::new(),
        }
    }

    /// Parse generic parameter names from text (e.g., "T", "K", "V" from "<T, K: Hash, V>")
    fn parse_generic_params(&self, text: &str) -> Vec<String> {
        let mut params = Vec::new();
        let mut depth: usize = 0;
        let mut current = String::new();

        for c in text.chars() {
            match c {
                '<' => {
                    depth += 1;
                    if depth == 1 {
                        current.clear();
                    }
                }
                '>' => {
                    if depth == 1 && !current.trim().is_empty() {
                        // Extract param name (before any colon)
                        let param = current.split(':').next().unwrap_or("").trim();
                        if !param.is_empty()
                            && param.chars().next().is_some_and(|c| c.is_uppercase())
                        {
                            params.push(param.to_string());
                        }
                    }
                    depth = depth.saturating_sub(1);
                }
                ',' if depth == 1 => {
                    let param = current.split(':').next().unwrap_or("").trim();
                    if !param.is_empty() && param.chars().next().is_some_and(|c| c.is_uppercase()) {
                        params.push(param.to_string());
                    }
                    current.clear();
                }
                _ if depth >= 1 => {
                    current.push(c);
                }
                _ => {}
            }
        }

        params
    }

    /// Extract full generic bounds from item (the "Live Signature")
    fn extract_generic_bounds(&self, item: &ParsedItem) -> Vec<GenericBound> {
        // Read the source file to get the actual signature
        let content = std::fs::read_to_string(&item.file_path).unwrap_or_default();
        let lines: Vec<&str> = content.lines().collect();

        // Get lines around the item definition
        let start = item.span.start_line.saturating_sub(1);
        let end = (start + 5).min(lines.len()); // Look at first few lines of definition

        let signature: String = lines[start..end].join(" ");

        self.parse_generic_bounds(&signature)
    }

    /// Parse generic bounds from a signature string
    fn parse_generic_bounds(&self, signature: &str) -> Vec<GenericBound> {
        let mut bounds = Vec::new();

        // Find the generic parameter section <...>
        let Some(start) = signature.find('<') else {
            return bounds;
        };

        let mut depth = 0;
        let mut generic_section = String::new();

        for c in signature[start..].chars() {
            match c {
                '<' => {
                    depth += 1;
                    if depth > 1 {
                        generic_section.push(c);
                    }
                }
                '>' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    generic_section.push(c);
                }
                _ => {
                    if depth >= 1 {
                        generic_section.push(c);
                    }
                }
            }
        }

        // Also check for where clauses
        let where_clause = if let Some(where_pos) = signature.find("where") {
            let end_pos = signature[where_pos..]
                .find('{')
                .unwrap_or(signature.len() - where_pos);
            &signature[where_pos..where_pos + end_pos]
        } else {
            ""
        };

        // Parse each generic parameter
        for part in generic_section.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            // Split on first colon for param: bounds
            let mut parts = part.splitn(2, ':');
            let param = parts.next().unwrap_or("").trim().to_string();

            if param.is_empty() || !param.chars().next().is_some_and(|c| c.is_uppercase()) {
                continue;
            }

            let mut param_bounds: Vec<String> = Vec::new();

            // Bounds from <T: Bound1 + Bound2>
            if let Some(bound_str) = parts.next() {
                for b in bound_str.split('+') {
                    let b = b.trim();
                    if !b.is_empty() {
                        param_bounds.push(b.to_string());
                    }
                }
            }

            // Additional bounds from where clause
            let where_pattern = format!("{}: ", param);
            if let Some(pos) = where_clause.find(&where_pattern) {
                let rest = &where_clause[pos + where_pattern.len()..];
                let end = rest.find(',').unwrap_or(rest.len());
                for b in rest[..end].split('+') {
                    let b = b.trim();
                    if !b.is_empty() && !param_bounds.contains(&b.to_string()) {
                        param_bounds.push(b.to_string());
                    }
                }
            }

            if !param_bounds.is_empty() {
                bounds.push(GenericBound {
                    param,
                    bounds: param_bounds,
                });
            } else {
                // Include params without bounds too (for completeness)
                bounds.push(GenericBound {
                    param,
                    bounds: vec![],
                });
            }
        }

        bounds
    }

    /// Get parent context (e.g., impl block for a method)
    fn get_parent_context(&self, item: &ParsedItem) -> Option<String> {
        // For now, if it's a method we'd look for the containing impl
        // This requires positional analysis - check if there's an impl that contains this span
        let Some(file) = self.files.iter().find(|f| f.path == item.file_path) else {
            return None;
        };

        for other in &file.items {
            if let ItemKind::Impl {
                self_type,
                trait_name,
                methods,
            } = &other.kind
            {
                if methods.contains(&item.name) {
                    return Some(if let Some(trait_n) = trait_name {
                        format!("impl {} for {}", trait_n, self_type)
                    } else {
                        format!("impl {}", self_type)
                    });
                }
            }
        }

        None
    }

    /// Get impl information for a type
    fn get_impl_info(&self, type_name: &str) -> (usize, Vec<String>) {
        match self.impl_map.get(type_name) {
            Some(impl_items) => {
                let impl_count = impl_items.len();
                let trait_impls: Vec<String> = impl_items
                    .iter()
                    .filter_map(|item| {
                        if let ItemKind::Impl { trait_name, .. } = &item.kind {
                            trait_name.clone()
                        } else {
                            None
                        }
                    })
                    .collect();
                (impl_count, trait_impls)
            }
            None => (0, Vec::new()),
        }
    }

    /// Search for items and return ranked results
    pub fn search(&self, query: &str) -> Vec<WorkSiteScore> {
        let query_lower = query.to_lowercase();

        let mut results: Vec<WorkSiteScore> = self
            .files
            .iter()
            .flat_map(|f| &f.items)
            .filter(|item| {
                item.name.to_lowercase().contains(&query_lower)
                    || item
                        .doc_comment
                        .as_ref()
                        .is_some_and(|d| d.to_lowercase().contains(&query_lower))
            })
            .map(|item| self.score_item(item))
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Get local usages of an external symbol
    pub fn get_external_usages(&self, external_path: &str) -> Vec<&ExternalReference> {
        self.reference_map
            .references
            .get(external_path)
            .map(|refs| refs.iter().collect())
            .unwrap_or_default()
    }

    /// Get the most complex usage of an external symbol
    pub fn get_most_complex_usage(&self, external_path: &str) -> Option<&ExternalReference> {
        self.reference_map
            .references
            .get(external_path)?
            .iter()
            .max_by_key(|r| r.complexity)
    }

    /// Get all external symbols used in the project
    pub fn get_all_external_symbols(&self) -> Vec<(&String, usize)> {
        let mut symbols: Vec<_> = self
            .reference_map
            .references
            .iter()
            .map(|(path, refs)| (path, refs.len()))
            .collect();
        symbols.sort_by(|a, b| b.1.cmp(&a.1));
        symbols
    }

    /// Get all impl blocks for a struct/enum
    pub fn get_impls_for_type(&self, type_name: &str) -> Vec<&ParsedItem> {
        self.impl_map
            .get(type_name)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Find call sites for a function
    pub fn find_call_sites(&self, fn_name: &str) -> Vec<&CallSite> {
        self.call_graph
            .callers
            .get(fn_name)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Find what a function calls
    pub fn find_callees(&self, fn_name: &str) -> Vec<&String> {
        self.call_graph
            .callees
            .get(fn_name)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get distance from entry point
    pub fn get_entry_distance(&self, path: &Path) -> Option<usize> {
        self.distance_cache.get(path).copied()
    }

    /// Get the module tree
    pub fn get_module_tree(&self) -> &ModuleTree {
        &self.module_tree
    }

    /// Get all parsed files
    pub fn get_files(&self) -> &[ParsedFile] {
        &self.files
    }

    /// Get call graph
    pub fn get_call_graph(&self) -> &CallGraph {
        &self.call_graph
    }

    /// Get reference map
    pub fn get_reference_map(&self) -> &ReferenceMap {
        &self.reference_map
    }

    /// Get top N most important items (highest work-site scores)
    /// Automatically excludes test functions unless explicitly searching
    pub fn get_hotspots(&self, n: usize) -> Vec<WorkSiteScore> {
        let mut all_scores: Vec<WorkSiteScore> = self
            .files
            .iter()
            .flat_map(|f| &f.items)
            .filter(|item| {
                matches!(
                    item.kind,
                    ItemKind::Function { .. }
                        | ItemKind::Struct { .. }
                        | ItemKind::Enum { .. }
                        | ItemKind::Trait { .. }
                ) && !self.is_test_item(item)
            })
            .map(|item| self.score_item(item))
            .collect();

        all_scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_scores.truncate(n);
        all_scores
    }

    /// Get significant hub functions (filtered, cross-module usage prioritized)
    pub fn get_significant_hubs(&self, n: usize) -> Vec<(String, usize, usize)> {
        let mut hubs: Vec<_> = self
            .call_graph
            .callers
            .iter()
            .filter(|(name, _)| !self.is_prelude_method(name))
            .map(|(name, sites)| {
                let cross_module = self.count_cross_module_callers(name);
                (name.clone(), sites.len(), cross_module)
            })
            .filter(|(_, _, cross_module)| *cross_module > 0)
            .collect();

        // Sort by cross-module count first, then total count
        hubs.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| b.1.cmp(&a.1)));
        hubs.truncate(n);
        hubs
    }

    /// Generate a summary of the project architecture
    pub fn summarize(&self) -> ProjectSummary {
        let mut summary = ProjectSummary::default();

        for file in &self.files {
            summary.total_files += 1;
            summary.total_parse_errors += file.parse_errors.len();

            for item in &file.items {
                match &item.kind {
                    ItemKind::Function { .. } => summary.total_functions += 1,
                    ItemKind::Struct { .. } => summary.total_structs += 1,
                    ItemKind::Enum { .. } => summary.total_enums += 1,
                    ItemKind::Trait { .. } => summary.total_traits += 1,
                    ItemKind::Impl { .. } => summary.total_impls += 1,
                    ItemKind::Mod { .. } => summary.total_modules += 1,
                    _ => {}
                }
            }
        }

        summary.hotspots = self.get_hotspots(10);
        summary.hub_functions = self.get_significant_hubs(10);
        summary.external_usage_count = self.reference_map.references.len();

        summary
    }
}

impl Default for SemanticGravity {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of project architecture
#[derive(Debug, Default)]
pub struct ProjectSummary {
    pub total_files: usize,
    pub total_functions: usize,
    pub total_structs: usize,
    pub total_enums: usize,
    pub total_traits: usize,
    pub total_impls: usize,
    pub total_modules: usize,
    pub total_parse_errors: usize,
    pub hotspots: Vec<WorkSiteScore>,
    pub hub_functions: Vec<(String, usize, usize)>,
    pub external_usage_count: usize,
}

impl std::fmt::Display for ProjectSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Project Summary ===")?;
        writeln!(f, "Files: {}", self.total_files)?;
        writeln!(f, "Functions: {}", self.total_functions)?;
        writeln!(f, "Structs: {}", self.total_structs)?;
        writeln!(f, "Enums: {}", self.total_enums)?;
        writeln!(f, "Traits: {}", self.total_traits)?;
        writeln!(f, "Impl blocks: {}", self.total_impls)?;
        writeln!(f, "Modules: {}", self.total_modules)?;
        writeln!(f, "Parse errors: {}", self.total_parse_errors)?;
        writeln!(f, "External symbols tracked: {}", self.external_usage_count)?;

        if !self.hotspots.is_empty() {
            writeln!(f, "\n=== Top Work Sites (non-test) ===")?;
            for (i, hs) in self.hotspots.iter().take(5).enumerate() {
                writeln!(
                    f,
                    "{}. {} (score: {:.1}, x-mod: {}, generics: {})",
                    i + 1,
                    hs.item.name,
                    hs.score,
                    hs.factors.cross_module_count,
                    hs.factors.generic_depth
                )?;
            }
        }

        if !self.hub_functions.is_empty() {
            writeln!(f, "\n=== Significant Hubs (cross-module) ===")?;
            for (name, total, cross_mod) in self.hub_functions.iter().take(5) {
                writeln!(f, "  {} ({} calls, {} modules)", name, total, cross_mod)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_type() {
        let gravity = SemanticGravity::new();
        assert_eq!(gravity.normalize_type_name("MyStruct"), "MyStruct");
        assert_eq!(gravity.normalize_type_name("&MyStruct"), "MyStruct");
        assert_eq!(gravity.normalize_type_name("&mut MyStruct"), "MyStruct");
        assert_eq!(gravity.normalize_type_name("Vec<T>"), "Vec");
    }

    #[test]
    fn test_prelude_filter() {
        let gravity = SemanticGravity::new();
        assert!(gravity.is_prelude_method("clone"));
        assert!(gravity.is_prelude_method("iter"));
        assert!(gravity.is_prelude_method("map"));
        assert!(!gravity.is_prelude_method("my_custom_function"));
    }
}
