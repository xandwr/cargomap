//! Rustin - Rust Architecture Analysis Tool
//!
//! A CLI tool for analyzing Rust project architecture with:
//! - Resilient partial parsing (handles broken code)
//! - Dependency bridge to cargo registry sources
//! - Semantic gravity ranking for intelligent search
//! - Call-site teleportation (local usage of external symbols)

use clap::{Parser, Subcommand};
use rustin::{DependencyBridge, SemanticGravity};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rustin")]
#[command(author, version, about = "Rust Architecture Analysis Tool", long_about = None)]
struct Cli {
    /// Path to the Rust project to analyze
    #[arg(short, long, default_value = ".")]
    path: PathBuf,

    /// Suppress non-essential output
    #[arg(short, long)]
    quiet: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze project and show summary
    Analyze {
        /// Show external symbol usage
        #[arg(short, long)]
        externals: bool,

        /// Maximum number of items to display
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Search for items by name
    Search {
        /// Search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Resolve an external crate path and show local usages
    Resolve {
        /// External path to resolve (e.g., tokio::spawn)
        path: String,

        /// Maximum number of usages to show
        #[arg(short, long, default_value = "5")]
        limit: usize,
    },

    /// List all dependencies
    Deps {
        /// Maximum number of dependencies to show
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    let project_root = if cli.path.is_absolute() {
        cli.path.clone()
    } else {
        std::env::current_dir()
            .expect("Failed to get current directory")
            .join(&cli.path)
    };

    // Initialize components
    let mut gravity = SemanticGravity::new();
    let mut dep_bridge = match DependencyBridge::new(&project_root) {
        Ok(bridge) => Some(bridge),
        Err(e) => {
            if !cli.quiet {
                eprintln!("Warning: Could not initialize dependency bridge: {}", e);
            }
            None
        }
    };

    // Analyze the project
    if !cli.quiet {
        println!("Analyzing: {}", project_root.display());
    }

    if let Err(e) = gravity.analyze_project(&project_root) {
        eprintln!("Error analyzing project: {}", e);
        std::process::exit(1);
    }

    match cli.command {
        Some(Commands::Analyze { externals, limit }) => {
            cmd_analyze(&gravity, &mut dep_bridge, externals, limit, cli.quiet);
        }
        Some(Commands::Search { query, limit }) => {
            cmd_search(&gravity, &query, limit);
        }
        Some(Commands::Resolve { path, limit }) => {
            cmd_resolve(&gravity, &mut dep_bridge, &path, limit);
        }
        Some(Commands::Deps { limit }) => {
            cmd_deps(&mut dep_bridge, limit);
        }
        None => {
            // Default behavior: show summary
            cmd_analyze(&gravity, &mut dep_bridge, false, 10, cli.quiet);
        }
    }
}

fn cmd_analyze(
    gravity: &SemanticGravity,
    dep_bridge: &mut Option<DependencyBridge>,
    show_externals: bool,
    limit: usize,
    quiet: bool,
) {
    let files = gravity.get_files();
    let total_errors: usize = files.iter().map(|f| f.parse_errors.len()).sum();

    if !quiet {
        println!(
            "Parsed {} files ({} with partial recovery)",
            files.len(),
            total_errors
        );
    }

    // Load dependencies
    if let Some(bridge) = dep_bridge {
        if let Ok(deps) = bridge.load_dependencies() {
            if !quiet {
                println!("Found {} external dependencies", deps.len());
            }
        }
    }

    // Generate summary
    let summary = gravity.summarize();
    println!("{}", summary);

    // Show top external symbols used
    if show_externals {
        let external_symbols = gravity.get_all_external_symbols();
        if !external_symbols.is_empty() {
            println!("\n=== External Symbol Usage ===");
            for (path, count) in external_symbols.iter().take(limit) {
                println!("  {} ({} usages)", path, count);
            }
            if external_symbols.len() > limit {
                println!("  ... and {} more", external_symbols.len() - limit);
            }
        }
    }
}

fn cmd_search(gravity: &SemanticGravity, query: &str, limit: usize) {
    println!("=== Search Results for '{}' ===\n", query);

    let results = gravity.search(query);
    if results.is_empty() {
        println!("No results found.");
        return;
    }

    for (i, result) in results.iter().take(limit).enumerate() {
        let test_marker = if result.factors.is_test {
            " [TEST]"
        } else {
            ""
        };
        println!(
            "{}. {}{} (score: {:.1})",
            i + 1,
            result.item.name,
            test_marker,
            result.score
        );

        // Breadcrumbs (module path)
        println!("   Path: {}", result.context.breadcrumbs);

        println!(
            "   File: {}:{}",
            result.item.file_path.display(),
            result.item.span.start_line
        );

        // Parent context if available
        if let Some(parent) = &result.context.parent_context {
            println!("   In: {}", parent);
        }

        // Generic bounds (the "Live Signature")
        if !result.context.generic_bounds.is_empty() {
            let bounds_str: Vec<String> = result
                .context
                .generic_bounds
                .iter()
                .map(|gb| {
                    if gb.bounds.is_empty() {
                        gb.param.clone()
                    } else {
                        format!("{}: {}", gb.param, gb.bounds.join(" + "))
                    }
                })
                .collect();
            println!("   Generics: <{}>", bounds_str.join(", "));
        }

        println!(
            "   Factors: x-mod={}, generics={}, calls={}, site={}",
            result.factors.cross_module_count,
            result.factors.generic_depth,
            result.factors.call_count,
            result.factors.is_site
        );

        if result.factors.impl_count > 0 {
            println!(
                "   Impls: {} ({:?})",
                result.factors.impl_count, result.factors.trait_impls
            );
        }

        // Siblings with shared generics
        let siblings_with_shared: Vec<_> = result
            .context
            .siblings
            .iter()
            .filter(|s| !s.shared_generics.is_empty())
            .collect();
        if !siblings_with_shared.is_empty() {
            println!("   Related (shared generics):");
            for sib in siblings_with_shared.iter().take(3) {
                println!(
                    "     - {} {} (shares: {})",
                    sib.kind,
                    sib.name,
                    sib.shared_generics.join(", ")
                );
            }
        }

        println!();
    }

    if results.len() > limit {
        println!("... and {} more results", results.len() - limit);
    }
}

fn cmd_resolve(
    gravity: &SemanticGravity,
    dep_bridge: &mut Option<DependencyBridge>,
    path: &str,
    limit: usize,
) {
    println!("=== Call-Site Teleportation for '{}' ===", path);

    // Show local usages (the "bridge")
    let local_usages = gravity.get_external_usages(path);
    if !local_usages.is_empty() {
        println!(
            "\nLocal usages in your project ({} sites):",
            local_usages.len()
        );

        // Sort by complexity
        let mut sorted_usages: Vec<_> = local_usages.iter().collect();
        sorted_usages.sort_by(|a, b| b.complexity.cmp(&a.complexity));

        for (i, usage) in sorted_usages.iter().take(limit).enumerate() {
            let complexity_label = match usage.complexity {
                0..=2 => "simple",
                3..=5 => "moderate",
                _ => "complex",
            };
            println!(
                "  {}. {}:{} in {}() [{}]",
                i + 1,
                usage.file.display(),
                usage.line,
                usage.caller_context,
                complexity_label
            );
        }

        if let Some(most_complex) = gravity.get_most_complex_usage(path) {
            println!(
                "\n  Most complex usage: {}:{} in {}()",
                most_complex.file.display(),
                most_complex.line,
                most_complex.caller_context
            );
        }
    } else {
        println!("  No local usages found for '{}'", path);
    }

    // Show registry location
    if let Some(bridge) = dep_bridge {
        match bridge.resolve_path(path) {
            Some(resolved) => {
                println!("\nRegistry source:");
                println!("  {}", resolved);
                println!("  Path: {}", resolved.registry_path.display());
            }
            None => {
                println!("\n  Could not resolve in registry");
            }
        }
    }
}

fn cmd_deps(dep_bridge: &mut Option<DependencyBridge>, limit: usize) {
    println!("=== Dependencies ===\n");

    let Some(bridge) = dep_bridge else {
        eprintln!("Could not initialize dependency bridge");
        return;
    };

    match bridge.load_dependencies() {
        Ok(deps) => {
            println!("Found {} external dependencies:\n", deps.len());
            for (name, dep) in deps.iter().take(limit) {
                let status = if dep.registry_path.is_some() {
                    "+"
                } else {
                    "?"
                };
                println!("  {} {} v{}", status, name, dep.version);
            }
            if deps.len() > limit {
                println!("\n  ... and {} more", deps.len() - limit);
            }
            println!("\n  + = resolved in registry, ? = not found locally");
        }
        Err(e) => {
            eprintln!("Error loading dependencies: {}", e);
        }
    }
}

/// Interactive analysis session (for future REPL mode)
#[allow(dead_code)]
struct AnalysisSession {
    gravity: SemanticGravity,
    dep_bridge: Option<DependencyBridge>,
    project_root: PathBuf,
}

#[allow(dead_code)]
impl AnalysisSession {
    fn new(project_root: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let mut gravity = SemanticGravity::new();
        gravity.analyze_project(&project_root)?;

        let dep_bridge = DependencyBridge::new(&project_root).ok();

        Ok(Self {
            gravity,
            dep_bridge,
            project_root,
        })
    }

    /// Search for items by name
    fn search(&self, query: &str) -> Vec<rustin::WorkSiteScore> {
        self.gravity.search(query)
    }

    /// Get all implementations for a type
    fn get_impls(&self, type_name: &str) -> Vec<&rustin::ParsedItem> {
        self.gravity.get_impls_for_type(type_name)
    }

    /// Find where a function is called
    fn find_callers(&self, fn_name: &str) -> Vec<&rustin::CallSite> {
        self.gravity.find_call_sites(fn_name)
    }

    /// Resolve an external crate path
    fn resolve_external(&mut self, path: &str) -> Option<rustin::dependency::ResolvedPath> {
        self.dep_bridge.as_mut()?.resolve_path(path)
    }

    /// Get local usages of an external symbol
    fn get_local_usages(&self, path: &str) -> Vec<&rustin::ExternalReference> {
        self.gravity.get_external_usages(path)
    }

    /// Get the project summary
    fn summary(&self) -> rustin::gravity::ProjectSummary {
        self.gravity.summarize()
    }
}
