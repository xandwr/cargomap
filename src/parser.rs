//! Partial Parser - LSP-Lite with resilient parsing
//!
//! The key insight: instead of failing on parse errors, we split files by
//! top-level items and parse each individually. If one function has a syntax
//! error, we can still "see" the rest of the module.

use crate::types::*;
use regex::Regex;
use std::path::Path;
use syn::visit::Visit;
use syn::{self, Attribute, File, Item, Visibility as SynVisibility};
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Error, Debug)]
pub enum ParserError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error in {file}: {message}")]
    Parse { file: String, message: String },
}

/// Partial parser that handles broken code gracefully
pub struct PartialParser {
    // Reserved for future regex-based optimizations
}

impl Default for PartialParser {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialParser {
    pub fn new() -> Self {
        Self {}
    }

    /// Parse a project directory
    pub fn parse_project(&self, root: &Path) -> Result<Vec<ParsedFile>, ParserError> {
        let mut files = Vec::new();

        for entry in WalkDir::new(root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path().extension().is_some_and(|ext| ext == "rs")
                    && !e.path().to_string_lossy().contains("/target/")
            })
        {
            let path = entry.path();
            match self.parse_file(path) {
                Ok(parsed) => files.push(parsed),
                Err(e) => {
                    eprintln!("Warning: Failed to parse {}: {}", path.display(), e);
                }
            }
        }

        Ok(files)
    }

    /// Parse a single file with fallback to partial parsing
    pub fn parse_file(&self, path: &Path) -> Result<ParsedFile, ParserError> {
        let content = std::fs::read_to_string(path)?;
        let module_path = self.derive_module_path(path);

        // First, try to parse the whole file
        match syn::parse_file(&content) {
            Ok(file) => {
                let items = self.extract_items(&file, path);
                Ok(ParsedFile {
                    path: path.to_path_buf(),
                    items,
                    parse_errors: Vec::new(),
                    module_path,
                })
            }
            Err(_) => {
                // File has errors - fall back to partial parsing
                self.parse_partial(path, &content, module_path)
            }
        }
    }

    /// Parse file partially, extracting whatever items we can
    fn parse_partial(
        &self,
        path: &Path,
        content: &str,
        module_path: Vec<String>,
    ) -> Result<ParsedFile, ParserError> {
        let mut items = Vec::new();
        let mut errors = Vec::new();

        // Split the file into chunks by top-level item boundaries
        let chunks = self.split_into_items(content);

        for chunk in chunks {
            match self.parse_chunk(&chunk.text, path) {
                Ok(mut parsed_items) => {
                    // Adjust line numbers based on chunk offset
                    for item in &mut parsed_items {
                        item.span.start_line += chunk.start_line;
                        item.span.end_line += chunk.start_line;
                    }
                    items.extend(parsed_items);
                }
                Err(e) => {
                    errors.push(ParseError {
                        message: e.to_string(),
                        span: Some(Span {
                            start_line: chunk.start_line,
                            start_col: 0,
                            end_line: chunk.start_line + chunk.text.lines().count(),
                            end_col: 0,
                        }),
                        raw_text: chunk.text.chars().take(200).collect(),
                    });

                    // Still create an Unknown item so we have some info
                    items.push(ParsedItem {
                        kind: ItemKind::Unknown {
                            raw_text: chunk.text.chars().take(500).collect(),
                            error: e.to_string(),
                        },
                        name: self.guess_item_name(&chunk.text),
                        visibility: Visibility::Private,
                        span: Span {
                            start_line: chunk.start_line,
                            start_col: 0,
                            end_line: chunk.start_line + chunk.text.lines().count(),
                            end_col: 0,
                        },
                        file_path: path.to_path_buf(),
                        attributes: Vec::new(),
                        doc_comment: None,
                    });
                }
            }
        }

        Ok(ParsedFile {
            path: path.to_path_buf(),
            items,
            parse_errors: errors,
            module_path,
        })
    }

    /// Split file content into individual item chunks
    fn split_into_items(&self, content: &str) -> Vec<ItemChunk> {
        let mut chunks = Vec::new();
        let mut current_start = 0;
        let mut brace_depth = 0;
        let mut in_string = false;
        let mut in_char = false;
        let mut escape_next = false;
        let chars: Vec<char> = content.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];

            if escape_next {
                escape_next = false;
                i += 1;
                continue;
            }

            if c == '\\' {
                escape_next = true;
                i += 1;
                continue;
            }

            if !in_char && c == '"' {
                in_string = !in_string;
            } else if !in_string && c == '\'' {
                // Simple char literal detection
                in_char = !in_char;
            }

            if !in_string && !in_char {
                match c {
                    '{' => brace_depth += 1,
                    '}' => {
                        brace_depth -= 1;
                        if brace_depth == 0 {
                            // Found end of a top-level item
                            let chunk_text: String = chars[current_start..=i].iter().collect();
                            let start_line = content[..current_start].lines().count();

                            if !chunk_text.trim().is_empty() {
                                chunks.push(ItemChunk {
                                    text: chunk_text,
                                    start_line,
                                });
                            }
                            current_start = i + 1;
                        }
                    }
                    ';' if brace_depth == 0 => {
                        // End of a semicolon-terminated item (use, const, etc.)
                        let chunk_text: String = chars[current_start..=i].iter().collect();
                        let start_line = content[..current_start].lines().count();

                        if !chunk_text.trim().is_empty() && self.looks_like_item(chunk_text.trim())
                        {
                            chunks.push(ItemChunk {
                                text: chunk_text,
                                start_line,
                            });
                        }
                        current_start = i + 1;
                    }
                    _ => {}
                }
            }

            i += 1;
        }

        // Handle any remaining content
        if current_start < chars.len() {
            let remaining: String = chars[current_start..].iter().collect();
            if !remaining.trim().is_empty() && self.looks_like_item(remaining.trim()) {
                let start_line = content[..current_start].lines().count();
                chunks.push(ItemChunk {
                    text: remaining,
                    start_line,
                });
            }
        }

        chunks
    }

    /// Check if text looks like a Rust item
    fn looks_like_item(&self, text: &str) -> bool {
        let trimmed = text.trim_start();
        let keywords = [
            "fn ",
            "pub ",
            "struct ",
            "enum ",
            "impl ",
            "mod ",
            "trait ",
            "type ",
            "const ",
            "static ",
            "use ",
            "macro_rules!",
            "#[",
            "async ",
            "unsafe ",
        ];
        keywords.iter().any(|kw| trimmed.starts_with(kw))
    }

    /// Try to guess item name from unparseable text
    fn guess_item_name(&self, text: &str) -> String {
        let name_pattern =
            Regex::new(r"(?:fn|struct|enum|impl|mod|trait|type|const|static|macro_rules!)\s+(\w+)")
                .unwrap();

        name_pattern
            .captures(text)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| "<unknown>".to_string())
    }

    /// Parse a single chunk of code
    fn parse_chunk(&self, chunk: &str, path: &Path) -> Result<Vec<ParsedItem>, ParserError> {
        // Wrap in a module context for parsing
        let wrapped = format!("mod __wrapper__ {{ {} }}", chunk);

        match syn::parse_file(&wrapped) {
            Ok(file) => {
                let items = self.extract_items(&file, path);
                Ok(items)
            }
            Err(e) => Err(ParserError::Parse {
                file: path.display().to_string(),
                message: e.to_string(),
            }),
        }
    }

    /// Extract ParsedItems from a syn::File
    fn extract_items(&self, file: &File, path: &Path) -> Vec<ParsedItem> {
        let mut visitor = ItemVisitor::new(path);
        visitor.visit_file(file);
        visitor.items
    }

    /// Derive module path from file path
    fn derive_module_path(&self, path: &Path) -> Vec<String> {
        let mut parts = Vec::new();
        let mut found_src = false;

        for component in path.components() {
            if let std::path::Component::Normal(os_str) = component {
                if let Some(s) = os_str.to_str() {
                    // Start collecting after "src" directory
                    if s == "src" {
                        found_src = true;
                        continue;
                    }

                    if found_src && s != "lib.rs" && s != "main.rs" && s != "mod.rs" {
                        let name = s.strip_suffix(".rs").unwrap_or(s);
                        parts.push(name.to_string());
                    }
                }
            }
        }

        parts
    }
}

struct ItemChunk {
    text: String,
    start_line: usize,
}

/// Visitor to extract items from syn AST
struct ItemVisitor {
    items: Vec<ParsedItem>,
    path: std::path::PathBuf,
}

impl ItemVisitor {
    fn new(path: &Path) -> Self {
        Self {
            items: Vec::new(),
            path: path.to_path_buf(),
        }
    }

    fn convert_visibility(&self, vis: &SynVisibility) -> Visibility {
        match vis {
            SynVisibility::Public(_) => Visibility::Public,
            SynVisibility::Restricted(r) => {
                let path = r.path.segments.first().map(|s| s.ident.to_string());
                match path.as_deref() {
                    Some("crate") => Visibility::Crate,
                    Some("super") => Visibility::Super,
                    _ => Visibility::Restricted,
                }
            }
            SynVisibility::Inherited => Visibility::Private,
        }
    }

    fn extract_doc_comment(&self, attrs: &[Attribute]) -> Option<String> {
        let docs: Vec<String> = attrs
            .iter()
            .filter_map(|attr| {
                if attr.path().is_ident("doc") {
                    if let syn::Meta::NameValue(nv) = &attr.meta {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(s),
                            ..
                        }) = &nv.value
                        {
                            return Some(s.value());
                        }
                    }
                }
                None
            })
            .collect();

        if docs.is_empty() {
            None
        } else {
            Some(docs.join("\n"))
        }
    }

    fn attrs_to_strings(&self, attrs: &[Attribute]) -> Vec<String> {
        attrs
            .iter()
            .filter(|a| !a.path().is_ident("doc"))
            .map(|a| {
                let path = a
                    .path()
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                format!("#[{}]", path)
            })
            .collect()
    }

    fn type_to_string(&self, ty: &syn::Type) -> String {
        quote::quote!(#ty).to_string()
    }
}

impl<'ast> Visit<'ast> for ItemVisitor {
    fn visit_item(&mut self, item: &'ast Item) {
        let parsed = match item {
            Item::Fn(f) => {
                let params: Vec<Parameter> = f
                    .sig
                    .inputs
                    .iter()
                    .map(|arg| match arg {
                        syn::FnArg::Receiver(r) => Parameter {
                            name: "self".to_string(),
                            ty: if r.reference.is_some() {
                                if r.mutability.is_some() {
                                    "&mut self"
                                } else {
                                    "&self"
                                }
                            } else {
                                "self"
                            }
                            .to_string(),
                            is_self: true,
                        },
                        syn::FnArg::Typed(t) => Parameter {
                            name: quote::quote!(#t.pat).to_string(),
                            ty: self.type_to_string(&t.ty),
                            is_self: false,
                        },
                    })
                    .collect();

                let return_type = match &f.sig.output {
                    syn::ReturnType::Default => None,
                    syn::ReturnType::Type(_, ty) => Some(self.type_to_string(ty)),
                };

                Some(ParsedItem {
                    kind: ItemKind::Function {
                        is_async: f.sig.asyncness.is_some(),
                        parameters: params,
                        return_type,
                    },
                    name: f.sig.ident.to_string(),
                    visibility: self.convert_visibility(&f.vis),
                    span: Span::default(),
                    file_path: self.path.clone(),
                    attributes: self.attrs_to_strings(&f.attrs),
                    doc_comment: self.extract_doc_comment(&f.attrs),
                })
            }

            Item::Struct(s) => {
                let (fields, is_tuple) = match &s.fields {
                    syn::Fields::Named(named) => {
                        let fields = named
                            .named
                            .iter()
                            .map(|f| StructField {
                                name: f.ident.as_ref().map(|i| i.to_string()),
                                ty: self.type_to_string(&f.ty),
                                visibility: self.convert_visibility(&f.vis),
                            })
                            .collect();
                        (fields, false)
                    }
                    syn::Fields::Unnamed(unnamed) => {
                        let fields = unnamed
                            .unnamed
                            .iter()
                            .map(|f| StructField {
                                name: None,
                                ty: self.type_to_string(&f.ty),
                                visibility: self.convert_visibility(&f.vis),
                            })
                            .collect();
                        (fields, true)
                    }
                    syn::Fields::Unit => (Vec::new(), false),
                };

                Some(ParsedItem {
                    kind: ItemKind::Struct { fields, is_tuple },
                    name: s.ident.to_string(),
                    visibility: self.convert_visibility(&s.vis),
                    span: Span::default(),
                    file_path: self.path.clone(),
                    attributes: self.attrs_to_strings(&s.attrs),
                    doc_comment: self.extract_doc_comment(&s.attrs),
                })
            }

            Item::Enum(e) => {
                let variants = e
                    .variants
                    .iter()
                    .map(|v| {
                        let fields = match &v.fields {
                            syn::Fields::Named(named) => named
                                .named
                                .iter()
                                .map(|f| StructField {
                                    name: f.ident.as_ref().map(|i| i.to_string()),
                                    ty: self.type_to_string(&f.ty),
                                    visibility: self.convert_visibility(&f.vis),
                                })
                                .collect(),
                            syn::Fields::Unnamed(unnamed) => unnamed
                                .unnamed
                                .iter()
                                .map(|f| StructField {
                                    name: None,
                                    ty: self.type_to_string(&f.ty),
                                    visibility: self.convert_visibility(&f.vis),
                                })
                                .collect(),
                            syn::Fields::Unit => Vec::new(),
                        };
                        EnumVariant {
                            name: v.ident.to_string(),
                            fields,
                        }
                    })
                    .collect();

                Some(ParsedItem {
                    kind: ItemKind::Enum { variants },
                    name: e.ident.to_string(),
                    visibility: self.convert_visibility(&e.vis),
                    span: Span::default(),
                    file_path: self.path.clone(),
                    attributes: self.attrs_to_strings(&e.attrs),
                    doc_comment: self.extract_doc_comment(&e.attrs),
                })
            }

            Item::Impl(i) => {
                let self_type = self.type_to_string(&i.self_ty);
                let trait_name = i.trait_.as_ref().map(|(_, path, _)| {
                    path.segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect::<Vec<_>>()
                        .join("::")
                });

                let methods: Vec<String> = i
                    .items
                    .iter()
                    .filter_map(|item| {
                        if let syn::ImplItem::Fn(m) = item {
                            Some(m.sig.ident.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();

                Some(ParsedItem {
                    kind: ItemKind::Impl {
                        self_type: self_type.clone(),
                        trait_name,
                        methods,
                    },
                    name: format!("impl {}", self_type),
                    visibility: Visibility::Private,
                    span: Span::default(),
                    file_path: self.path.clone(),
                    attributes: self.attrs_to_strings(&i.attrs),
                    doc_comment: None,
                })
            }

            Item::Trait(t) => {
                let methods: Vec<String> = t
                    .items
                    .iter()
                    .filter_map(|item| {
                        if let syn::TraitItem::Fn(m) = item {
                            Some(m.sig.ident.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();

                let supertraits: Vec<String> = t
                    .supertraits
                    .iter()
                    .filter_map(|bound| {
                        if let syn::TypeParamBound::Trait(tb) = bound {
                            Some(
                                tb.path
                                    .segments
                                    .iter()
                                    .map(|s| s.ident.to_string())
                                    .collect::<Vec<_>>()
                                    .join("::"),
                            )
                        } else {
                            None
                        }
                    })
                    .collect();

                Some(ParsedItem {
                    kind: ItemKind::Trait {
                        methods,
                        supertraits,
                    },
                    name: t.ident.to_string(),
                    visibility: self.convert_visibility(&t.vis),
                    span: Span::default(),
                    file_path: self.path.clone(),
                    attributes: self.attrs_to_strings(&t.attrs),
                    doc_comment: self.extract_doc_comment(&t.attrs),
                })
            }

            Item::Mod(m) => Some(ParsedItem {
                kind: ItemKind::Mod {
                    inline: m.content.is_some(),
                },
                name: m.ident.to_string(),
                visibility: self.convert_visibility(&m.vis),
                span: Span::default(),
                file_path: self.path.clone(),
                attributes: self.attrs_to_strings(&m.attrs),
                doc_comment: self.extract_doc_comment(&m.attrs),
            }),

            Item::Use(u) => {
                let path = quote::quote!(#u.tree).to_string();
                Some(ParsedItem {
                    kind: ItemKind::Use { path: path.clone() },
                    name: path,
                    visibility: self.convert_visibility(&u.vis),
                    span: Span::default(),
                    file_path: self.path.clone(),
                    attributes: self.attrs_to_strings(&u.attrs),
                    doc_comment: None,
                })
            }

            Item::Const(c) => Some(ParsedItem {
                kind: ItemKind::Const {
                    ty: self.type_to_string(&c.ty),
                },
                name: c.ident.to_string(),
                visibility: self.convert_visibility(&c.vis),
                span: Span::default(),
                file_path: self.path.clone(),
                attributes: self.attrs_to_strings(&c.attrs),
                doc_comment: self.extract_doc_comment(&c.attrs),
            }),

            Item::Static(s) => Some(ParsedItem {
                kind: ItemKind::Static {
                    ty: self.type_to_string(&s.ty),
                    is_mut: matches!(s.mutability, syn::StaticMutability::Mut(_)),
                },
                name: s.ident.to_string(),
                visibility: self.convert_visibility(&s.vis),
                span: Span::default(),
                file_path: self.path.clone(),
                attributes: self.attrs_to_strings(&s.attrs),
                doc_comment: self.extract_doc_comment(&s.attrs),
            }),

            Item::Type(t) => Some(ParsedItem {
                kind: ItemKind::TypeAlias {
                    ty: self.type_to_string(&t.ty),
                },
                name: t.ident.to_string(),
                visibility: self.convert_visibility(&t.vis),
                span: Span::default(),
                file_path: self.path.clone(),
                attributes: self.attrs_to_strings(&t.attrs),
                doc_comment: self.extract_doc_comment(&t.attrs),
            }),

            Item::Macro(m) => Some(ParsedItem {
                kind: ItemKind::Macro {
                    is_declarative: true,
                },
                name: m
                    .ident
                    .as_ref()
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| "<anonymous>".to_string()),
                visibility: Visibility::Private,
                span: Span::default(),
                file_path: self.path.clone(),
                attributes: self.attrs_to_strings(&m.attrs),
                doc_comment: None,
            }),

            _ => None,
        };

        if let Some(item) = parsed {
            self.items.push(item);
        }

        syn::visit::visit_item(self, item);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_parsing() {
        let parser = PartialParser::new();

        let broken_code = r#"
fn good_function() {
    println!("works");
}

fn broken_function() {
    let x = // syntax error here
}

struct StillWorks {
    field: i32,
}
"#;

        let chunks = parser.split_into_items(broken_code);
        assert!(chunks.len() >= 2, "Should split into multiple chunks");
    }
}
