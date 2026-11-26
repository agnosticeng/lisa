//! Guard the loading/execution split, including aliases and grouped imports.
use std::path::{Path, PathBuf};
use syn::{
    ItemUse, UseTree,
    visit::{self, Visit},
};

#[derive(Default)]
struct Boundary {
    forbidden: Vec<String>,
    allow_data_io: bool,
}
impl Boundary {
    fn check(&mut self, names: &[String]) {
        let names = names.iter().map(String::as_str).collect::<Vec<_>>();
        let bad = matches!(names.first(), Some(&"lisa_gguf" | &"lisa_mlx" | &"memmap2"))
            || names.starts_with(&["crate", "checkpoint"])
            || names.starts_with(&["crate", "loading"])
            || names.starts_with(&["crate", "model_binding"])
            || names.starts_with(&["crate", "image_generation"])
            || (!self.allow_data_io
                && names.starts_with(&["std", "fs"])
                && !matches!(names.get(2), Some(&"write" | &"create_dir_all"))
                && names.as_slice() != ["std", "fs", "File", "create"])
            || (names.starts_with(&["crate", "weights"])
                && !matches!(
                    names.get(2),
                    Some(
                        &"source"
                            | &"prepared"
                            | &"affine"
                            | &"ResidencyBytes"
                            | &"ResidencyPolicy"
                    )
                ));
        if bad {
            self.forbidden.push(names.join("::"));
        }
    }
    fn imports(&mut self, prefix: &mut Vec<String>, tree: &UseTree) {
        match tree {
            UseTree::Path(p) => {
                prefix.push(p.ident.to_string());
                self.imports(prefix, &p.tree);
                prefix.pop();
            }
            UseTree::Group(g) => {
                for item in &g.items {
                    self.imports(prefix, item);
                }
            }
            UseTree::Name(n) => {
                prefix.push(n.ident.to_string());
                self.check(prefix);
                prefix.pop();
            }
            UseTree::Rename(n) => {
                prefix.push(n.ident.to_string());
                self.check(prefix);
                prefix.pop();
            }
            UseTree::Glob(_) => self.check(prefix),
        }
    }
}
fn test_only(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("cfg")
            && attr
                .parse_args::<syn::Path>()
                .is_ok_and(|p| p.is_ident("test"))
    })
}
impl<'ast> Visit<'ast> for Boundary {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if !test_only(&item.attrs) {
            visit::visit_item_mod(self, item);
        }
    }
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        if !test_only(&item.attrs) {
            visit::visit_item_fn(self, item);
        }
    }
    fn visit_item_use(&mut self, item: &'ast ItemUse) {
        if !test_only(&item.attrs) {
            self.imports(&mut Vec::new(), &item.tree);
        }
    }
    fn visit_path(&mut self, path: &'ast syn::Path) {
        self.check(
            &path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>(),
        );
        visit::visit_path(self, path);
    }
}
fn sources(dir: &Path, paths: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            sources(&path, paths);
        } else if path.extension().is_some_and(|x| x == "rs")
            && !path
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .ends_with("_tests")
        {
            paths.push(path);
        }
    }
}

#[test]
fn execution_depends_on_prepared_tensors_not_checkpoint_readers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut paths = vec![];
    for dir in ["bindings", "core", "vision", "laya"] {
        sources(&root.join(dir), &mut paths);
    }
    for name in ["source", "prepared", "affine"] {
        paths.push(root.join(format!("weights/{name}.rs")));
    }
    let mut failures = vec![];
    for path in paths {
        let tree = syn::parse_file(&std::fs::read_to_string(&path).unwrap()).unwrap();
        // These read cost profiles, user images and opt-in diagnostic captures.
        // All other execution modules may write diagnostics but cannot read files.
        let allow_data_io = matches!(
            path.strip_prefix(&root).unwrap().to_str().unwrap(),
            "core/round_cost/mod.rs" | "vision/mod.rs"
        );
        let mut boundary = Boundary {
            allow_data_io,
            ..Default::default()
        };
        boundary.visit_file(&tree);
        for import in boundary.forbidden {
            failures.push(format!(
                "{} imports {import}",
                path.strip_prefix(&root).unwrap().display()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "move checkpoint access to a loading adapter:\n{}",
        failures.join("\n")
    );
}

#[test]
fn boundary_guard_detects_grouped_renamed_and_qualified_reader_access() {
    let tree = syn::parse_file(
        r#"
        use crate::{weights::{mlx::Shard as Hidden, source::TensorSource}, checkpoint::config};
        use lisa_gguf as hidden;
        fn load() { let _ = lisa_mlx::Shard::open(); }
        fn bypass() { let _ = std::fs::read("config.json"); crate::model_binding::language(); }
        #[cfg(test)] mod fixtures { use lisa_gguf::Header; }
    "#,
    )
    .unwrap();
    let mut boundary = Boundary::default();
    boundary.visit_file(&tree);
    assert_eq!(
        boundary.forbidden,
        [
            "crate::weights::mlx::Shard",
            "crate::checkpoint::config",
            "lisa_gguf",
            "lisa_mlx::Shard::open",
            "std::fs::read",
            "crate::model_binding::language"
        ]
    );
}
