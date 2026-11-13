//! Local profiles: package directories on this machine, visible without the hub.
//!
//! Every directory holding a `model.json` under a profile root (`--profiles`,
//! then `$LOOM_LAUNCH_PROFILES`) is a local profile, as is every package
//! registered with `install --from-path`. Reading one only touches its own
//! files: a local profile never triggers a hub request.

use crate::deployment::{absolute, expand_user};
use crate::install::default_models_directory;
use crate::profile::{Profile, CARD_DOCUMENT};
use crate::scenario::{is_scenario, Scenario};
use crate::store::{InstallRegistry, InstalledModel};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The path-list environment variable naming extra profile roots.
pub const PROFILES_ENVIRONMENT: &str = "LOOM_LAUNCH_PROFILES";
/// How deep below a root the search for package directories goes.
const SEARCH_DEPTH: usize = 4;

/// Profile roots: the explicit `--profiles` directories, then the entries of
/// `$LOOM_LAUNCH_PROFILES`, each made absolute.
pub fn profile_roots(explicit: &[String]) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = explicit
        .iter()
        .filter(|p| !p.is_empty())
        .map(|p| absolute(&expand_user(Path::new(p))))
        .collect();
    if let Some(list) = std::env::var_os(PROFILES_ENVIRONMENT) {
        roots.extend(
            std::env::split_paths(&list)
                .filter(|p| !p.as_os_str().is_empty())
                .map(|p| absolute(&expand_user(&p))),
        );
    }
    let mut seen = BTreeSet::new();
    roots.retain(|root| seen.insert(root.clone()));
    roots
}

/// One package directory found on this machine.
#[derive(Clone, Debug)]
pub struct LocalProfile {
    pub directory: PathBuf,
    /// The install record naming it, when it is registered.
    pub registered: Option<InstalledModel>,
    /// Under the state directory's `models/` (or registered there).
    pub installed: bool,
    /// Holds its own `deployment.json` and `devices.json`.
    pub scenario: bool,
}

impl LocalProfile {
    /// The name it is listed and served under: its install record's, else the
    /// card's model id, else the directory name.
    pub fn read(&self) -> crate::errors::Result<Profile> {
        Profile::from_local(
            &self.directory,
            self.registered.as_ref().map(|r| r.repo.as_str()),
        )
    }

    /// The scenario it carries, when it is one.
    pub fn scenario(&self) -> Option<Scenario> {
        Scenario::detect(&self.directory).ok().flatten()
    }
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| absolute(path))
}

/// Package directories under `root` (the root itself when it is one).
fn walk(root: &Path, depth: usize, seen: &mut BTreeSet<PathBuf>, found: &mut Vec<PathBuf>) {
    if !root.is_dir() || !seen.insert(canonical(root)) {
        return;
    }
    if root.join(CARD_DOCUMENT).is_file() {
        found.push(root.to_path_buf());
        return;
    }
    if depth == 0 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let mut children: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|name| !name.starts_with('.'))
        })
        .collect();
    children.sort();
    for child in children {
        walk(&child, depth - 1, seen, found);
    }
}

/// True for an install record made by `install --from-path`.
pub fn is_local_install(installed: &InstalledModel) -> bool {
    installed.source.as_deref() == Some("local") || installed.repo.starts_with("local/")
}

/// Every local profile: package directories under `roots`, then packages
/// registered with `install --from-path`, each listed once.
pub fn discover(state: &Path, roots: &[PathBuf]) -> Vec<LocalProfile> {
    let registry = InstallRegistry::new(state);
    let installs = registry.list();
    let models = canonical(&default_models_directory(state));
    let mut directories = Vec::new();
    let mut walked = BTreeSet::new();
    for root in roots {
        walk(root, SEARCH_DEPTH, &mut walked, &mut directories);
    }
    directories.extend(
        installs
            .iter()
            .filter(|installed| is_local_install(installed) && installed.card_path().is_file())
            .map(|installed| PathBuf::from(&installed.path)),
    );
    let mut listed = BTreeSet::new();
    let mut profiles = Vec::new();
    for directory in directories {
        let resolved = canonical(&directory);
        if !listed.insert(resolved.clone()) {
            continue;
        }
        let registered = installs
            .iter()
            .find(|m| canonical(Path::new(&m.path)) == resolved)
            .cloned();
        profiles.push(LocalProfile {
            installed: resolved.starts_with(&models) || registered.is_some(),
            scenario: is_scenario(&directory),
            directory: absolute(&directory),
            registered,
        });
    }
    profiles
}

/// The local profile `target` names: its install name, its card's model id,
/// or its directory name. More than one match is an error naming them all.
pub fn find(
    state: &Path,
    roots: &[PathBuf],
    target: &str,
) -> crate::errors::Result<Option<(LocalProfile, Profile)>> {
    let mut matches = Vec::new();
    for local in discover(state, roots) {
        let Ok(profile) = local.read() else {
            continue;
        };
        let directory_name = local
            .directory
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if profile.repo == target || profile.model_id == target || directory_name == target {
            matches.push((local, profile));
        }
    }
    if matches.len() > 1 {
        let paths: Vec<String> = matches
            .iter()
            .map(|(local, _)| local.directory.display().to_string())
            .collect();
        return Err(crate::errors::Error::launcher(format!(
            "{target} names {} local profiles ({}); pass the directory instead",
            matches.len(),
            paths.join(", ")
        )));
    }
    Ok(matches.pop())
}

/// Whether a local profile matches a search: an `owner/` prefix filters by
/// install name, any other text by name, model id or path.
pub fn matches_query(
    profile: &Profile,
    directory: &Path,
    query: &str,
    author: Option<&str>,
) -> bool {
    if let Some(author) = author.filter(|a| !a.is_empty()) {
        return profile
            .repo
            .to_lowercase()
            .starts_with(&format!("{}/", author.to_lowercase()));
    }
    let query = query.trim().to_lowercase();
    if query.is_empty() {
        return true;
    }
    [
        profile.repo.as_str(),
        profile.model_id.as_str(),
        profile.name.as_str(),
        &directory.to_string_lossy(),
    ]
    .iter()
    .any(|text| text.to_lowercase().contains(&query))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::tests::scratch;

    fn card(directory: &Path, id: &str) {
        std::fs::create_dir_all(directory).unwrap();
        std::fs::write(
            directory.join("model.json"),
            format!(r#"{{"model": {{"id": "{id}"}}, "presets": {{}}}}"#),
        )
        .unwrap();
    }

    #[test]
    fn package_directories_under_a_root_are_found_once() {
        let root = scratch("local-discover");
        let state = root.join("state");
        let profiles = root.join("profiles");
        card(&profiles.join("plain"), "test/plain");
        let scenario = crate::scenario::tests::hybrid_package(&profiles.join("nested"));
        card(&profiles.join(".hidden"), "test/hidden");
        // A package's own subdirectories are not searched.
        card(&profiles.join("plain").join("inner"), "test/inner");
        let roots = vec![profiles.clone(), profiles.join("plain")];
        let found = discover(&state, &roots);
        let directories: Vec<_> = found.iter().map(|l| l.directory.clone()).collect();
        assert_eq!(directories, [scenario.clone(), profiles.join("plain")]);
        assert!(found[0].scenario);
        assert!(!found[1].scenario);
        assert!(!found[0].installed);
        let (local, profile) = find(&state, &roots, "qwen3.8-27b").unwrap().unwrap();
        assert_eq!(local.directory, scenario);
        assert_eq!(profile.context_length, Some(40960));
        assert_eq!(profile.native_context_length, Some(262144));
        assert!(find(&state, &roots, "nothing").unwrap().is_none());
        assert!(matches_query(&profile, &scenario, "QWEN3.8", None));
        assert!(matches_query(&profile, &scenario, "nested", None));
        assert!(!matches_query(&profile, &scenario, "gemma", None));
        assert!(!matches_query(&profile, &scenario, "", Some("loom-fun")));
    }

    #[test]
    fn registered_local_installs_are_listed_with_their_name() {
        let root = scratch("local-registered");
        let state = root.join("state");
        let package = root.join("elsewhere").join("pkg");
        card(&package, "test/pkg");
        InstallRegistry::new(&state)
            .record(&InstalledModel {
                repo: "local/pkg".into(),
                path: package.to_string_lossy().into_owned(),
                source: Some("local".into()),
                ..Default::default()
            })
            .unwrap();
        let found = discover(&state, &[]);
        assert_eq!(found.len(), 1);
        assert!(found[0].installed);
        assert_eq!(found[0].read().unwrap().repo, "local/pkg");
        let models = default_models_directory(&state).join("copied");
        card(&models, "test/copied");
        let found = discover(&state, &[default_models_directory(&state)]);
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|l| l.installed));
    }

    #[test]
    fn roots_come_from_flags_then_the_environment() {
        let roots = profile_roots(&["/a/b".into(), "/a/b".into(), String::new()]);
        assert_eq!(roots[0], PathBuf::from("/a/b"));
        assert_eq!(
            roots
                .iter()
                .filter(|r| r.as_path() == Path::new("/a/b"))
                .count(),
            1
        );
    }
}
