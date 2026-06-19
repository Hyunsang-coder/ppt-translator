//! Desktop shell for PPT 번역캣.
//!
//! Responsibilities:
//!   1. Store the user's API keys in the OS keychain (macOS Keychain / Windows
//!      Credential Manager) via the `keyring` crate.
//!   2. Spawn the bundled Python sidecar (FastAPI server) on a free port,
//!      injecting the stored keys as environment variables.
//!   3. Parse the sidecar's `SIDECAR_READY port=N` handshake from stdout and
//!      expose that port to the WebView so the frontend can reach the API.
//!
//! The sidecar is a trusted, app-bundled binary, so we spawn it directly with
//! `std::process::Command` rather than going through tauri-plugin-shell (whose
//! capability scopes can't express a runtime-resolved resource path).

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{Emitter, Manager, RunEvent, State};

const KEYRING_SERVICE: &str = "ppt-translator";

/// Providers we hold keys for. Maps to keyring accounts and sidecar env vars.
fn env_var_for(provider: &str) -> Option<&'static str> {
    match provider {
        "openai" => Some("OPENAI_API_KEY"),
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        _ => None,
    }
}

#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("unknown provider: {0}")]
    UnknownProvider(String),
    #[error("keychain error: {0}")]
    Keyring(#[from] keyring::Error),
    #[error("sidecar error: {0}")]
    Sidecar(String),
}

impl Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

/// Shared runtime state: the bound port (once known) and the child handle so we
/// can terminate the sidecar when the app exits.
#[derive(Default)]
struct SidecarState {
    port: Mutex<Option<u16>>,
    child: Mutex<Option<Child>>,
}

fn keyring_entry(provider: &str) -> Result<keyring::Entry, AppError> {
    if env_var_for(provider).is_none() {
        return Err(AppError::UnknownProvider(provider.to_string()));
    }
    Ok(keyring::Entry::new(KEYRING_SERVICE, provider)?)
}

#[tauri::command]
fn save_api_key(provider: String, key: String) -> Result<(), AppError> {
    keyring_entry(&provider)?.set_password(&key)?;
    Ok(())
}

#[tauri::command]
fn delete_api_key(provider: String) -> Result<(), AppError> {
    match keyring_entry(&provider)?.delete_credential() {
        Ok(()) => Ok(()),
        // Deleting a key that was never set is not an error for our UX.
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

#[tauri::command]
fn has_api_key(provider: String) -> Result<bool, AppError> {
    match keyring_entry(&provider)?.get_password() {
        Ok(_) => Ok(true),
        Err(keyring::Error::NoEntry) => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Returns the sidecar port once the server is up, or None if not yet ready.
#[tauri::command]
fn get_sidecar_port(state: State<'_, SidecarState>) -> Option<u16> {
    *state.port.lock().unwrap()
}

/// Read a stored key, returning None if absent. Used at spawn time.
fn read_key(provider: &str) -> Option<String> {
    keyring::Entry::new(KEYRING_SERVICE, provider)
        .ok()?
        .get_password()
        .ok()
}

/// Kill the running sidecar (if any) and start a fresh one. Used after the user
/// changes API keys so the new keys take effect without an app restart. Waits
/// for the new sidecar to report its port (up to ~20s) before returning, so the
/// caller can translate immediately afterwards.
#[tauri::command]
fn restart_sidecar(app: tauri::AppHandle) -> Result<(), AppError> {
    {
        let state = app.state::<SidecarState>();
        if let Some(mut child) = state.child.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        *state.port.lock().unwrap() = None;
    }
    spawn_sidecar(&app)?;

    // Wait for the stdout reader thread to record the new port.
    for _ in 0..200 {
        if app.state::<SidecarState>().port.lock().unwrap().is_some() {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    Err(AppError::Sidecar(
        "sidecar did not report a port after restart".into(),
    ))
}

/// Resolve the bundled sidecar executable inside the app's resource dir.
/// The onedir PyInstaller bundle is shipped under resources/sidecar/.
fn sidecar_executable(app: &tauri::AppHandle) -> Result<std::path::PathBuf, AppError> {
    let exe_name = if cfg!(windows) {
        "ppt-translator-sidecar.exe"
    } else {
        "ppt-translator-sidecar"
    };

    // Candidate resource roots. In a packaged app `resource_dir()` is correct;
    // in `tauri dev` it can fail ("unknown path"), so also try the dir next to
    // the running executable (target/debug/resources/sidecar/...).
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(dir) = app.path().resource_dir() {
        roots.push(dir);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            roots.push(parent.to_path_buf());
        }
    }

    for root in &roots {
        let path = root.join("resources").join("sidecar").join(exe_name);
        if path.exists() {
            return Ok(path);
        }
    }

    Err(AppError::Sidecar(format!(
        "sidecar binary '{exe_name}' not found under any of: {}",
        roots
            .iter()
            .map(|r| r.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

/// Spawn the Python sidecar with stored keys injected as env vars, and read its
/// `SIDECAR_READY port=N` handshake from stdout on a background thread.
fn spawn_sidecar(app: &tauri::AppHandle) -> Result<(), AppError> {
    let exe = sidecar_executable(app)?;

    let mut command = Command::new(exe);
    command
        .args(["--host", "127.0.0.1", "--port", "0"])
        .env("PYTHONUNBUFFERED", "1")
        // The sidecar only ever binds loopback, and the WebView origin varies by
        // platform/mode (tauri://localhost, https://tauri.localhost,
        // http://localhost:3000 in dev). Allow any origin in the desktop build.
        .env("CORS_ALLOW_ALL", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(k) = read_key("openai") {
        command.env("OPENAI_API_KEY", k);
    }
    if let Some(k) = read_key("anthropic") {
        command.env("ANTHROPIC_API_KEY", k);
    }

    // Suppress the console window the sidecar would otherwise flash on Windows.
    // The sidecar is built console=True (so its sys.stdout stays valid for the
    // SIDECAR_READY handshake); CREATE_NO_WINDOW (0x08000000) gives it no console
    // *window* while leaving the inherited stdout/stderr pipes intact. Closing a
    // visible console used to kill the sidecar — now there is no window to close.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }

    let mut child = command
        .spawn()
        .map_err(|e| AppError::Sidecar(format!("failed to spawn sidecar: {e}")))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| AppError::Sidecar("no stdout pipe".into()))?;
    let stderr = child.stderr.take();

    // Stash the child so we can kill it on exit.
    app.state::<SidecarState>()
        .child
        .lock()
        .unwrap()
        .replace(child);

    // Read stdout for the READY handshake.
    let app_handle = app.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            if let Some(port) = parse_ready_port(&line) {
                let state = app_handle.state::<SidecarState>();
                *state.port.lock().unwrap() = Some(port);
                let _ = app_handle.emit("sidecar-ready", port);
            }
        }
        // stdout closed => process exited.
        let _ = app_handle.emit("sidecar-terminated", ());
    });

    // Drain stderr (uvicorn logs here) so the pipe never fills and blocks.
    if let Some(stderr) = stderr {
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                eprintln!("[sidecar] {line}");
            }
        });
    }

    Ok(())
}

/// Parse `SIDECAR_READY port=12345` -> Some(12345).
fn parse_ready_port(line: &str) -> Option<u16> {
    let line = line.trim();
    let rest = line.strip_prefix("SIDECAR_READY port=")?;
    rest.trim().parse::<u16>().ok()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .manage(SidecarState::default())
        .invoke_handler(tauri::generate_handler![
            save_api_key,
            delete_api_key,
            has_api_key,
            get_sidecar_port,
            restart_sidecar,
        ])
        .setup(|app| {
            // Don't let a sidecar failure abort app startup — surface it to the
            // UI instead so the user can still reach settings / see the error.
            if let Err(e) = spawn_sidecar(&app.handle()) {
                eprintln!("[sidecar] failed to start: {e}");
                let _ = app.handle().emit("sidecar-error", e.to_string());
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            // Terminate the sidecar when the app is exiting.
            if let RunEvent::Exit = event {
                if let Some(mut child) = app_handle
                    .state::<SidecarState>()
                    .child
                    .lock()
                    .unwrap()
                    .take()
                {
                    let _ = child.kill();
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::parse_ready_port;

    #[test]
    fn parses_ready_line() {
        assert_eq!(parse_ready_port("SIDECAR_READY port=54321"), Some(54321));
        assert_eq!(parse_ready_port("  SIDECAR_READY port=8000  "), Some(8000));
        assert_eq!(parse_ready_port("INFO: something else"), None);
        assert_eq!(parse_ready_port("SIDECAR_READY port=notaport"), None);
    }
}
