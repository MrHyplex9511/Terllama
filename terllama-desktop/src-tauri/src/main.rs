// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod config;
mod download;
mod server;

use commands::AppState;
use server::ServerManager;
use std::sync::Arc;
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Manager,
};
use tracing_subscriber::prelude::*;

fn main() {
    // Init logging
    let log_dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".terllama/logs");
    std::fs::create_dir_all(&log_dir).ok();

    let log_file = std::fs::File::create(log_dir.join("desktop.log")).ok();
    let file_layer = log_file.map(|f| tracing_subscriber::fmt::layer().with_writer(std::sync::Mutex::new(f)));

    let subscriber = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .with(file_layer);
    tracing::subscriber::set_global_default(subscriber).ok();

    tracing::info!("Terllama Desktop v1.0.0 starting");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(AppState {
            server: Arc::new(ServerManager::new()),
        })
        .invoke_handler(tauri::generate_handler![
            commands::fetch_registry,
            commands::download_model,
            commands::list_downloaded_models,
            commands::delete_model,
            commands::start_server,
            commands::stop_server,
            commands::server_status,
            commands::send_chat_message,
            commands::stream_chat,
            commands::get_settings,
            commands::save_settings,
        ])
        .setup(|app| {
            // Tray icon and menu
            let show = MenuItem::with_id(app, "show", "Show Window", true, None::<&str>)?;
            let start_srv = MenuItem::with_id(app, "start", "Start Server", true, None::<&str>)?;
            let stop_srv = MenuItem::with_id(app, "stop", "Stop Server", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, Some("CmdOrCtrl+Q"))?;

            let menu = Menu::with_items(app, &[&show, &start_srv, &stop_srv, &quit])?;

            let _tray = TrayIconBuilder::new()
                .menu(&menu)
                .tooltip("Terllama")
                .on_menu_event(|app, event| {
                    match event.id().as_ref() {
                        "show" => {
                            if let Some(window) = app.get_webview_window("main") {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                        "start" => {
                            // Start server with active model
                            if let Some(state) = app.try_state::<AppState>() {
                                let port = config::Settings::load().port;
                                let model_id = "default".to_string();
                                let state_clone = state.server.clone();
                                tauri::async_runtime::spawn(async move {
                                    let _ = state_clone.start(model_id, port).await;
                                });
                            }
                        }
                        "stop" => {
                            if let Some(state) = app.try_state::<AppState>() {
                                let _ = state.server.stop();
                            }
                        }
                        "quit" => {
                            // Stop server first
                            if let Some(state) = app.try_state::<AppState>() {
                                let _ = state.server.stop();
                            }
                            app.exit(0);
                        }
                        _ => {}
                    }
                })
                .build(app)?;

            // Auto-start server if configured
            let settings = config::Settings::load();
            if settings.auto_start {
                if let Some(state) = app.try_state::<AppState>() {
                    let port = settings.port;
                    let model_id = "default".to_string();
                    let state_clone = state.server.clone();
                    tauri::async_runtime::spawn(async move {
                        let _ = state_clone.start(model_id, port).await;
                    });
                }
            }

            // Prevent close - hide instead
            if let Some(window) = app.get_webview_window("main") {
                let w = window.clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = w.hide();
                    }
                });
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
