import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var settings: AppSettings = .load()
    private var daemon: DaemonController!
    private var watcher: EventWatcher!
    private var dashboardWindow: DashboardWindowController?
    private var settingsWindow: NSWindow?
    private var statusMenuItem: NSMenuItem!
    private var workspaceMenuItem: NSMenuItem!

    func applicationDidFinishLaunching(_ notification: Notification) {
        Notifier.shared.requestAuthorization()

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        configureStatusButton(for: .stopped)
        statusItem.menu = makeMenu()

        daemon = DaemonController(settings: settings)
        daemon.onStateChange = { [weak self] state in
            DispatchQueue.main.async {
                self?.handleDaemonState(state)
            }
        }

        watcher = EventWatcher(settings: settings)

        daemon.start()
    }

    func applicationWillTerminate(_ notification: Notification) {
        watcher.stop()
        daemon.stop()
    }

    private func makeMenu() -> NSMenu {
        let menu = NSMenu()
        statusMenuItem = NSMenuItem(title: "Status: starting…", action: nil, keyEquivalent: "")
        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)
        menu.addItem(NSMenuItem.separator())

        let openItem = NSMenuItem(title: "Open Dashboard", action: #selector(openDashboard), keyEquivalent: "o")
        openItem.target = self
        menu.addItem(openItem)

        workspaceMenuItem = NSMenuItem(title: "Workspace: \(settings.workspace)", action: nil, keyEquivalent: "")
        workspaceMenuItem.isEnabled = false
        menu.addItem(workspaceMenuItem)

        let settingsItem = NSMenuItem(title: "Settings…", action: #selector(openSettings), keyEquivalent: ",")
        settingsItem.target = self
        menu.addItem(settingsItem)

        menu.addItem(NSMenuItem.separator())
        let quit = NSMenuItem(title: "Quit TeamAI", action: #selector(quit), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)
        return menu
    }

    private func handleDaemonState(_ state: DaemonState) {
        configureStatusButton(for: state)
        switch state {
        case .stopped:
            statusMenuItem.title = "Status: stopped"
            watcher.stop()
        case .starting:
            statusMenuItem.title = "Status: starting…"
        case .ready:
            statusMenuItem.title = "Status: ready (port \(settings.port))"
            watcher.update(settings: settings)
            watcher.start()
        case .error(let message):
            statusMenuItem.title = "Status: error — \(message)"
            watcher.stop()
        }
    }

    private func configureStatusButton(for state: DaemonState) {
        guard let button = statusItem.button else { return }
        let image = NSImage(systemSymbolName: "circle.fill", accessibilityDescription: "TeamAI status")
        let symbolConfig: NSImage.SymbolConfiguration
        switch state {
        case .stopped:
            symbolConfig = NSImage.SymbolConfiguration(pointSize: 11, weight: .regular)
                .applying(.init(paletteColors: [.systemGray]))
        case .starting:
            symbolConfig = NSImage.SymbolConfiguration(pointSize: 11, weight: .regular)
                .applying(.init(paletteColors: [.systemGray]))
        case .ready:
            symbolConfig = NSImage.SymbolConfiguration(pointSize: 11, weight: .regular)
                .applying(.init(paletteColors: [.systemGreen]))
        case .error:
            symbolConfig = NSImage.SymbolConfiguration(pointSize: 11, weight: .regular)
                .applying(.init(paletteColors: [.systemRed]))
        }
        let configuredImage = image?.withSymbolConfiguration(symbolConfig)
        button.image = configuredImage
        button.imagePosition = .imageLeft
        button.title = "TeamAI"
    }

    @objc private func openDashboard() {
        if let window = dashboardWindow {
            window.update(url: settings.dashboardURL)
            window.showAndFocus()
            return
        }
        let controller = DashboardWindowController(url: settings.dashboardURL)
        dashboardWindow = controller
        controller.showAndFocus()
    }

    @objc private func openSettings() {
        if let window = settingsWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        let view = SettingsView(initial: settings) { [weak self] updated in
            self?.applyUpdated(settings: updated)
        }
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.title = "TeamAI Settings"
        window.styleMask = [.titled, .closable, .miniaturizable]
        window.setContentSize(NSSize(width: 520, height: 320))
        window.isReleasedWhenClosed = false
        window.center()
        settingsWindow = window
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func applyUpdated(settings new: AppSettings) {
        let portChanged = new.port != settings.port
        let pythonChanged = new.pythonPath != settings.pythonPath
        let workspaceChanged = new.workspace != settings.workspace

        settings = new
        settings.save()
        workspaceMenuItem.title = "Workspace: \(settings.workspace)"

        let result = LaunchAtLogin.set(new.launchAtLogin)
        if case .failure(let error) = result {
            Notifier.shared.notify(title: "Launch-at-login error", body: error.localizedDescription)
        }

        daemon.update(settings: settings)
        watcher.update(settings: settings)
        dashboardWindow?.update(url: settings.dashboardURL)

        if portChanged || pythonChanged || workspaceChanged {
            daemon.restart()
        }
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }
}

struct SettingsView: View {
    @State private var port: String
    @State private var pythonPath: String
    @State private var workspace: String
    @State private var launchAtLogin: Bool
    private let onSave: (AppSettings) -> Void

    init(initial: AppSettings, onSave: @escaping (AppSettings) -> Void) {
        _port = State(initialValue: String(initial.port))
        _pythonPath = State(initialValue: initial.pythonPath)
        _workspace = State(initialValue: initial.workspace)
        _launchAtLogin = State(initialValue: initial.launchAtLogin)
        self.onSave = onSave
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("TeamAI").font(.title2).bold()
            Form {
                LabeledContent("Port") {
                    TextField("8000", text: $port)
                        .frame(width: 100)
                }
                LabeledContent("Python") {
                    HStack {
                        TextField("/path/to/python", text: $pythonPath)
                        Button("Browse…", action: pickPython)
                    }
                }
                LabeledContent("Workspace") {
                    HStack {
                        TextField("/path/to/workspace", text: $workspace)
                        Button("Browse…", action: pickWorkspace)
                    }
                }
                Toggle("Launch at login", isOn: $launchAtLogin)
            }
            HStack {
                Spacer()
                Button("Save") {
                    let portValue = Int(port) ?? AppSettings.defaultPort
                    let updated = AppSettings(
                        port: portValue,
                        pythonPath: pythonPath,
                        workspace: workspace,
                        launchAtLogin: launchAtLogin
                    )
                    onSave(updated)
                    NSApp.keyWindow?.close()
                }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 520, height: 320)
    }

    private func pickPython() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.treatsFilePackagesAsDirectories = true
        if panel.runModal() == .OK, let url = panel.url {
            pythonPath = url.path
        }
    }

    private func pickWorkspace() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            workspace = url.path
        }
    }
}
