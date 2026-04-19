import Foundation
import ServiceManagement

enum SettingsKey {
    static let port = "TeamAI.port"
    static let pythonPath = "TeamAI.pythonPath"
    static let workspace = "TeamAI.workspace"
    static let launchAtLogin = "TeamAI.launchAtLogin"
}

struct AppSettings {
    var port: Int
    var pythonPath: String
    var workspace: String
    var launchAtLogin: Bool

    static let defaultPort = 8000

    static func load() -> AppSettings {
        let defaults = UserDefaults.standard
        let port = defaults.object(forKey: SettingsKey.port) as? Int ?? defaultPort
        let python = defaults.string(forKey: SettingsKey.pythonPath) ?? AppSettings.detectPython()
        let workspace = defaults.string(forKey: SettingsKey.workspace) ?? AppSettings.detectWorkspace()
        let launchAtLogin = defaults.bool(forKey: SettingsKey.launchAtLogin)
        return AppSettings(port: port, pythonPath: python, workspace: workspace, launchAtLogin: launchAtLogin)
    }

    func save() {
        let defaults = UserDefaults.standard
        defaults.set(port, forKey: SettingsKey.port)
        defaults.set(pythonPath, forKey: SettingsKey.pythonPath)
        defaults.set(workspace, forKey: SettingsKey.workspace)
        defaults.set(launchAtLogin, forKey: SettingsKey.launchAtLogin)
    }

    var healthURL: URL {
        URL(string: "http://127.0.0.1:\(port)/healthz")!
    }

    var dashboardURL: URL {
        URL(string: "http://127.0.0.1:\(port)/dashboard")!
    }

    var summaryURL: URL {
        URL(string: "http://127.0.0.1:\(port)/v1/dashboard/summary")!
    }

    func eventStreamURL(jobID: String) -> URL {
        URL(string: "http://127.0.0.1:\(port)/v1/jobs/\(jobID)/events/stream")!
    }

    private static func detectWorkspace() -> String {
        let candidates = [
            "/Users/home/Documents/teamAI",
            FileManager.default.currentDirectoryPath,
            NSHomeDirectory(),
        ]
        for candidate in candidates {
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: candidate, isDirectory: &isDir), isDir.boolValue {
                return candidate
            }
        }
        return NSHomeDirectory()
    }

    private static func detectPython() -> String {
        let workspace = detectWorkspace()
        let venvPython = (workspace as NSString).appendingPathComponent(".venv/bin/python")
        if FileManager.default.isExecutableFile(atPath: venvPython) {
            return venvPython
        }
        for fallback in ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"] {
            if FileManager.default.isExecutableFile(atPath: fallback) {
                return fallback
            }
        }
        return "/usr/bin/python3"
    }
}

enum LaunchAtLogin {
    static func set(_ enabled: Bool) -> Result<Void, Error> {
        let service = SMAppService.mainApp
        do {
            if enabled {
                if service.status != .enabled {
                    try service.register()
                }
            } else {
                if service.status == .enabled {
                    try service.unregister()
                }
            }
            return .success(())
        } catch {
            return .failure(error)
        }
    }

    static var isEnabled: Bool {
        SMAppService.mainApp.status == .enabled
    }
}
