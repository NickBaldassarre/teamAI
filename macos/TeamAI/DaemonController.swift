import Foundation

enum DaemonState: Equatable {
    case stopped
    case starting
    case ready
    case error(String)
}

final class DaemonController {
    private(set) var state: DaemonState = .stopped {
        didSet {
            if oldValue != state {
                onStateChange?(state)
            }
        }
    }

    private var settings: AppSettings
    private var launcherProcess: Process?
    private var pollTimer: DispatchSourceTimer?
    private var startedDaemonPID: pid_t?

    var onStateChange: ((DaemonState) -> Void)?

    init(settings: AppSettings) {
        self.settings = settings
    }

    func update(settings: AppSettings) {
        self.settings = settings
    }

    func start() {
        if case .ready = state { return }
        if case .starting = state { return }
        state = .starting

        guard FileManager.default.isExecutableFile(atPath: settings.pythonPath) else {
            state = .error("Python not found at \(settings.pythonPath)")
            return
        }

        let workspaceURL = URL(fileURLWithPath: settings.workspace, isDirectory: true)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: settings.pythonPath)
        process.arguments = [
            "-m", "teamai", "daemon", "start",
            "--host", "127.0.0.1",
            "--port", String(settings.port),
            "--workspace", settings.workspace,
        ]
        process.currentDirectoryURL = workspaceURL

        var environment = ProcessInfo.processInfo.environment
        environment["TEAMAI_WORKSPACE_ROOT"] = settings.workspace
        environment["TEAMAI_HOST"] = "127.0.0.1"
        environment["TEAMAI_PORT"] = String(settings.port)
        // Prefer the workspace source over any stale `teamai` install in
        // the venv's site-packages so the daemon CLI is always in sync
        // with the checked-out code.
        let existingPYTHONPATH = environment["PYTHONPATH"] ?? ""
        if existingPYTHONPATH.isEmpty {
            environment["PYTHONPATH"] = settings.workspace
        } else {
            environment["PYTHONPATH"] = "\(settings.workspace):\(existingPYTHONPATH)"
        }
        process.environment = environment

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
            launcherProcess = process
            beginHealthPoll()
        } catch {
            state = .error("Failed to launch daemon: \(error.localizedDescription)")
        }
    }

    func stop(timeout: TimeInterval = 5.0) {
        cancelPoll()
        let pid = startedDaemonPID ?? readPIDFile()
        startedDaemonPID = nil

        if let pid {
            kill(pid, SIGTERM)
            let deadline = Date().addingTimeInterval(timeout)
            while Date() < deadline {
                if !pidAlive(pid) { break }
                Thread.sleep(forTimeInterval: 0.2)
            }
            if pidAlive(pid) {
                kill(pid, SIGKILL)
            }
            removePIDFile()
        }

        if let launcher = launcherProcess, launcher.isRunning {
            launcher.terminate()
        }
        launcherProcess = nil
        state = .stopped
    }

    func restart() {
        stop()
        start()
    }

    private func beginHealthPoll() {
        cancelPoll()
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now() + .milliseconds(500), repeating: .milliseconds(500))
        let deadline = Date().addingTimeInterval(30.0)
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            if Date() > deadline {
                self.cancelPoll()
                DispatchQueue.main.async {
                    if case .starting = self.state {
                        self.state = .error("Daemon did not become healthy within 30s")
                    }
                }
                return
            }
            if self.probeHealth() {
                self.cancelPoll()
                if let pid = self.readPIDFile() {
                    self.startedDaemonPID = pid
                }
                DispatchQueue.main.async {
                    self.state = .ready
                }
            }
        }
        timer.resume()
        pollTimer = timer
    }

    private func cancelPoll() {
        pollTimer?.cancel()
        pollTimer = nil
    }

    private func probeHealth() -> Bool {
        var request = URLRequest(url: settings.healthURL)
        request.timeoutInterval = 1.5
        let semaphore = DispatchSemaphore(value: 0)
        var ok = false
        let task = URLSession.shared.dataTask(with: request) { _, response, _ in
            if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                ok = true
            }
            semaphore.signal()
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + 2.0)
        return ok
    }

    private func readPIDFile() -> pid_t? {
        let path = (NSHomeDirectory() as NSString).appendingPathComponent(".teamai/daemon.pid")
        guard let raw = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return pid_t(trimmed)
    }

    private func removePIDFile() {
        let path = (NSHomeDirectory() as NSString).appendingPathComponent(".teamai/daemon.pid")
        try? FileManager.default.removeItem(atPath: path)
    }

    private func pidAlive(_ pid: pid_t) -> Bool {
        kill(pid, 0) == 0
    }
}
