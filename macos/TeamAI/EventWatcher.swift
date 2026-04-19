import Foundation

struct DashboardJob: Decodable, Equatable {
    let job_id: String
    let status: String
    let task: String?
}

struct DashboardSafety: Decodable, Equatable {
    let allow_writes: Bool
    let allow_shell: Bool
    let posture: String
}

struct DashboardApprovalItem: Decodable, Equatable {
    let approval_id: String?
    let target_path: String?
    let task_id: String?
}

struct DashboardApprovals: Decodable, Equatable {
    let count: Int
    let items: [DashboardApprovalItem]?
}

struct DashboardJobs: Decodable {
    let recent: [DashboardJob]
}

struct DashboardSummary: Decodable {
    let safety: DashboardSafety
    let jobs: DashboardJobs
    let approvals: DashboardApprovals
}

struct SSEEvent {
    let event: String
    let data: String
}

final class EventWatcher {
    private var settings: AppSettings
    private var summaryTimer: DispatchSourceTimer?
    private let session: URLSession

    private var lastJobStatus: [String: String] = [:]
    private var lastJobTask: [String: String] = [:]
    private var lastApprovalCount: Int = -1
    private var lastApprovalIDs: Set<String> = []
    private var lastSafetyPosture: String?
    private var trackedStreams: [String: SSEStream] = [:]
    private let streamQueue = DispatchQueue(label: "teamai.event.streams")

    init(settings: AppSettings) {
        self.settings = settings
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 5
        config.timeoutIntervalForResource = 600
        self.session = URLSession(configuration: config)
    }

    func update(settings: AppSettings) {
        self.settings = settings
    }

    func start() {
        stop()
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now() + .seconds(2), repeating: .seconds(5))
        timer.setEventHandler { [weak self] in
            self?.pollSummary()
        }
        timer.resume()
        summaryTimer = timer
    }

    func stop() {
        summaryTimer?.cancel()
        summaryTimer = nil
        streamQueue.sync {
            for (_, stream) in trackedStreams {
                stream.cancel()
            }
            trackedStreams.removeAll()
        }
    }

    private func pollSummary() {
        var request = URLRequest(url: settings.summaryURL)
        request.timeoutInterval = 4
        let task = session.dataTask(with: request) { [weak self] data, response, _ in
            guard let self,
                  let http = response as? HTTPURLResponse,
                  http.statusCode == 200,
                  let data else { return }
            do {
                let summary = try JSONDecoder().decode(DashboardSummary.self, from: data)
                self.handleSummary(summary)
            } catch {
                // ignore parse errors silently — keep polling
            }
        }
        task.resume()
    }

    private func handleSummary(_ summary: DashboardSummary) {
        if let posture = lastSafetyPosture, posture != summary.safety.posture {
            Notifier.shared.notify(
                title: "Write mode changed",
                body: "Posture is now \(summary.safety.posture)."
            )
        }
        lastSafetyPosture = summary.safety.posture

        let approvalIDs = Set((summary.approvals.items ?? []).compactMap { $0.approval_id })
        if lastApprovalCount >= 0 {
            let newIDs = approvalIDs.subtracting(lastApprovalIDs)
            if summary.approvals.count > lastApprovalCount, !newIDs.isEmpty {
                for newID in newIDs {
                    let target = (summary.approvals.items ?? [])
                        .first(where: { $0.approval_id == newID })?
                        .target_path ?? "(unknown)"
                    Notifier.shared.notify(
                        title: "Approval pending",
                        body: target
                    )
                }
            } else if summary.approvals.count > lastApprovalCount {
                Notifier.shared.notify(
                    title: "Approval pending",
                    body: "\(summary.approvals.count) approvals waiting."
                )
            }
        }
        lastApprovalCount = summary.approvals.count
        lastApprovalIDs = approvalIDs

        for job in summary.jobs.recent {
            let previous = lastJobStatus[job.job_id]
            if previous == nil {
                lastJobStatus[job.job_id] = job.status
                if let task = job.task { lastJobTask[job.job_id] = task }
                if job.status == "running" || job.status == "queued" {
                    subscribe(to: job.job_id)
                }
                continue
            }
            if previous != job.status {
                lastJobStatus[job.job_id] = job.status
                if previous == "queued" || previous == "running" {
                    if job.status == "completed" {
                        Notifier.shared.notify(
                            title: "Run completed",
                            body: shortLabel(for: job)
                        )
                    } else if job.status == "failed" {
                        Notifier.shared.notify(
                            title: "Run failed",
                            body: shortLabel(for: job)
                        )
                    }
                }
            }
            if (job.status == "running" || job.status == "queued") {
                subscribe(to: job.job_id)
            } else {
                unsubscribe(from: job.job_id)
            }
        }
    }

    private func shortLabel(for job: DashboardJob) -> String {
        let task = job.task ?? lastJobTask[job.job_id]
        let prefix = "Run \(job.job_id.prefix(8))"
        guard let task, !task.isEmpty else { return prefix }
        let trimmed = task.trimmingCharacters(in: .whitespacesAndNewlines)
        let snippet = trimmed.prefix(80)
        return "\(prefix): \(snippet)"
    }

    private func subscribe(to jobID: String) {
        streamQueue.sync {
            guard trackedStreams[jobID] == nil else { return }
            let url = settings.eventStreamURL(jobID: jobID)
            let stream = SSEStream(url: url, session: session)
            stream.onEvent = { [weak self] event in
                self?.handleStream(event: event, jobID: jobID)
            }
            stream.start()
            trackedStreams[jobID] = stream
        }
    }

    private func unsubscribe(from jobID: String) {
        streamQueue.sync {
            trackedStreams[jobID]?.cancel()
            trackedStreams.removeValue(forKey: jobID)
        }
    }

    private func handleStream(event: SSEEvent, jobID: String) {
        // Streams currently inform us about live progress only. Final
        // state transitions are still authoritatively detected via the
        // 5s summary poll, so SSE is a future-friendly signal channel.
        _ = event
        _ = jobID
    }
}

final class SSEStream: NSObject, URLSessionDataDelegate {
    private let url: URL
    private let session: URLSession
    private var task: URLSessionDataTask?
    private var buffer = Data()
    private var dedicatedSession: URLSession?

    var onEvent: ((SSEEvent) -> Void)?

    init(url: URL, session: URLSession) {
        self.url = url
        self.session = session
        super.init()
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 0
        config.timeoutIntervalForResource = 0
        self.dedicatedSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    func start() {
        var request = URLRequest(url: url)
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 0
        task = dedicatedSession?.dataTask(with: request)
        task?.resume()
    }

    func cancel() {
        task?.cancel()
        task = nil
        dedicatedSession?.invalidateAndCancel()
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        buffer.append(data)
        flushBuffer()
    }

    private func flushBuffer() {
        guard let text = String(data: buffer, encoding: .utf8) else { return }
        let chunks = text.components(separatedBy: "\n\n")
        if chunks.count <= 1 { return }
        for chunk in chunks.dropLast() {
            var eventName = "message"
            var dataLines: [String] = []
            for line in chunk.split(separator: "\n", omittingEmptySubsequences: false) {
                let lineString = String(line)
                if lineString.hasPrefix(":") { continue }
                if lineString.hasPrefix("event:") {
                    eventName = lineString.dropFirst("event:".count).trimmingCharacters(in: .whitespaces)
                } else if lineString.hasPrefix("data:") {
                    dataLines.append(lineString.dropFirst("data:".count).trimmingCharacters(in: .whitespaces))
                }
            }
            let payload = dataLines.joined(separator: "\n")
            if !payload.isEmpty {
                onEvent?(SSEEvent(event: eventName, data: payload))
            }
        }
        if let lastChunk = chunks.last {
            buffer = lastChunk.data(using: .utf8) ?? Data()
        } else {
            buffer.removeAll()
        }
    }
}
