import AppKit
import WebKit

final class DashboardWindowController: NSWindowController, NSWindowDelegate {
    private let webView: WKWebView
    private var dashboardURL: URL

    init(url: URL) {
        self.dashboardURL = url
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        self.webView = WKWebView(frame: .zero, configuration: config)

        let style: NSWindow.StyleMask = [.titled, .closable, .resizable, .miniaturizable]
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 800),
            styleMask: style,
            backing: .buffered,
            defer: false
        )
        window.title = "TeamAI Dashboard"
        window.contentView = webView
        window.center()
        window.setFrameAutosaveName("TeamAIDashboardWindow")
        window.isReleasedWhenClosed = false

        super.init(window: window)
        window.delegate = self
        loadDashboard()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func update(url: URL) {
        guard url != dashboardURL else { return }
        dashboardURL = url
        loadDashboard()
    }

    func reload() {
        loadDashboard()
    }

    private func loadDashboard() {
        var request = URLRequest(url: dashboardURL)
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        webView.load(request)
    }

    func showAndFocus() {
        if window?.isVisible != true {
            window?.makeKeyAndOrderFront(nil)
        } else {
            window?.orderFrontRegardless()
        }
        NSApp.activate(ignoringOtherApps: true)
    }
}
