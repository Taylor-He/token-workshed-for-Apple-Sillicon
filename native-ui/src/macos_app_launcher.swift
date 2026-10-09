import AppKit
import Foundation

final class AppLauncherDelegate: NSObject, NSApplicationDelegate {
    private let arguments: [String]
    private var runnerProcess: Process?

    init(arguments: [String]) {
        self.arguments = arguments
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let executable = arguments.first else {
            NSLog("token-workshed launcher did not receive an executable path.")
            NSApplication.shared.terminate(nil)
            return
        }

        let appDirectory = URL(fileURLWithPath: executable).deletingLastPathComponent()
        let executableName = URL(fileURLWithPath: executable).lastPathComponent
        let runner = appDirectory.appendingPathComponent("\(executableName)-runner")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [runner.path] + Array(arguments.dropFirst())
        process.environment = ProcessInfo.processInfo.environment
        process.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                self?.runnerProcess = nil
                NSApplication.shared.terminate(nil)
            }
        }

        do {
            runnerProcess = process
            try process.run()
        } catch {
            runnerProcess = nil
            NSLog("Could not start token-workshed: %@", error.localizedDescription)
            NSApplication.shared.terminate(nil)
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        if let process = runnerProcess, process.isRunning {
            process.terminate()
        }
    }
}

let application = NSApplication.shared
application.setActivationPolicy(.regular)
let launcherDelegate = AppLauncherDelegate(arguments: CommandLine.arguments)
application.delegate = launcherDelegate
application.activate(ignoringOtherApps: true)
application.run()
