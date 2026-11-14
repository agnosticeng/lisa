import SwiftUI
import AVFoundation
import Speech

@MainActor final class SpeechController: ObservableObject {
    @Published var recording = false
    private let engine = AVAudioEngine()
    private let synthesizer = AVSpeechSynthesizer()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var hasTap = false
    func start(onText: @escaping (String) -> Void, onError: @escaping (String) -> Void) async {
        stop()
        let speech = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0 == .authorized) }
        }
        guard speech else { onError("Enable Speech Recognition for Lisa in system settings."); return }
        let microphone = await AVAudioApplication.requestRecordPermission()
        guard microphone else { onError("Enable Microphone access for Lisa in system settings."); return }
        guard let recognizer = SFSpeechRecognizer(), recognizer.isAvailable, recognizer.supportsOnDeviceRecognition else {
            onError("On-device speech recognition is unavailable for the current language."); return
        }
        do {
            #if os(iOS)
            let audio = AVAudioSession.sharedInstance()
            try audio.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audio.setActive(true, options: .notifyOthersOnDeactivation)
            #endif
            let input = engine.inputNode
            let format = input.outputFormat(forBus: 0)
            guard format.sampleRate > 0 && format.channelCount > 0 else { throw CocoaError(.featureUnsupported) }
            let request = SFSpeechAudioBufferRecognitionRequest()
            request.shouldReportPartialResults = true; request.requiresOnDeviceRecognition = true; request.addsPunctuation = true
            self.request = request
            task = recognizer.recognitionTask(with: request) { [weak self] result, error in
                Task { @MainActor in
                    guard let self, self.recording else { return }
                    if let result { onText(result.bestTranscription.formattedString); if result.isFinal { self.stop() } }
                    else if let error { self.stop(); onError(error.localizedDescription) }
                }
            }
            input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in request.append(buffer) }
            hasTap = true; engine.prepare(); try engine.start(); recording = true
        } catch { stop(); onError(error.localizedDescription) }
    }
    func stop() {
        recording = false
        if hasTap { engine.inputNode.removeTap(onBus: 0); hasTap = false }
        engine.stop(); request?.endAudio(); task?.cancel(); task = nil; request = nil
        synthesizer.stopSpeaking(at: .immediate)
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        #endif
    }
    func speak(_ text: String) {
        guard !text.isEmpty else { return }; stop()
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio)
        try? AVAudioSession.sharedInstance().setActive(true)
        #endif
        synthesizer.speak(AVSpeechUtterance(string: text))
    }
}
struct VoiceButton: View {
    @ObservedObject var controller: SpeechController
    let disabled: Bool
    let onText: (String) -> Void
    let onError: (String) -> Void
    var body: some View {
        Button {
            if controller.recording { controller.stop() }
            else { Task { await controller.start(onText: onText, onError: onError) } }
        } label: { Image(systemName: controller.recording ? "mic.fill" : "mic") }
            .foregroundStyle(controller.recording ? Color.red : Color.primary)
            .help(controller.recording ? "Stop recording" : "Dictate a message")
            .disabled(disabled)
    }
}
