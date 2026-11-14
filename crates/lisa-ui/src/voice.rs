//! macOS speech input and spoken replies.

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::Bool;
use objc2_avf_audio::{
    AVAudioApplication, AVAudioApplicationRecordPermission, AVAudioEngine, AVAudioInputNode,
    AVAudioPCMBuffer, AVAudioTime, AVSpeechBoundary, AVSpeechSynthesizer, AVSpeechUtterance,
};
use objc2_foundation::{NSError, NSString};
use objc2_speech::{
    SFSpeechAudioBufferRecognitionRequest, SFSpeechRecognitionResult, SFSpeechRecognizer,
    SFSpeechRecognizerAuthorizationStatus,
};
use std::ptr::NonNull;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Permission {
    Authorized,
    NotDetermined,
    Denied,
}

pub fn speech_permission() -> Permission {
    match unsafe { SFSpeechRecognizer::authorizationStatus() } {
        SFSpeechRecognizerAuthorizationStatus::Authorized => Permission::Authorized,
        SFSpeechRecognizerAuthorizationStatus::NotDetermined => Permission::NotDetermined,
        _ => Permission::Denied,
    }
}

pub fn microphone_permission() -> Permission {
    match unsafe { AVAudioApplication::sharedInstance().recordPermission() } {
        AVAudioApplicationRecordPermission::Granted => Permission::Authorized,
        AVAudioApplicationRecordPermission::Undetermined => Permission::NotDetermined,
        _ => Permission::Denied,
    }
}

pub fn request_speech_permission(callback: impl Fn(bool) + Send + Sync + 'static) {
    let block = RcBlock::new(move |status| {
        callback(status == SFSpeechRecognizerAuthorizationStatus::Authorized);
    });
    unsafe { SFSpeechRecognizer::requestAuthorization(&block) };
}

pub fn request_microphone_permission(callback: impl Fn(bool) + Send + Sync + 'static) {
    let block = RcBlock::new(move |granted: Bool| callback(granted.as_bool()));
    unsafe { AVAudioApplication::requestRecordPermissionWithCompletionHandler(&block) };
}

#[derive(Debug)]
pub struct VoiceCapture {
    engine: Retained<AVAudioEngine>,
    input: Retained<AVAudioInputNode>,
    request: Retained<SFSpeechAudioBufferRecognitionRequest>,
    task: Retained<objc2_speech::SFSpeechRecognitionTask>,
    active: bool,
}

impl VoiceCapture {
    pub fn start(
        on_transcript: impl Fn(String, bool) + Send + Sync + 'static,
        on_error: impl Fn(String) + Send + Sync + 'static,
    ) -> anyhow::Result<Self> {
        let recognizer = unsafe { SFSpeechRecognizer::new() };
        anyhow::ensure!(
            unsafe { recognizer.isAvailable() },
            "Apple speech recognition is currently unavailable"
        );
        anyhow::ensure!(
            unsafe { recognizer.supportsOnDeviceRecognition() },
            "On-device speech recognition is unavailable for the current system language"
        );

        let engine = unsafe { AVAudioEngine::new() };
        let input = unsafe { engine.inputNode() };
        let format = unsafe { input.outputFormatForBus(0) };
        anyhow::ensure!(
            unsafe { format.sampleRate() > 0.0 && format.channelCount() > 0 },
            "No microphone input is available"
        );

        let request = unsafe { SFSpeechAudioBufferRecognitionRequest::new() };
        unsafe {
            request.setShouldReportPartialResults(true);
            request.setAddsPunctuation(true);
            request.setRequiresOnDeviceRecognition(true);
        }
        let transcript_callback = Arc::new(on_transcript);
        let error_callback = Arc::new(on_error);
        let handler = RcBlock::new(
            move |result: *mut SFSpeechRecognitionResult, error: *mut NSError| {
                if !result.is_null() {
                    let result = unsafe { &*result };
                    let text = unsafe { result.bestTranscription().formattedString() }.to_string();
                    transcript_callback(text, unsafe { result.isFinal() });
                }
                if result.is_null() && !error.is_null() {
                    let message = unsafe { &*error }.localizedDescription().to_string();
                    error_callback(message);
                }
            },
        );
        let task =
            unsafe { recognizer.recognitionTaskWithRequest_resultHandler(&request, &handler) };

        let request_for_tap = request.clone();
        let tap = RcBlock::new(
            move |buffer: NonNull<AVAudioPCMBuffer>, _time: NonNull<AVAudioTime>| {
                unsafe { request_for_tap.appendAudioPCMBuffer(buffer.as_ref()) };
            },
        );
        unsafe {
            input.installTapOnBus_bufferSize_format_block(
                0,
                1_024,
                Some(&format),
                RcBlock::as_ptr(&tap),
            );
            engine.prepare();
        }
        if let Err(error) = unsafe { engine.startAndReturnError() } {
            unsafe {
                input.removeTapOnBus(0);
                task.cancel();
            }
            anyhow::bail!(error.localizedDescription().to_string());
        }

        Ok(Self {
            engine,
            input,
            request,
            task,
            active: true,
        })
    }

    /// Stop microphone capture and let the recognizer finish its final transcript.
    pub fn finish(&mut self) {
        if !self.active {
            return;
        }
        unsafe {
            self.input.removeTapOnBus(0);
            self.engine.stop();
            self.request.endAudio();
            self.task.finish();
        }
        self.active = false;
    }
}

impl Drop for VoiceCapture {
    fn drop(&mut self) {
        self.finish();
    }
}

#[derive(Debug)]
pub struct VoiceSpeaker {
    synthesizer: Retained<AVSpeechSynthesizer>,
}

impl Default for VoiceSpeaker {
    fn default() -> Self {
        Self {
            synthesizer: unsafe { AVSpeechSynthesizer::new() },
        }
    }
}

impl VoiceSpeaker {
    pub fn speak(&self, text: &str) {
        let text = speech_text(text);
        if text.trim().is_empty() {
            return;
        }
        unsafe {
            self.synthesizer
                .stopSpeakingAtBoundary(AVSpeechBoundary::Immediate);
            let utterance =
                AVSpeechUtterance::speechUtteranceWithString(&NSString::from_str(&text));
            self.synthesizer.speakUtterance(&utterance);
        }
    }

    pub fn stop(&self) {
        unsafe {
            self.synthesizer
                .stopSpeakingAtBoundary(AVSpeechBoundary::Immediate);
        }
    }
}

/// Remove common Markdown punctuation while preserving the words read aloud.
fn speech_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '`' | '*' | '_' | '#' | '>' => {}
            '[' => {
                let mut label = String::new();
                let mut found_end = false;
                for next in chars.by_ref() {
                    if next == ']' {
                        found_end = true;
                        break;
                    }
                    label.push(next);
                }
                if found_end && chars.peek() == Some(&'(') {
                    chars.next();
                    for next in chars.by_ref() {
                        if next == ')' {
                            break;
                        }
                    }
                    out.push_str(&label);
                } else {
                    out.push('[');
                    out.push_str(&label);
                    if found_end {
                        out.push(']');
                    }
                }
            }
            _ => out.push(ch),
        }
    }
    out
}
