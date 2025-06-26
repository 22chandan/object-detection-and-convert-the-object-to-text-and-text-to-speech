"""Audio management for text-to-speech announcements"""
import threading
import queue
import subprocess
import platform


class AudioManager:
    def __init__(self, queue_size=2):
        self.detection_queue = queue.Queue(maxsize=queue_size)
        self.audio_thread = None
        self.running = True
        self._start_audio_thread()

    def _start_audio_thread(self):
        """Start background thread for audio processing"""

        def audio_worker():
            while self.running:
                try:
                    text = self.detection_queue.get(timeout=1.0)
                    if text:
                        self._speak_fast(text)
                except queue.Empty:
                    continue
                except Exception:
                    break

        self.audio_thread = threading.Thread(target=audio_worker, daemon=True)
        self.audio_thread.start()

    def _speak_fast(self, text):
        """Optimized offline TTS"""
        try:
            system = platform.system().lower()

            if system == "windows":
                subprocess.run([
                    "powershell", "-Command",
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$speak.Rate = 2; $speak.Speak('{text}')"
                ], capture_output=True, timeout=3)

            elif system == "darwin":  # macOS
                subprocess.run(["say", "-r", "200", text], timeout=3)

            elif system == "linux":
                subprocess.run(["espeak", "-s", "180", text], timeout=3)

        except Exception:
            pass  # Silent fail for speed

    def announce(self, text):
        """Add text to announcement queue"""
        if not self.detection_queue.full():
            try:
                self.detection_queue.put_nowait(text)
                return True
            except queue.Full:
                pass
        return False

    def stop(self):
        """Stop audio manager"""
        self.running = False
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
