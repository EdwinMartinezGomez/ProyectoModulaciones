import numpy as np
import matplotlib.pyplot as plt
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation
import sounddevice as sd
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
import scipy.signal as signal
import scipy


class AudioRecorder:
    def __init__(self):
        self.audio_data = None
        self.is_recording = False

    def record_audio(self, duration=5):
        """Graba audio de entrada y lo convierte en una señal usable."""
        if not self.is_recording:
            self.is_recording = True
            QTimer.singleShot(5000, self.stop_recording)  # Grabar durante 5 segundos
            try:
                duration = 5  # Duración de la grabación en segundos
                self.audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
            except Exception as e:
                print(f"Error en la grabación: {e}")
                self.audio_data = None
        else:
            self.stop_recording()

    def stop_recording(self):
        """Detiene la grabación de audio."""
        self.is_recording = False
        sd.wait()
        if self.audio_data is not None:
            self.audio_data = self.audio_data.flatten()

    def audio_to_signal(self, t):
        """Convierte el audio grabado en una señal analógica."""
        if self.audio_data is not None:
            return np.interp(t, np.linspace(0, 1, len(self.audio_data)), self.audio_data)
        return np.zeros_like(t)
