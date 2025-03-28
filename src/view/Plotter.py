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

class Plotter:
    def __init__(self, layout):
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 10), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.fig_spectrum, self.ax_spectrum = plt.subplots(figsize=(8, 3), tight_layout=True)
        self.canvas_spectrum = FigureCanvas(self.fig_spectrum)
        layout.addWidget(self.canvas_spectrum)

    def update_plot(self, t, message, carrier, modulated, demodulated):
        """Actualiza las gráficas en tiempo real."""
        # Limpiar los ejes antes de dibujar
        for ax in self.axs:
            ax.clear()

        # Colores personalizados para cada gráfica
        self.axs[0].plot(t, message, label="Mensaje", color="blue")
        self.axs[0].set_title("Señal de Mensaje")
        self.axs[0].legend()
        self.axs[0].grid(True)

        self.axs[1].plot(t, carrier, label="Portadora", color="green")
        self.axs[1].set_title("Señal Portadora")
        self.axs[1].legend()
        self.axs[1].grid(True)

        self.axs[2].plot(t, modulated, label="Modulada", color="red")
        self.axs[2].set_title("Señal Modulada")
        self.axs[2].legend()
        self.axs[2].grid(True)

        self.axs[3].plot(t, demodulated, label="Demodulada", color="purple")
        self.axs[3].set_title("Señal Demodulada")
        self.axs[3].legend()
        self.axs[3].grid(True)

        # Verificar la conversión
        self.verify_conversion(message, demodulated)

        # Mostrar espectro de la señal modulada
        self.plot_spectrum(modulated, "Espectro de la Señal Modulada")

        # Redibujar el canvas
        self.canvas.draw()

    def plot_spectrum(self, signal, title):
        """Grafica el espectro de frecuencia de una señal."""
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1 / 1000)  # Asumiendo un muestreo de 1000 Hz
        self.ax_spectrum.clear()
        self.ax_spectrum.plot(xf, np.abs(yf))
        self.ax_spectrum.set_title(title)
        self.ax_spectrum.set_xlim(0, 100)  # Limit frequency range for better visualization
        self.ax_spectrum.grid(True)
        self.canvas_spectrum.draw()
    
    def verify_conversion(self, message_signal, demodulated_signal):
        """Verifica la precisión de la demodulación."""
        error = np.abs(message_signal - demodulated_signal)
        print(f"Error máximo: {np.max(error)}")
        print(f"Error promedio: {np.mean(error)}")