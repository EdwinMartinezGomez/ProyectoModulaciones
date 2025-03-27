import numpy as np
import matplotlib.pyplot as plt
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import scipy.signal as signal

class ModulationSimulator(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulación de Modulación")
        self.setGeometry(100, 100, 800, 600)

        # Layout principal
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        # Tipos de modulación
        self.mod_types = ["AM", "FM", "PM", "ASK", "FSK", "PSK", "PCM"]
        self.selected_mod = QtWidgets.QComboBox()
        self.selected_mod.addItems(self.mod_types)
        self.selected_mod.currentTextChanged.connect(self.toggle_input_method)

        self.layout.addWidget(QtWidgets.QLabel("Seleccione el tipo de modulación:"))
        self.layout.addWidget(self.selected_mod)

        # Entrada de datos
        self.input_label = QtWidgets.QLabel("Ingrese un mensaje de texto o grabe un audio")
        self.layout.addWidget(self.input_label)

        self.text_input = QtWidgets.QLineEdit()
        self.layout.addWidget(self.text_input)

        self.record_button = QtWidgets.QPushButton("Grabar Audio")
        self.record_button.clicked.connect(self.record_audio)
        self.layout.addWidget(self.record_button)

        # Controles de frecuencia
        self.freq_msg, self.freq_msg_label = self.create_slider("Frecuencia de mensaje", 1, 20, 5)
        self.freq_carrier, self.freq_carrier_label = self.create_slider("Frecuencia de portadora", 5, 50, 20)

        # Gráficos
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 10), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        self.audio_data = None

        # Animación en tiempo real
        self.anim = FuncAnimation(self.fig, self.update_plot, interval=100)

        self.toggle_input_method()

    def create_slider(self, label, min_val, max_val, default):
        """Crea un slider con su etiqueta asociada."""
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        slider.valueChanged.connect(self.update_labels)
        slider_label = QtWidgets.QLabel(f"{label}: {slider.value()} Hz")
        self.layout.addWidget(slider_label)
        self.layout.addWidget(slider)
        return slider, slider_label

    def toggle_input_method(self):
        """Habilita/deshabilita la entrada según el tipo de modulación."""
        mod_type = self.selected_mod.currentText()
        is_digital = mod_type in ["ASK", "FSK", "PSK", "PCM"]
        self.text_input.setEnabled(is_digital)
        self.record_button.setEnabled(not is_digital)

    def record_audio(self):
        """Graba audio de entrada y lo convierte en una señal usable."""
        try:
            duration = 5  # Duración de la grabación en segundos
            self.audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
            sd.wait()
            self.audio_data = self.audio_data.flatten()
        except Exception as e:
            print(f"Error en la grabación: {e}")
            self.audio_data = None

    def update_labels(self):
        """Actualiza las etiquetas de los sliders."""
        self.freq_msg_label.setText(f"Frecuencia de mensaje: {self.freq_msg.value()} Hz")
        self.freq_carrier_label.setText(f"Frecuencia de portadora: {self.freq_carrier.value()} Hz")

    def generate_signals(self):
        """Genera las señales para la simulación de modulación."""
        try:
            t = np.linspace(0, 1, 1000)  # Vector de tiempo
            mod_type = self.selected_mod.currentText()

            # Generación del mensaje
            if mod_type in ["ASK", "FSK", "PSK", "PCM"]:
                text = self.text_input.text()
                message = self.text_to_signal(text, len(t))
            else:
                message = self.audio_to_signal(t) if self.audio_data is not None else np.sin(2 * np.pi * self.freq_msg.value() * t)

            # Asegurar que `message` tenga la misma longitud que `t`
            message = self.ensure_length(message, len(t))

            # Generar la portadora
            carrier = np.sin(2 * np.pi * self.freq_carrier.value() * t)

            # Aplicar modulación
            modulated, demodulated = self.apply_modulation(mod_type, message, carrier, t)

            return t, message, carrier, modulated, demodulated

        except Exception as e:
            print(f"Error en generate_signals: {e}")
            return t, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)

    def ensure_length(self, signal, length):
        """Ajusta la señal para que tenga exactamente la longitud deseada."""
        return np.resize(signal, length)

    def text_to_signal(self, text, length):
        """Convierte texto en una señal binaria."""
        try:
            if text:
                message = np.array([int(bit) for bit in ''.join(format(ord(c), '08b') for c in text)])
                message = np.repeat(message, length // len(message))[:length]
            else:
                message = np.zeros(length)
            return self.ensure_length(message, length)
        except Exception as e:
            print(f"Error en text_to_signal: {e}")
            return np.zeros(length)

    def audio_to_signal(self, t):
        """Convierte el audio grabado en una señal analógica."""
        if self.audio_data is not None:
            return np.interp(t, np.linspace(0, 1, len(self.audio_data)), self.audio_data)
        return np.zeros_like(t)


    def apply_modulation(self, mod_type, message, carrier, t):
        """Aplica la modulación seleccionada y hace la demodulación correcta."""
        try:
            sampling_rate = len(t)

            if mod_type == "AM":
                modulated = (1 + message) * carrier
                analytic = signal.hilbert(modulated)
                demodulated = np.abs(analytic) - 1

            elif mod_type == "FM":
                modulated = np.sin(2 * np.pi * self.freq_carrier.value() * t + 2 * np.pi * message)
                analytic = signal.hilbert(modulated)
                instantaneous_phase = np.unwrap(np.angle(analytic))
                instantaneous_frequency = np.gradient(instantaneous_phase) * sampling_rate / (2 * np.pi)
                demodulated = instantaneous_frequency - self.freq_carrier.value()

            elif mod_type == "PM":
                modulated = np.sin(2 * np.pi * self.freq_carrier.value() * t + np.pi * message)
                analytic = signal.hilbert(modulated)
                instantaneous_phase = np.unwrap(np.angle(analytic))
                demodulated = instantaneous_phase

            elif mod_type == "ASK":
                modulated = (message > 0) * carrier
                demodulated = (modulated * carrier) > 0

            elif mod_type == "FSK":
                f1 = self.freq_carrier.value()
                f2 = f1 + 50  # Subimos el salto para que la demodulación sea más clara

                samples_per_bit = int(len(t) / len(message))
                duration = t[-1] - t[0]
                bit_duration = duration / len(message)

                modulated = np.zeros(len(t))
                phase = 0

                # ===========================
                # Modulación FSK (fase continua)
                # ===========================
                for i, bit in enumerate(message):
                    start = i * samples_per_bit
                    end = (i + 1) * samples_per_bit
                    freq = f2 if bit == 1 else f1
                    t_rel = t[start:end] - t[start]
                    modulated[start:end] = np.sin(phase + 2 * np.pi * freq * t_rel)
                    phase += 2 * np.pi * freq * (t[end-1] - t[start]) + 2 * np.pi * freq * (t[1] - t[0])

                # ===============================
                # DEMODULACIÓN FSK (Detector de Energía)
                # ===============================
                demodulated_bits = []

                for i in range(len(message)):
                    start = i * samples_per_bit
                    end = (i + 1) * samples_per_bit

                    segment = modulated[start:end]
                    t_rel = np.linspace(0, bit_duration, samples_per_bit, endpoint=False)

                    ref1 = np.sin(2 * np.pi * f1 * t_rel)
                    ref2 = np.sin(2 * np.pi * f2 * t_rel)

                    # Producto escalar (correlación coherente)
                    corr_f1 = np.dot(segment, ref1)
                    corr_f2 = np.dot(segment, ref2)

                    # Determinación por energía
                    bit = 1 if corr_f2 ** 2 > corr_f1 ** 2 else 0
                    demodulated_bits.append(bit)

                # Reconstruir señal para graficar
                demodulated = np.zeros(len(t))
                for i, bit in enumerate(demodulated_bits):
                    start = i * samples_per_bit
                    end = (i + 1) * samples_per_bit
                    demodulated[start:end] = bit

            elif mod_type == "PSK":
                modulated = np.sin(2 * np.pi * self.freq_carrier.value() * t + np.pi * message)

                # Demodulación coherente
                carrier_ref = np.sin(2 * np.pi * self.freq_carrier.value() * t)
                mixed = modulated * carrier_ref

                # Filtro pasa bajos (promedio móvil)
                kernel_size = 50
                filtered = np.convolve(mixed, np.ones(kernel_size) / kernel_size, mode='same')

                demodulated = (filtered < 0).astype(float)

            elif mod_type == "PCM":
                modulated = np.round(message * 2) / 2
                demodulated = modulated

            else:
                modulated = demodulated = np.zeros_like(t)

            return self.ensure_length(modulated, len(t)), self.ensure_length(demodulated, len(t))

        except Exception as e:
            print(f"Error en apply_modulation: {e}")
            return np.zeros_like(t), np.zeros_like(t)

    def update_plot(self, frame):
        """Actualiza las gráficas en tiempo real."""
        t, message, carrier, modulated, demodulated = self.generate_signals()

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

        # Redibujar el canvas
        self.canvas.draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = ModulationSimulator()
    window.show()
    app.exec_()