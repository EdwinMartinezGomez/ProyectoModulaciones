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


class ModulationSimulator(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulación de Modulación")
        self.setGeometry(100, 100, 800, 800)

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

        # Indicador de grabación
        self.recording_indicator = QtWidgets.QLabel("")
        self.layout.addWidget(self.recording_indicator)

        # Controles de frecuencia
        self.freq_msg, self.freq_msg_label = self.create_slider("Frecuencia de mensaje", 1, 20, 5)
        self.freq_carrier, self.freq_carrier_label = self.create_slider("Frecuencia de portadora", 5, 50, 20)

        # Control de desviación de fase para PM
        self.phase_dev, self.phase_dev_label = self.create_slider("Desviación de fase (PM)", 1, 10, 5)

        # Gráficos principales
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 10), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # Gráfico adicional para el espectro
        self.fig_spectrum, self.ax_spectrum = plt.subplots(figsize=(8, 3), tight_layout=True)
        self.canvas_spectrum = FigureCanvas(self.fig_spectrum)
        self.layout.addWidget(self.canvas_spectrum)

        self.audio_data = None
        self.is_recording = False

        # Animación en tiempo real
        self.anim = FuncAnimation(self.fig, self.update_plot, interval=100, cache_frame_data=False, save_count=50)

        self.toggle_input_method()

    def create_slider(self, label, min_val, max_val, default):
        """Crea un slider con su etiqueta asociada."""
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        slider.valueChanged.connect(self.update_labels)
        slider_label = QtWidgets.QLabel(f"{label}: {slider.value()}")
        self.layout.addWidget(slider_label)
        self.layout.addWidget(slider)
        return slider, slider_label

    def toggle_input_method(self):
        """Habilita/deshabilita la entrada según el tipo de modulación."""
        mod_type = self.selected_mod.currentText()
        # Para modulación digital (ASK, FSK, PSK) se usa texto; en PCM usaremos audio
        if mod_type == "PCM":
            self.text_input.setEnabled(False)
            self.record_button.setEnabled(True)
        else:
            is_digital = mod_type in ["ASK", "FSK", "PSK"]
            self.text_input.setEnabled(is_digital)
            self.record_button.setEnabled(not is_digital)
        # Mostrar/ocultar control de desviación de fase según modulación
        self.phase_dev.setVisible(mod_type == "PM")
        self.phase_dev_label.setVisible(mod_type == "PM")

    def record_audio(self):
        """Graba audio de entrada y lo convierte en una señal usable."""
        if not self.is_recording:
            self.is_recording = True
            self.record_button.setText("Detener Grabación")
            self.recording_indicator.setText("Grabando...")
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
        self.record_button.setText("Grabar Audio")
        self.recording_indicator.setText("")
        sd.wait()
        if self.audio_data is not None:
            self.audio_data = self.audio_data.flatten()

    def update_labels(self):
        """Actualiza las etiquetas de los sliders."""
        self.freq_msg_label.setText(f"Frecuencia de mensaje: {self.freq_msg.value()} Hz")
        self.freq_carrier_label.setText(f"Frecuencia de portadora: {self.freq_carrier.value()} Hz")
        self.phase_dev_label.setText(f"Desviación de fase: {self.phase_dev.value()/10.0}π rad")

    def generate_signals(self):
        """Genera las señales para la simulación de modulación."""
        try:
            t = np.linspace(0, 1, 1000)  # Vector de tiempo
            mod_type = self.selected_mod.currentText()

            # Para PCM usaremos la grabación de audio (o una señal generada si no se grabó)
            if mod_type == "PCM":
                message = self.audio_to_signal(t) if self.audio_data is not None else np.sin(2 * np.pi * self.freq_msg.value() * t)
            elif mod_type in ["ASK", "FSK", "PSK"]:
                text = self.text_input.text()
                message = self.text_to_signal(text, len(t))
            else:
                # Para AM, FM, PM se usa audio o una senoidal
                message = self.audio_to_signal(t) if self.audio_data is not None else np.sin(2 * np.pi * self.freq_msg.value() * t)

            # Asegurar que `message` tenga la misma longitud que `t`
            message = self.ensure_length(message, len(t))

            # Normalizar la señal de mensaje para PM
            if mod_type == "PM":
                message = message / np.max(np.abs(message)) if np.max(np.abs(message)) > 0 else message

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
                # Convert text to binary sequence
                binary_str = ''.join(format(ord(c), '08b') for c in text)
                # Create square wave signal
                samples_per_bit = length // len(binary_str)
                message = np.zeros(length)
                for i, bit in enumerate(binary_str):
                    start = i * samples_per_bit
                    end = (i + 1) * samples_per_bit
                    message[start:end] = int(bit)
                # Ensure we fill the entire length
                if len(message) < length:
                    message = np.pad(message, (0, length - len(message)), 'constant')
            else:
                message = np.zeros(length)
            return message
        except Exception as e:
            print(f"Error en text_to_signal: {e}")
            return np.zeros(length)

    def audio_to_signal(self, t):
        """Convierte el audio grabado en una señal analógica."""
        if self.audio_data is not None:
            return np.interp(t, np.linspace(0, 1, len(self.audio_data)), self.audio_data)
        return np.zeros_like(t)
    
    def apply_bandpass_filter(self, signal, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        return scipy.signal.filtfilt(b, a, signal)

    def apply_modulation(self, mod_type, message, carrier, t):
        """Aplica la modulación seleccionada."""
        try:
            sampling_rate = len(t)

            if mod_type == "AM":
                # --- AM Mejorada ---
                # 1) Normalizar mensaje a [-1, 1]
                if np.max(np.abs(message)) > 0:
                    msg_norm = message / np.max(np.abs(message))
                else:
                    msg_norm = message

                # 2) Parámetros
                carrier_amplitude = 1.0
                mod_index = 0.5  # Índice de modulación (0 < m < 1 para AM convencional)
                
                # 3) Ajustar longitudes
                min_length = min(len(msg_norm), len(carrier), len(t))
                msg_norm = msg_norm[:min_length]
                carrier = carrier[:min_length]
                t = t[:min_length]

                # 4) Modulación
                modulated = carrier_amplitude * (1 + mod_index * msg_norm) * carrier

                # 5) Demodulación mediante detección de envolvente
                analytic_signal = scipy.signal.hilbert(modulated)
                envelope = np.abs(analytic_signal)

                #    a) Recuperar el mensaje original
                demodulated = (envelope - carrier_amplitude) / mod_index

            elif mod_type == "FM":
                # 1) Normalizar el mensaje para asegurar variaciones
                if np.max(np.abs(message)) > 0:
                    msg_norm = message / np.max(np.abs(message))
                else:
                    msg_norm = message

                # 2) Parámetros mejorados
                freq_dev = 50.0  # Aumentar la desviación de frecuencia para mayor variabilidad
                carrier_amplitude = 1.0

                # 3) Ajustar longitud de señal
                min_length = min(len(msg_norm), len(t))
                msg_norm = msg_norm[:min_length]
                t = t[:min_length]

                # 4) Modulación FM
                dt = np.mean(np.diff(t))
                phase_deviation = 2 * np.pi * freq_dev * np.cumsum(msg_norm) * dt
                total_phase = 2 * np.pi * self.freq_carrier.value() * t + phase_deviation
                modulated = carrier_amplitude * np.cos(total_phase)

                # 5) Demodulación FM mejorada
                analytic_signal = scipy.signal.hilbert(modulated)  # Obtener señal analítica
                inst_phase = np.unwrap(np.angle(analytic_signal))  # Obtener fase instantánea
                inst_freq = np.gradient(inst_phase, dt) / (2.0 * np.pi)  # Derivar fase para obtener frecuencia instantánea

                # 6) Recuperar mensaje con menos ruido
                demodulated = (inst_freq - self.freq_carrier.value()) / freq_dev

                # 7) Aplicar filtrado suave
                b, a = scipy.signal.butter(5, 0.05, 'low')  # Filtro pasa-bajo
                demodulated_filtered = scipy.signal.filtfilt(b, a, demodulated)

                # 8) Ajustar amplitud para que coincida con el mensaje original
                demodulated_final = demodulated_filtered * (np.max(np.abs(message)) / np.max(np.abs(demodulated_filtered)))
            elif mod_type == "PM":
                # Improved PM implementation
                phase_deviation = self.phase_dev.value()/10.0 * np.pi  # Adjustable phase deviation
                modulated = np.sin(2 * np.pi * self.freq_carrier.value() * t + phase_deviation * message)
                analytic_signal = hilbert(modulated)
                demodulated = np.unwrap(np.angle(analytic_signal))
                demodulated = demodulated - 2 * np.pi * self.freq_carrier.value() * t  # Remove carrier phase
                demodulated = demodulated / phase_deviation  # Scale back to original signal
            elif mod_type == "ASK":
                # Improved ASK implementation
                high_amplitude = 1.0
                low_amplitude = 0.0
                # Create proper digital signal with sharp transitions
                digital_signal = np.where(message > 0.5, high_amplitude, low_amplitude)
                modulated = digital_signal * carrier
                # Improved demodulation using envelope detection
                demodulated = np.abs(modulated)
                # Apply threshold to recover digital signal
                threshold = (high_amplitude + low_amplitude) / 2
                demodulated = np.where(demodulated > threshold, high_amplitude, low_amplitude)
            elif mod_type == "FSK":
                f1 = self.freq_carrier.value()
                f2 = f1 + 100 

                samples_per_bit = int(len(t) / len(message))
                duration = t[-1] - t[0]
                bit_duration = duration / len(message)

                modulated = np.zeros(len(t))
                phase = 0

                for i, bit in enumerate(message):
                    start = i * samples_per_bit
                    end = (i + 1) * samples_per_bit
                    freq = f2 if bit == 1 else f1
                    t_rel = t[start:end] - t[start]
                    modulated[start:end] = np.sin(phase + 2 * np.pi * freq * t_rel)
                    phase += 2 * np.pi * freq * (t[end-1] - t[start]) + 2 * np.pi * freq * (t[1] - t[0])

                low_f1 = max(f1 - 30, 1)
                high_f1 = max(f1 + 30, low_f1 + 1)
                low_f2 = max(f2 - 30, 1)
                high_f2 = max(f2 + 30, low_f2 + 1)

                filtered_f1 = self.apply_bandpass_filter(modulated, low_f1, high_f1, sampling_rate)
                filtered_f2 = self.apply_bandpass_filter(modulated, low_f2, high_f2, sampling_rate)

                energy_f1 = np.array([np.sum(filtered_f1[i * samples_per_bit:(i + 1) * samples_per_bit]**2) for i in range(len(message))])
                energy_f2 = np.array([np.sum(filtered_f2[i * samples_per_bit:(i + 1) * samples_per_bit]**2) for i in range(len(message))])

                threshold = (np.max(energy_f1) + np.max(energy_f2)) / 2
                demodulated_bits = (energy_f2 > energy_f1).astype(int)

                demodulated = np.zeros(len(t))
                for i, bit in enumerate(demodulated_bits):
                    start = i * samples_per_bit
                    end = (i + 1) * samples_per_bit
                    demodulated[start:end] = bit

                from scipy.signal import medfilt
                demodulated = medfilt(demodulated, kernel_size=5)

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
                # ==============================
                # 1) Definir parámetros de PCM
                # ==============================
                total_length = 1000      # longitud de la señal en el tiempo de graficación
                num_levels = 16         # niveles de cuantización (4 bits)
                bits_per_sample = 4
                num_samples = 50        # número de muestras que tomaremos de la señal
                # chunk_size es el número de puntos (en el vector t) que usaremos para representar cada bit
                # con num_samples=50 y bits_per_sample=4 => total_bits=200 => chunk_size=1000/200=5
                chunk_size = total_length // (num_samples * bits_per_sample)

                # ==============================
                # 2) Muestreo de la señal
                # ==============================
                # Tomamos 50 muestras uniformemente a lo largo de 'message'
                sample_indices = np.linspace(0, len(message)-1, num_samples).astype(int)
                sampled_values = message[sample_indices]

                # Normalizar las muestras a [-1, 1]
                max_val = np.max(np.abs(sampled_values))
                if max_val > 0:
                    sampled_values = sampled_values / max_val

                # ==============================
                # 3) Cuantización
                # ==============================
                # Convertimos cada muestra a un entero entre 0 y num_levels-1
                quantized_indices = np.round((sampled_values + 1) / 2 * (num_levels - 1)).astype(int)
                quantized_indices = np.clip(quantized_indices, 0, num_levels - 1)

                # ==============================
                # 4) Codificación a bits
                # ==============================
                bit_array = []
                for q in quantized_indices:
                    # 4 bits por muestra
                    bits_str = format(q, '04b')
                    bit_array.extend([int(b) for b in bits_str])

                # ==============================
                # 5) Construir la señal PCM (digital) de longitud 1000
                #    Cada bit se representará con 'chunk_size' puntos en el tiempo
                # ==============================
                modulated_wave = np.zeros(total_length)
                for i, bit in enumerate(bit_array):
                    start = i * chunk_size
                    end = start + chunk_size
                    if end > total_length:
                        end = total_length
                    modulated_wave[start:end] = bit

                # ==============================
                # 6) Demodulación: leer cada grupo de 4 bits y reconstruir la muestra
                # ==============================
                demod_samples = []
                for i in range(num_samples):
                    # cada muestra tiene 4 bits => 4 * chunk_size puntos
                    bit_offset = i * bits_per_sample * chunk_size
                    current_bits = []
                    for b in range(bits_per_sample):
                        chunk_start = bit_offset + b * chunk_size
                        chunk_end = chunk_start + chunk_size
                        # Se toma el promedio para decidir si es 0 o 1
                        bit_val = np.mean(modulated_wave[chunk_start:chunk_end]) > 0.5
                        current_bits.append(int(bit_val))
                    # Convertir 4 bits a entero
                    index_val = current_bits[0]*8 + current_bits[1]*4 + current_bits[2]*2 + current_bits[3]
                    demod_samples.append(index_val)

                demod_samples = np.array(demod_samples)

                # Mapear de vuelta a [-1, 1]
                demod_samples = (demod_samples / (num_levels - 1)) * 2 - 1

                # ==============================
                # 7) Upsample (para graficar) esas 50 muestras a 1000 puntos
                # ==============================
                upsampled_demod = np.repeat(demod_samples, chunk_size * bits_per_sample)
                if len(upsampled_demod) < total_length:
                    upsampled_demod = np.pad(upsampled_demod, (0, total_length - len(upsampled_demod)), 'edge')
                else:
                    upsampled_demod = upsampled_demod[:total_length]

                modulated = modulated_wave
                demodulated = upsampled_demod
            else:
                modulated = demodulated = np.zeros_like(t)

            return self.ensure_length(modulated, len(t)), self.ensure_length(demodulated, len(t))

        except Exception as e:
            print(f"Error en apply_modulation: {e}")
            return np.zeros_like(t), np.zeros_like(t)

    def verify_conversion(self, message_signal, demodulated_signal):
        """Verifica la precisión de la demodulación."""
        error = np.abs(message_signal - demodulated_signal)
        print(f"Error máximo: {np.max(error)}")
        print(f"Error promedio: {np.mean(error)}")

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

        # Verificar la conversión
        self.verify_conversion(message, demodulated)

        # Mostrar espectro de la señal modulada
        self.plot_spectrum(modulated, "Espectro de la Señal Modulada")

        # Redibujar el canvas
        self.canvas.draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = ModulationSimulator()
    window.show()
    app.exec_()