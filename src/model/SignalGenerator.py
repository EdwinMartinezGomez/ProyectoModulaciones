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
from src.util.AudioRecorder import AudioRecorder

class SignalGenerator:
    def __init__(self):
        self.audio_recorder = AudioRecorder()

    def generate_signals(self, mod_type, message, freq_carrier, freq_msg, t):
        """Genera las señales para la simulación de modulación."""
        try:
            t = np.linspace(0, 1, 1000)  # Vector de tiempo

            # Para PCM usaremos la grabación de audio (o una señal generada si no se grabó)
            if mod_type == "PCM":
                message = self.audio_recorder.audio_to_signal(t) if self.audio_recorder.audio_data is not None else np.sin(2 * np.pi * freq_msg * t)
            elif mod_type in ["ASK", "FSK", "PSK"]:
                text = message
                message = self.text_to_signal(text, len(t))
            else:
                # Para AM, FM, PM se usa audio o una senoidal
                message = self.audio_recorder.audio_to_signal(t) if self.audio_recorder.audio_data is not None else np.sin(2 * np.pi * freq_msg * t)

            # Asegurar que `message` tenga la misma longitud que `t`
            message = self.ensure_length(message, len(t))

            # Normalizar la señal de mensaje para PM
            if mod_type == "PM":
                message = message / np.max(np.abs(message)) if np.max(np.abs(message)) > 0 else message

            # Generar la portadora
            carrier = np.sin(2 * np.pi * freq_carrier * t)

            # Aplicar modulación
            modulated, demodulated = self.apply_modulation(mod_type, message, carrier, t, freq_carrier)

            return t, message, carrier, modulated, demodulated

        except Exception as e:
            print(f"Error en generate_signals: {e}")
            return t, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
        
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

    def ensure_length(self, signal, length):
        """Ajusta la señal para que tenga exactamente la longitud deseada."""
        return np.resize(signal, length)
    
    def apply_bandpass_filter(self, signal, lowcut, highcut, fs, order=4):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = scipy.signal.butter(order, [low, high], btype='band')
            return scipy.signal.filtfilt(b, a, signal)
    def apply_modulation(self, mod_type, message, carrier, t, freq_carrier, phase_dev=1):
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
                total_phase = 2 * np.pi * freq_carrier * t + phase_deviation
                modulated = carrier_amplitude * np.cos(total_phase)

                # 5) Demodulación FM mejorada
                analytic_signal = scipy.signal.hilbert(modulated)  # Obtener señal analítica
                inst_phase = np.unwrap(np.angle(analytic_signal))  # Obtener fase instantánea
                inst_freq = np.gradient(inst_phase, dt) / (2.0 * np.pi)  # Derivar fase para obtener frecuencia instantánea

                # 6) Recuperar mensaje con menos ruido
                demodulated = (inst_freq - freq_carrier) / freq_dev

                # 7) Aplicar filtrado suave
                b, a = scipy.signal.butter(5, 0.05, 'low')  # Filtro pasa-bajo
                demodulated_filtered = scipy.signal.filtfilt(b, a, demodulated)

                # 8) Ajustar amplitud para que coincida con el mensaje original
                demodulated_final = demodulated_filtered * (np.max(np.abs(message)) / np.max(np.abs(demodulated_filtered)))
            elif mod_type == "PM":
                # Improved PM implementation
                phase_deviation = phase_dev/10.0 * np.pi  # Adjustable phase deviation
                modulated = np.sin(2 * np.pi * freq_carrier * t + phase_deviation * message)
                analytic_signal = hilbert(modulated)
                demodulated = np.unwrap(np.angle(analytic_signal))
                demodulated = demodulated - 2 * np.pi * freq_carrier * t  # Remove carrier phase
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
                f1 = freq_carrier
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
                modulated = np.sin(2 * np.pi * freq_carrier * t + np.pi * message)

                # Demodulación coherente
                carrier_ref = np.sin(2 * np.pi * freq_carrier * t)
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