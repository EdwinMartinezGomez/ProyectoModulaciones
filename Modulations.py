import numpy as np
import matplotlib.pyplot as plt
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation
import sounddevice as sd

class ModulationSimulator(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulación de Modulación")
        
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.mod_types = ["AM", "FM", "PM", "ASK", "FSK", "PSK", "PCM"]
        self.selected_mod = QtWidgets.QComboBox()
        self.selected_mod.addItems(self.mod_types)
        self.selected_mod.currentTextChanged.connect(self.toggle_input_method)
        self.layout.addWidget(QtWidgets.QLabel("Seleccione el tipo de modulación:"))
        self.layout.addWidget(self.selected_mod)
        
        self.input_label = QtWidgets.QLabel("Ingrese un mensaje de texto para datos digitales o grabe un audio para datos analógicos")
        self.layout.addWidget(self.input_label)
        
        self.text_input = QtWidgets.QLineEdit()
        self.layout.addWidget(self.text_input)
        
        self.record_button = QtWidgets.QPushButton("Grabar Audio")
        self.record_button.clicked.connect(self.record_audio)
        self.layout.addWidget(self.record_button)
        
        self.freq_msg = QtWidgets.QSlider(Qt.Horizontal)
        self.freq_msg.setMinimum(1)
        self.freq_msg.setMaximum(20)
        self.freq_msg.setValue(5)
        self.freq_msg.valueChanged.connect(self.update_labels)
        self.freq_msg_label = QtWidgets.QLabel(f"Frecuencia de mensaje: {self.freq_msg.value()} Hz")
        self.layout.addWidget(self.freq_msg_label)
        self.layout.addWidget(self.freq_msg)
        
        self.freq_carrier = QtWidgets.QSlider(Qt.Horizontal)
        self.freq_carrier.setMinimum(5)
        self.freq_carrier.setMaximum(50)
        self.freq_carrier.setValue(20)
        self.freq_carrier.valueChanged.connect(self.update_labels)
        self.freq_carrier_label = QtWidgets.QLabel(f"Frecuencia de portadora: {self.freq_carrier.value()} Hz")
        self.layout.addWidget(self.freq_carrier_label)
        self.layout.addWidget(self.freq_carrier)
        
        self.fig, self.axs = plt.subplots(4, 1, figsize=(6, 8))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        
        self.audio_data = None
        
        self.anim = FuncAnimation(self.fig, self.update_plot, interval=100)
        self.toggle_input_method()
    
    def toggle_input_method(self):
        mod_type = self.selected_mod.currentText()
        if mod_type in ["ASK", "FSK", "PSK", "PCM"]:
            self.text_input.setEnabled(True)
            self.record_button.setEnabled(False)
        else:
            self.text_input.setEnabled(False)
            self.record_button.setEnabled(True)
    
    def record_audio(self):
        duration = 2  # segundos
        self.audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
        sd.wait()
        self.audio_data = self.audio_data.flatten()
    
    def update_labels(self):
        self.freq_msg_label.setText(f"Frecuencia de mensaje: {self.freq_msg.value()} Hz")
        self.freq_carrier_label.setText(f"Frecuencia de portadora: {self.freq_carrier.value()} Hz")
        
    def generate_signals(self):
        t = np.linspace(0, 1, 1000)
        mod_type = self.selected_mod.currentText()
        
        if mod_type in ["ASK", "FSK", "PSK", "PCM"]:
            text = self.text_input.text()
            if text:
                message = np.array([int(bit) for bit in ''.join(format(ord(c), '08b') for c in text)])
                message = np.repeat(message, int(np.ceil(len(t) / len(message))))[:len(t)]  # Ajusta la longitud
            else:
                message = np.zeros_like(t)
        else:
            if self.audio_data is not None and len(self.audio_data) > 0:
                message = np.interp(t, np.linspace(0, 1, len(self.audio_data)), self.audio_data)
            else:
                message = np.sin(2 * np.pi * self.freq_msg.value() * t)
        
        carrier = np.sin(2 * np.pi * self.freq_carrier.value() * t)
        
        if mod_type == "AM":
            modulated = (1 + message) * carrier
            demodulated = np.abs(modulated) - 1
        elif mod_type == "FM":
            modulated = np.sin(2 * np.pi * self.freq_carrier.value() * t + 2 * np.pi * message)
            demodulated = np.gradient(np.unwrap(np.angle(modulated)))
        elif mod_type == "PM":
            modulated = np.sin(2 * np.pi * self.freq_carrier.value() * t + np.pi * message)
            demodulated = np.unwrap(np.angle(modulated))
        elif mod_type == "ASK":
            modulated = (message > 0) * carrier
            demodulated = (modulated > 0).astype(float)
        elif mod_type == "FSK":
            modulated = np.sin(2 * np.pi * (self.freq_carrier.value() + message * 10) * t)
            demodulated = np.gradient(np.unwrap(np.angle(modulated)))
        elif mod_type == "PSK":
            modulated = np.sin(2 * np.pi * self.freq_carrier.value() * t + np.pi * (message > 0))
            demodulated = np.unwrap(np.angle(modulated))
        elif mod_type == "PCM":
            modulated = np.round(message * 2) / 2
            demodulated = modulated
        else:
            modulated = np.zeros_like(t)
            demodulated = np.zeros_like(t)
        
        return t, message, carrier, modulated, demodulated
    
    def update_plot(self, frame):
        t, message, carrier, modulated, demodulated = self.generate_signals()
        
        for ax in self.axs:
            ax.cla()
        
        self.axs[0].plot(t, message, 'g')
        self.axs[0].set_title("Señal del Mensaje")
        self.axs[0].grid()
        
        self.axs[1].plot(t, carrier, 'b')
        self.axs[1].set_title("Señal Portadora")
        self.axs[1].grid()
        
        self.axs[2].plot(t, modulated, 'r')
        self.axs[2].set_title(f"Señal Modulada ({self.selected_mod.currentText()})")
        self.axs[2].grid()
        
        self.axs[3].plot(t, demodulated, 'm')
        self.axs[3].set_title(f"Señal Demodulada ({self.selected_mod.currentText()})")
        self.axs[3].grid()
        
        self.canvas.draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = ModulationSimulator()
    window.show()
    app.exec_()
