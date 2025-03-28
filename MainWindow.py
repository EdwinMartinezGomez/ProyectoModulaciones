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
from src.model.SignalGenerator import SignalGenerator
from src.view.Plotter import Plotter

class MainWindow(QtWidgets.QWidget):
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

        # Inicializar componentes
        self.signal_generator = SignalGenerator()
        self.plotter = Plotter(self.layout)

        # Animación en tiempo real
        self.anim = FuncAnimation(self.plotter.fig, self.update_plot, interval=100, cache_frame_data=False, save_count=50)

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
        if mod_type == "PCM":
            self.text_input.setEnabled(False)
            self.record_button.setEnabled(True)
        else:
            is_digital = mod_type in ["ASK", "FSK", "PSK"]
            self.text_input.setEnabled(is_digital)
            self.record_button.setEnabled(not is_digital)

    def record_audio(self):
        """Graba audio de entrada."""
        self.signal_generator.audio_recorder.record_audio()

    def update_labels(self):
        """Actualiza las etiquetas de los sliders."""
        self.freq_msg_label.setText(f"Frecuencia de mensaje: {self.freq_msg.value()} Hz")
        self.freq_carrier_label.setText(f"Frecuencia de portadora: {self.freq_carrier.value()} Hz")
        self.phase_dev_label.setText(f"Desviación de fase: {self.phase_dev.value()/10.0}π rad")

    def update_plot(self, frame):
        """Actualiza las gráficas en tiempo real."""
        t = np.linspace(0, 1, 1000)  # Vector de tiempo
        mod_type = self.selected_mod.currentText()
        freq_msg = self.freq_msg.value()
        freq_carrier = self.freq_carrier.value()

        # Generar señales
        t, message, carrier, modulated, demodulated = self.signal_generator.generate_signals(mod_type, self.text_input.text(), freq_carrier, freq_msg, t)

        # Actualizar gráficos
        self.plotter.update_plot(t, message, carrier, modulated, demodulated)
        self.plotter.plot_spectrum(modulated, "Espectro de la Señal Modulada")

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()