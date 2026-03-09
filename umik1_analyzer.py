import sys
import time
import threading
from collections import deque
import numpy as np
import sounddevice as sd
if sys.platform == 'win32':
    try:
        import winsound as _winsound
    except ImportError:
        _winsound = None
else:
    _winsound = None
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QCheckBox,
                             QSlider, QGroupBox, QFormLayout, QPushButton, QFileDialog, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QEvent
import pyqtgraph as pg

# Patch items that crash PyqtGraph context menu
for cls in [pg.ScatterPlotItem, pg.TextItem, pg.LinearRegionItem, pg.ImageItem, pg.InfiniteLine]:
    if not hasattr(cls, 'setFftMode'):
        cls.setFftMode = lambda self, state: None

# ==========================================
# Custom Log Axis for UI
# ==========================================
class LogFreqAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        strings = []
        for val in values:
            try:
                real_val = 10 ** val
                if real_val >= 10:
                    strings.append(f"{int(round(real_val))}")
                else:
                    strings.append(f"{real_val:.1f}")
            except OverflowError:
                strings.append("")
        return strings

# ==========================================
# Custom ViewBox for Separate Pan/Zoom
# ==========================================
class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.setMouseMode(self.PanMode)
        self._zoom_axis = None

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.RightButton:
            ev.accept()
            if ev.isStart():
                self._zoom_axis = None
                
            if ev.isFinish():
                self.setCursor(Qt.ArrowCursor)
                self._zoom_axis = None
                return
                
            delta = ev.pos() - ev.lastPos()
            dx = delta.x()
            dy = delta.y()
            
            if self._zoom_axis is None and (abs(dx) > 3 or abs(dy) > 3):
                if abs(dx) > abs(dy):
                    self._zoom_axis = 'x'
                    self.setCursor(Qt.SizeHorCursor)
                else:
                    self._zoom_axis = 'y'
                    self.setCursor(Qt.SizeVerCursor)
                    
            x_scale = 1.0
            y_scale = 1.0
            if self._zoom_axis == 'x':
                x_scale = 0.99 ** dx
            elif self._zoom_axis == 'y':
                y_scale = 0.99 ** -dy
                
            self.scaleBy(x=x_scale, y=y_scale)

        elif ev.button() == Qt.LeftButton:
            if ev.isStart():
                self.setCursor(Qt.SizeAllCursor)
            if ev.isFinish():
                self.setCursor(Qt.ArrowCursor)
            super().mouseDragEvent(ev, axis)
        else:
            super().mouseDragEvent(ev, axis)

# ==========================================
# DSP Constants and Weightings
# ==========================================
def a_weighting(f):
    if f == 0: return -np.inf
    f2 = f**2
    ra = (12194**2 * f2**2) / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2)*(f2 + 737.9**2)) * (f2 + 12194**2))
    return 20 * np.log10(ra) + 2.00

def c_weighting(f):
    if f == 0: return -np.inf
    f2 = f**2
    rc = (12194**2 * f2) / ((f2 + 20.6**2) * (f2 + 12194**2))
    return 20 * np.log10(rc) + 0.06

# ==========================================
# Audio Processing Engine
# ==========================================
class AudioEngine(QObject):
    # freqs, spectrum_spl, lz, la, lc, l_band, is_clipping, oct_freqs, oct_spl
    data_ready = pyqtSignal(object, object, float, float, float, float, bool, object, object)
    
    def __init__(self):
        super().__init__()
        self.stream = None
        
        # Audio Settings
        self.sample_rate = 48000
        self.block_size = 65536  # Default N
        self.device_id = None
        self.subtract_dc = True
        self.smoothing_factor = 0.3
        self.window_type = 'blackman-harris'
        
        # Calibration Settings
        self.cal_file = ""
        self.sens_factor = 0.0
        self.base_offset = 130.0 # Standard UMIK digital offset
        self.cal_freqs = []
        self.cal_mags = []
        self.cal_interp = None
        
        # Internal State
        self.window = np.ones(self.block_size)
        self.freqs = np.zeros(self.block_size // 2 + 1)
        self.a_weights = np.zeros_like(self.freqs)
        self.c_weights = np.zeros_like(self.freqs)
        self.cal_weights = np.zeros_like(self.freqs)
        self.smoothed_power = None
        
        # Async Processing State
        self.audio_queue = deque()
        self.audio_buffer = np.zeros(self.block_size)
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self.process_queue)
        
        # Band Integration State
        self.band_min = 1.0
        self.band_max = 20.0
        
        # IEC 1/3 Octave Standard Center Frequencies
        # From ~10Hz up to 20kHz
        self.oct_centers = np.array([
            10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 
            250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 
            4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
        ], dtype=np.float32)
        
        # Calculate standardized band edges: f_lower = f_center / (2^(1/6)), f_upper = f_center * (2^(1/6))
        half_step = 2.0 ** (1.0 / 6.0)
        self.oct_lower = self.oct_centers / half_step
        self.oct_upper = self.oct_centers * half_step
        
        self._update_internal_arrays()

    def parse_calibration(self, filepath):
        self.cal_file = filepath
        sens = 0.0
        freqs, mags = [], []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if "Sens Factor" in line:
                        parts = line.split('=')
                        if len(parts) > 1:
                            sens = float(parts[1].split('dB')[0])
                    elif line[0].isdigit():
                        parts = line.split()
                        if len(parts) == 2:
                            freqs.append(float(parts[0]))
                            mags.append(float(parts[1]))
            if freqs:
                self.sens_factor = sens
                self.cal_freqs = np.array(freqs)
                self.cal_mags = np.array(mags)
                self.cal_interp = interp1d(self.cal_freqs, self.cal_mags, bounds_error=False, fill_value="extrapolate")
                self._update_internal_arrays()
                return True, f"Loaded UMIK-1 Sens Factor: {self.sens_factor} dB"
        except Exception as e:
            return False, str(e)
        return False, "Invalid format"

    def set_block_size(self, size):
        self.block_size = size
        self.smoothed_power = None
        self.audio_queue = deque()
        self.audio_buffer = np.zeros(self.block_size)
        self._update_internal_arrays()
        if self.stream is not None:
            self.restart_stream()
        
    def set_window(self, win_type):
        self.window_type = win_type
        self._update_internal_arrays()
        
    def set_band(self, fmin, fmax):
        self.band_min = fmin
        self.band_max = fmax

    def _update_internal_arrays(self):
        # Window function
        if self.window_type == 'hanning':
            self.window = np.hanning(self.block_size)
        elif self.window_type == 'blackman-harris':
            self.window = np.blackman(self.block_size) # Using blackman as fallback in numpy
        else:
            self.window = np.ones(self.block_size)
            
        self.freqs = np.fft.rfftfreq(self.block_size, 1.0 / self.sample_rate)
        
        # Cache weights
        self.a_weights = np.array([a_weighting(f) for f in self.freqs])
        self.c_weights = np.array([c_weighting(f) for f in self.freqs])
        
        if self.cal_interp is not None:
            self.cal_weights = self.cal_interp(self.freqs)
        else:
            self.cal_weights = np.zeros_like(self.freqs)

    def audio_callback(self, indata, frames, time, status):
        try:
            if status and 'input overflow' not in str(status):
                print(f"Audio status: {status}")
            # Fast non-blocking queue push; indata may be overwritten after return
            self.audio_queue.append(indata[:, 0].copy())
        except Exception as e:
            print(f"Exception in audio_callback: {e}")

    def process_queue(self):
        # Stream watchdog: restart if the stream has died silently
        if self.stream is not None and not self.stream.active:
            now = time.time()
            if now - getattr(self, '_last_watchdog_restart', 0) > 5.0:
                self._last_watchdog_restart = now
                print("[Watchdog] Audio stream inactive — restarting...")
                try:
                    self.restart_stream()
                except Exception as e:
                    print(f"[Watchdog] Restart failed: {e}")
                return

        if not self.audio_queue:
            return
            
        # Drain the current queue and concatenate
        chunks = []
        while self.audio_queue:
            chunks.append(self.audio_queue.popleft())
        new_data = np.concatenate(chunks)
        
        # If queue accumulated more than our window size, truncate to newest data
        if len(new_data) > self.block_size:
            new_data = new_data[-self.block_size:]
        
        # Roll buffer
        self.audio_buffer = np.roll(self.audio_buffer, -len(new_data))
        self.audio_buffer[-len(new_data):] = new_data
        
        is_clipping = np.max(np.abs(new_data)) > 0.99
        
        # Now do the heavy FFT math on the fixed size internal rolling buffer
        audio_data = self.audio_buffer.copy()
        
        if self.subtract_dc:
            audio_data = audio_data - np.mean(audio_data)
            
        windowed_data = audio_data * self.window
        fft_result = np.fft.rfft(windowed_data)
        
        # Rigorous window compensation: 
        # For amplitude of a sine wave, we divide by sum(window). 
        # We multiply by 2 because rfft only returns the positive half of the spectrum.
        mag = np.abs(fft_result) / (np.sum(self.window) / 2.0)
        mag[0] /= 2.0 # DC bin doesn't need the x2 compensation
        mag = np.maximum(mag, 1e-12)
        
        dbfs = 20 * np.log10(mag)
        
        inst_spl = dbfs + (self.base_offset + self.sens_factor) + self.cal_weights
        inst_power = 10 ** (inst_spl / 10.0)
        
        if self.smoothed_power is None or len(self.smoothed_power) != len(inst_power):
            self.smoothed_power = inst_power
        else:
            # Here alpha represents how much NEW data is used. 
            # 1.0 means instantaneous, 0.001 means very heavy smoothing
            alpha = self.smoothing_factor
            self.smoothed_power = alpha * inst_power + (1 - alpha) * self.smoothed_power
            
        spectrum_spl = 10 * np.log10(np.maximum(self.smoothed_power, 1e-12))
        
        z_power = np.sum(self.smoothed_power)
        power_a = 10 ** ((spectrum_spl + self.a_weights) / 10.0)
        power_c = 10 ** ((spectrum_spl + self.c_weights) / 10.0)
        
        band_mask = (self.freqs >= self.band_min) & (self.freqs <= self.band_max)
        band_power = np.sum(self.smoothed_power[band_mask]) if np.any(band_mask) else 0.0

        lz = 10 * np.log10(z_power) if z_power > 0 else 0
        la = 10 * np.log10(np.sum(power_a)) if np.sum(power_a) > 0 else 0
        lc = 10 * np.log10(np.sum(power_c)) if np.sum(power_c) > 0 else 0
        l_band = 10 * np.log10(band_power) if band_power > 0 else 0
        
        # 1/3 Octave Binning
        oct_spl = np.zeros_like(self.oct_centers)
        for i in range(len(self.oct_centers)):
            f_low = self.oct_lower[i]
            f_high = self.oct_upper[i]
            # Find indices in the FFT freqs that fall into this 1/3 octave band
            bin_mask = (self.freqs >= f_low) & (self.freqs < f_high)
            bin_power = np.sum(self.smoothed_power[bin_mask]) if np.any(bin_mask) else 0.0
            oct_spl[i] = 10 * np.log10(bin_power) if bin_power > 1e-12 else -20.0
            
        self.data_ready.emit(self.freqs, spectrum_spl, float(lz), float(la), float(lc), float(l_band), bool(is_clipping), self.oct_centers, oct_spl)

    def start_stream(self):
        try:
            # We now read faster (smaller chunks) from sounddevice to give low latency
            # while overlapping the big FFT window on the rolling buffer.
            # Using 4096 framing from mic keeps latency fast
            self.stream = sd.InputStream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=1,
                blocksize=min(4096, self.block_size),
                latency='high',
                callback=self.audio_callback
            )
            self.stream.start()
            
            if self.block_size <= 65536:
                interval = 30
            elif self.block_size <= 131072:
                interval = 100
            else:
                interval = 250
            # Start process timer dynamically based on FFT load
            self.process_timer.start(interval)
        except Exception as e:
            print(f"Error starting audio stream: {e}")

    def stop_stream(self):
        self.process_timer.stop()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
    def restart_stream(self):
        self.stop_stream()
        self.start_stream()


# ==========================================
# Main GUI Window
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = None
        self.max_hold_trace = None
        
        self.waterfall_history = 200 # rows in waterfall
        # [Freq_bins, Time_history] buffer
        self.waterfall_data = None 
        
        self.waterfall_skip_counter = 0
        self.current_crosshair_logx = None
        
        self.current_lang = 'ZH'
        
        # Advanced Features State
        self.spl_history = np.full(100, -20.0) # 100 frames history
        self.primary_spl_min = 999.0
        self.primary_spl_max = -999.0
        self.alarm_active = False
        self.peak_texts = []
        
        try:
            print("Init UI...")
            self.init_ui()
            print("Init Engine...")
            self.init_engine()
            print("Init Complete")
        except Exception as e:
            print(f"Error in MainWindow init: {e}")
        
    def init_ui(self):
        self.setWindowTitle("UMIK-1 Spectrum Analyzer / UMIK-1 频谱分析仪")
        self.resize(1300, 850)
        self.setStyleSheet("background-color: #0d0d0d; color: #e0e0e0;")
        
        # Main Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left Panel - Plots
        plot_layout = QVBoxLayout()
        main_layout.addLayout(plot_layout, stretch=3)
        
        # Utility Buttons Row (Above plots)
        top_btns = QHBoxLayout()
        self.btn_lang = QPushButton("EN / 中文")
        self.btn_lang.setStyleSheet("background-color: #333; color: white; padding: 5px; font-weight: bold;")
        self.btn_lang.setFixedSize(120, 30)
        self.btn_lang.clicked.connect(self.toggle_lang)
        
        self.btn_octave = QPushButton("1/3 Octave")
        self.btn_octave.setStyleSheet("background-color: #673AB7; color: white; padding: 5px; font-weight: bold;")
        self.btn_octave.setCheckable(True)
        self.btn_octave.setFixedSize(150, 30) # Increased to prevent EN clipping
        self.btn_octave.clicked.connect(self.toggle_octave)
        
        self.btn_hide = QPushButton("Hide Menu ➡")
        self.btn_hide.setStyleSheet("background-color: #C2185B; color: white; padding: 5px; font-weight: bold;")
        self.btn_hide.setFixedSize(150, 30)
        self.btn_hide.clicked.connect(self.toggle_menu)
        
        self.btn_screenshot = QPushButton("截图 (Screenshot)")
        self.btn_screenshot.setStyleSheet("background-color: #2196F3; color: white; padding: 5px; font-weight: bold;")
        self.btn_screenshot.clicked.connect(self.take_screenshot)
        
        self.btn_record = QPushButton("⏺ 记录数据 (Record)")
        self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px; font-weight: bold;")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.toggle_recording)
        
        self.spin_record_time = QSpinBox()
        self.spin_record_time.setPrefix("时长 (s): ")
        self.spin_record_time.setSpecialValueText("不限时 (Infinite)")
        self.spin_record_time.setRange(0, 36000)
        self.spin_record_time.setValue(0)
        self.spin_record_time.setFixedWidth(130)
        
        self.is_recording = False
        self.record_file = None
        self.record_start_time = 0
        self.record_duration = 0
        self.record_data_dict = {}
        
        top_btns.addWidget(self.btn_lang)
        top_btns.addWidget(self.btn_octave)
        top_btns.addWidget(self.btn_screenshot)
        top_btns.addWidget(self.spin_record_time)
        top_btns.addWidget(self.btn_record)
        top_btns.addStretch(1)
        top_btns.addWidget(self.btn_hide)
        plot_layout.addLayout(top_btns)
        
        # 1. Spectrum Plot
        self.pw_spectrum = pg.PlotWidget(title="实时频谱 (极大值保持 & 智能峰值检测)", viewBox=CustomViewBox(), axisItems={'bottom': LogFreqAxis(orientation='bottom')})
        self.btn_overlay_spec = QPushButton("Hide", self.pw_spectrum)
        self.btn_overlay_spec.setStyleSheet("background-color: rgba(33, 150, 243, 200); color: white; padding: 5px; font-weight: bold; border-radius: 3px;")
        self.btn_overlay_spec.move(10, 10)
        self.btn_overlay_spec.hide()
        self.btn_overlay_spec.clicked.connect(self.toggle_spectrum)
        self.pw_spectrum.installEventFilter(self)
        self.pw_spectrum.showGrid(x=True, y=True, alpha=0.3)
        self.pw_spectrum.setLabel('bottom', '频率 (Frequency)', units='Hz')
        self.pw_spectrum.setLabel('left', '幅值 (Amplitude)', units='dB SPL')
        self.pw_spectrum.setYRange(0, 130)
        self.pw_spectrum.setLimits(xMin=np.log10(1.0), xMax=np.log10(24000.0), yMin=-20, yMax=180) # Bound to 24kHz
        self.pw_spectrum.setXRange(np.log10(1.0), np.log10(200.0)) # 1Hz to 200Hz default
        
        self.curve_realtime = self.pw_spectrum.plot(pen=pg.mkPen('cyan', width=1.5), name="Realtime") # Changed to Cyan for pro look
        self.curve_maxhold = self.pw_spectrum.plot(pen=pg.mkPen('y', width=1.0, style=Qt.PenStyle.DashLine), name="Max Hold")
        
        # 1/3 Octave Bar Graph
        # We use a very specialized Log10 X-axis brush to simulate thick IEC standard columns
        self.bar_octave = pg.BarGraphItem(x=[0], height=[0], width=0.08, brush=pg.mkBrush(200, 200, 255, 180))
        self.pw_spectrum.addItem(self.bar_octave)
        self.bar_octave.hide()
        
        # Band Selection Region
        self.region = pg.LinearRegionItem([np.log10(2.0), np.log10(10.0)])
        self.region.setZValue(10)
        # Make boundary handles thicker so they are easier to grab
        for _line in self.region.lines:
            _line.setPen(pg.mkPen('#FFC107', width=3))
            _line.setHoverPen(pg.mkPen('white', width=6))
        self.pw_spectrum.addItem(self.region)
        self.region.sigRegionChanged.connect(self.update_band_selection)
        
        plot_layout.addWidget(self.pw_spectrum, stretch=1)
        
        # 2. Waterfall Plot
        self.pw_waterfall = pg.PlotWidget(title="历史信息瀑布图 (Waterfall Spectrogram)", axisItems={'bottom': LogFreqAxis(orientation='bottom')})
        self.btn_overlay_wf = QPushButton("Hide", self.pw_waterfall)
        self.btn_overlay_wf.setStyleSheet("background-color: rgba(33, 150, 243, 200); color: white; padding: 5px; font-weight: bold; border-radius: 3px;")
        self.btn_overlay_wf.move(10, 10)
        self.btn_overlay_wf.hide()
        self.btn_overlay_wf.clicked.connect(self.toggle_waterfall)
        self.pw_waterfall.installEventFilter(self)
        self.pw_waterfall.setLabel('bottom', '频率 (Frequency)', units='Hz')
        self.pw_waterfall.setLabel('left', '时间历史 (Time)')
        # Lock waterfall layout
        self.pw_waterfall.setMouseEnabled(x=False, y=False)
        self.pw_waterfall.hideButtons()
        
        self.img_waterfall = pg.ImageItem()
        self.pw_waterfall.addItem(self.img_waterfall)
        
        # We manually set X-axis ticks to look Logarithmic
        # But the plot space is strictly linear log10 units
        self.pw_waterfall.setXRange(np.log10(1.0), np.log10(100.0))
        
        # Colormap for Waterfall (Inferno/Viridis style)
        colormap = pg.colormap.get('inferno')
        self.img_waterfall.setLookupTable(colormap.getLookupTable())
        self.img_waterfall.setLevels([0, 130]) # Default dynamic range
        
        plot_layout.addWidget(self.pw_waterfall, stretch=1)
        
        # ScatterPlot for Peak Detection Auto-Labels
        self.peak_scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 200))
        self.pw_spectrum.addItem(self.peak_scatter)
        for i in range(10): # Prepare up to 10 peak labels
            text_item = pg.TextItem(text="", color="white", fill=(255,0,0,100), anchor=(0.5, 1.2))
            self.pw_spectrum.addItem(text_item)
            text_item.hide()
            self.peak_texts.append(text_item)
        
        # Match X-Axis zooming between Spectrum and Waterfall
        self.pw_spectrum.setXLink(self.pw_waterfall)
        
        # Match Spectrum Y-Axis to Waterfall Color Levels (Gain)
        self.pw_spectrum.getViewBox().sigYRangeChanged.connect(self.update_waterfall_gain)
        
        # Interactive Crosshair Lines and Text Marker
        self.vLine_spec = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('blue', width=2, style=Qt.PenStyle.DashLine))
        self.vLine_wf = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('blue', width=2, style=Qt.PenStyle.DashLine))
        self.vLine_spec.hide()
        self.vLine_wf.hide()
        
        self.crosshair_text = pg.TextItem(text="", color="white", fill=(0, 0, 255, 120), anchor=(0, 1))
        self.crosshair_text.hide()
        
        self.pw_spectrum.addItem(self.vLine_spec)
        self.pw_spectrum.addItem(self.crosshair_text)
        self.pw_waterfall.addItem(self.vLine_wf)
        
        self.pw_spectrum.scene().sigMouseClicked.connect(self.on_plot_clicked)
        self.pw_waterfall.scene().sigMouseClicked.connect(self.on_plot_clicked)
        
        # Right Panel - Settings Container (collapsible)
        self.settings_widget = QWidget()
        settings_layout = QVBoxLayout(self.settings_widget)
        main_layout.addWidget(self.settings_widget, stretch=1)
        
        # Readouts (Professional Style)
        readouts_layout = QFormLayout()
        self.combo_primary = QComboBox()
        self.combo_primary.addItems(["Z-Weighting (Unweighted)", "A-Weighting (Human Ear)", "C-Weighting (Loud)"])
        self.combo_primary.currentIndexChanged.connect(self.reset_minmax)
        readouts_layout.addRow("Primary SPL Unit:", self.combo_primary)
        settings_layout.addLayout(readouts_layout)
        
        self.lbl_lz = QLabel("Total SPL\n-- dB")
        self.lbl_lz.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_lz.setStyleSheet("font-family: 'Courier New', monospace; font-size: 28px; font-weight: bold; color: #00FF00; background-color: #111; border: 2px solid #333; border-radius: 5px; padding: 10px;")
        
        self.lbl_overload = QLabel("OVERLOAD / 过载")
        self.lbl_overload.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_overload.setStyleSheet("font-family: 'Courier New', monospace; font-size: 20px; font-weight: bold; color: #555; background-color: #111; border: 2px solid #333; border-radius: 5px; padding: 5px;")
        
        minmax_layout = QHBoxLayout()
        self.lbl_minmax = QLabel("Min: -- | Max: --")
        self.lbl_minmax.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_minmax.setStyleSheet("color: #e0e0e0; font-size: 20px; font-weight: bold;")
        
        self.btn_reset_minmax = QPushButton("Reset / 重置")
        self.btn_reset_minmax.setStyleSheet("background-color: #555; color: white; padding: 5px; font-size: 14px;")
        self.btn_reset_minmax.clicked.connect(self.reset_minmax)
        
        minmax_layout.addWidget(self.lbl_minmax, stretch=3)
        minmax_layout.addWidget(self.btn_reset_minmax, stretch=1)
        
        self.lbl_band = QLabel("Band SPL\n-- dB")
        self.lbl_band.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_band.setStyleSheet("font-family: 'Courier New', monospace; font-size: 24px; font-weight: bold; color: #FFC107; background-color: #111; border: 2px solid #333; border-radius: 5px; padding: 10px;")
        
        self.lbl_la = QLabel("LAeq: -- dB(A)")
        self.lbl_la.setStyleSheet("font-family: 'Courier New', monospace; font-size: 16px; color: #FF9800;")
        self.lbl_lc = QLabel("LCeq: -- dB(C)")
        self.lbl_lc.setStyleSheet("font-family: 'Courier New', monospace; font-size: 16px; color: #03A9F4;")
        
        self.lbl_crosshair_readout = QLabel("Marker:\n-- Hz | -- dB")
        self.lbl_crosshair_readout.setStyleSheet("font-size: 18px; color: #448AFF; font-weight: bold;")
        
        settings_layout.addWidget(self.lbl_lz)
        settings_layout.addWidget(self.lbl_overload)
        settings_layout.addLayout(minmax_layout)
        settings_layout.addWidget(self.lbl_band)
        settings_layout.addWidget(self.lbl_la)
        settings_layout.addWidget(self.lbl_lc)
        settings_layout.addSpacing(5)
        
        self.pw_history = pg.PlotWidget(title="SPL History")
        self.pw_history.setFixedHeight(200)
        self.pw_history.setMouseEnabled(x=False, y=False)
        self.pw_history.hideButtons()
        self.pw_history.showGrid(y=True, alpha=0.3)
        self.pw_history.setLabel('bottom', 'Past Frames', units='')
        
        self.curve_history = self.pw_history.plot(pen=pg.mkPen('g', width=2))
        self.line_hist_max = self.pw_history.addLine(y=-999, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine))
        self.line_hist_min = self.pw_history.addLine(y=999, pen=pg.mkPen('b', style=Qt.PenStyle.DashLine))
        
        settings_layout.addWidget(self.pw_history)
        
        settings_layout.addWidget(self.lbl_crosshair_readout)
        settings_layout.addSpacing(5)
        
        # Calibration Settings
        box_cal = QGroupBox("UMIK-1 Calibration")
        layout_cal = QVBoxLayout()
        self.btn_load_cal = QPushButton("Load Calibration File...")
        self.btn_load_cal.clicked.connect(self.load_cal_file)
        self.lbl_cal_status = QLabel("Calibration: Uncalibrated")
        self.lbl_cal_status.setWordWrap(True)
        layout_cal.addWidget(self.btn_load_cal)
        layout_cal.addWidget(self.lbl_cal_status)
        box_cal.setLayout(layout_cal)
        settings_layout.addWidget(box_cal)
        
        # Audio & DSP Settings Form
        box_dsp = QGroupBox("Audio & Processing")
        form_dsp = QFormLayout()
        
        self.combo_device = QComboBox()
        self.populate_audio_devices()
        self.combo_device.currentIndexChanged.connect(self.change_device)
        form_dsp.addRow("Audio Input:", self.combo_device)
        
        self.combo_fft = QComboBox()
        # Removed 512, 1024 as they introduce high minimum bin frequencies
        fft_sizes = ["2048", "4096", "8192", "16384", "32768", "65536", "131072", "262144"]
        self.combo_fft.addItems(fft_sizes)
        self.combo_fft.setCurrentText("65536") # Good default balance
        self.combo_fft.currentIndexChanged.connect(self.change_fft_size)
        self.lbl_fft_param = QLabel("FFT Size / 采样大小:")
        form_dsp.addRow(self.lbl_fft_param, self.combo_fft)
        
        self.combo_win = QComboBox()
        self.combo_win.addItems(["blackman-harris", "hanning", "rectangular"])
        self.combo_win.currentIndexChanged.connect(self.change_window)
        form_dsp.addRow("Window:", self.combo_win)
        
        self.chk_dc = QCheckBox("Subtract DC")
        self.chk_dc.setChecked(True)
        self.chk_dc.stateChanged.connect(self.change_dc)
        form_dsp.addRow("", self.chk_dc)
        
        self.chk_maxhold = QCheckBox("Max-Hold Trace")
        self.chk_maxhold.setChecked(True)
        self.chk_maxhold.stateChanged.connect(self.reset_max_hold)
        form_dsp.addRow("", self.chk_maxhold)
        
        self.slider_smooth = QSlider(Qt.Orientation.Horizontal)
        self.slider_smooth.setRange(1, 99)
        self.slider_smooth.setValue(80) # High value = heavily smoothed
        self.slider_smooth.valueChanged.connect(self.change_smoothing)
        self.lbl_smooth = QLabel("Smoothing (平滑度):")
        form_dsp.addRow(self.lbl_smooth, self.slider_smooth)
        
        self.slider_wf_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_wf_speed.setRange(1, 10)
        self.slider_wf_speed.setValue(10) # 10 = fastest
        form_dsp.addRow("WF Speed (Speed):", self.slider_wf_speed)
        
        self.spin_peaks = QSpinBox()
        self.spin_peaks.setRange(0, 10)
        self.spin_peaks.setValue(3)
        form_dsp.addRow("Detect Peaks:", self.spin_peaks)
        
        self.chk_alarm = QCheckBox("Enable SPL Alarm")
        form_dsp.addRow("", self.chk_alarm)
        
        self.combo_alarm_type = QComboBox()
        self.combo_alarm_type.addItems(["Band SPL", "Total Z-SPL", "Total A-SPL", "Total C-SPL"])
        form_dsp.addRow("Alarm Target:", self.combo_alarm_type)
        
        self.spin_alarm_thresh = QSpinBox()
        self.spin_alarm_thresh.setRange(-20, 200)
        self.spin_alarm_thresh.setValue(85)
        form_dsp.addRow("Alarm Thresh (dB):", self.spin_alarm_thresh)
        
        self.chk_buzzer = QCheckBox("Sound Buzzer (Warning)")
        form_dsp.addRow("", self.chk_buzzer)
        
        self.chk_alarm_rec = QCheckBox("Auto-Record on Alarm")
        self.chk_alarm_rec.setToolTip("Automatically start recording when alarm triggers. Stops 30s after alarm clears.")
        
        self.btn_alarm_rec_dir = QPushButton("Browse...")
        self.btn_alarm_rec_dir.setFixedWidth(120)
        self.btn_alarm_rec_dir.setStyleSheet("background-color: #444; color: white; padding: 3px;")
        self.btn_alarm_rec_dir.clicked.connect(self.pick_alarm_rec_dir)
        
        alarm_rec_row = QHBoxLayout()
        alarm_rec_row.addWidget(self.chk_alarm_rec)
        alarm_rec_row.addWidget(self.btn_alarm_rec_dir)
        alarm_rec_widget = QWidget()
        alarm_rec_widget.setLayout(alarm_rec_row)
        form_dsp.addRow("", alarm_rec_widget)
        
        self.lbl_alarm_rec_dir = QLabel("d:\\script\\FFT\\record")
        self.lbl_alarm_rec_dir.setStyleSheet("color: #888; font-size: 12px;")
        self.lbl_alarm_rec_dir.setWordWrap(True)
        form_dsp.addRow("Save to:", self.lbl_alarm_rec_dir)
        
        # User requested to link waterfall scales to spectrum height, 
        # so manual sliders were removed.
        
        box_dsp.setLayout(form_dsp)
        settings_layout.addWidget(box_dsp)
        
        # Instructions Label
        self.lbl_instructions = QLabel("说明:\n较低的 FFT Size 计算更快，但意味着最低频率下限会变高（例如 2048 点对应最低 23.4Hz 的分辨率）。\n移动和缩放频谱图会自动调整下方瀑布图的范围和对比度（增益）。")
        self.lbl_instructions.setWordWrap(True)
        self.lbl_instructions.setStyleSheet("color: #888; font-size: 16px; margin-top: 10px;")
        settings_layout.addWidget(self.lbl_instructions)
        
        settings_layout.addStretch(1) # Pushes everything up
        
        self.update_lang_text()

    def populate_audio_devices(self):
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_d = sd.default.device[0]
        
        idx = 0
        def_idx = 0
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                hostapi_name = hostapis[d['hostapi']]['name']
                name = f"[{hostapi_name}] {d['name']}"
                self.combo_device.addItem(name, userData=i)
                if i == default_d:
                    def_idx = idx
                idx += 1
        self.combo_device.setCurrentIndex(def_idx)

    def init_engine(self):
        self.engine = AudioEngine()
        
        # Connect signals
        self.engine.data_ready.connect(self.on_data_ready)
        
        # Auto-record alarm state
        self.alarm_last_triggered_time = None
        self._alarm_rec_started = False
        self.alarm_rec_dir = "d:\\script\\FFT\\record"
        
        # Load all settings (includes calibration with hardcoded fallback)
        self.load_settings()

        # Start
        self.engine.start_stream()

    def toggle_lang(self):
        self.current_lang = 'EN' if self.current_lang == 'ZH' else 'ZH'
        self.update_lang_text()
        
    def toggle_octave(self):
        self.update_lang_text()
        self.update_band_selection()
        
    def update_lang_text(self):
        if self.current_lang == 'ZH':
            if self.btn_octave.isChecked():
                self.pw_spectrum.setTitle("专业 1/3 倍频程分析 (1/3 Octave Band)")
            else:
                self.pw_spectrum.setTitle("实时频谱 (极大值保持 & 智能峰值检测)")
            self.pw_spectrum.setLabel('bottom', '频率', units='Hz')
            self.pw_spectrum.setLabel('left', '幅值', units='dB SPL')
            self.pw_waterfall.setTitle("历史信息瀑布图")
            self.pw_waterfall.setLabel('bottom', '频率', units='Hz')
            self.pw_waterfall.setLabel('left', '时间')
            self.pw_history.setTitle("声压级历史曲线")
            self.pw_history.setLabel('bottom', '时间', units='s')
            self.pw_history.setLabel('left', '声压级', units='dB')
            
            self.btn_lang.setText("EN / ZH")
            self.btn_octave.setText("1/3 倍频程")
            self.btn_hide.setText("隐藏菜单 ➡" if self.settings_widget.isVisible() else "显示菜单 ⬅")
            self.btn_screenshot.setText("截图")
            self.btn_record.setText("⏹ 停止记录" if self.is_recording else "⏺ 开始记录")
            self.spin_record_time.setPrefix("时长 (秒): ")
            self.spin_record_time.setSpecialValueText("无限制时间")
            
            self.btn_load_cal.setText("加载 UMIK-1 校准文件...")
            self.chk_dc.setText("移除直流偏移")
            self.chk_maxhold.setText("显示峰值保持线")
            self.chk_alarm.setText("启用超压警报阀值")
            self.chk_buzzer.setText("系统蜂鸣器提示音报警")
            self.lbl_fft_param.setText("快速傅里叶采样大小:")
            self.lbl_instructions.setText("说明:\n较低的 FFT Size 计算更快且延迟极低，但会使得最低可分辨频率上升（例如 2048 点对应最低 23.4Hz）。\n上下平移或缩放频谱图能直接充当瀑布图的『颜色增益』，截取对应的 SPL 区间！")
            if self.pw_waterfall.isHidden():
                self.btn_overlay_spec.setText("显示瀑布图")
            else:
                self.btn_overlay_spec.setText("隐藏")
            if self.pw_spectrum.isHidden():
                self.btn_overlay_wf.setText("显示频谱图")
            else:
                self.btn_overlay_wf.setText("隐藏")
        else:
            if self.btn_octave.isChecked():
                self.pw_spectrum.setTitle("Pro 1/3 Octave Band Analysis")
            else:
                self.pw_spectrum.setTitle("Real-Time Spectrum (Max Hold & Peak Detect)")
            self.pw_spectrum.setLabel('bottom', 'Frequency', units='Hz')
            self.pw_spectrum.setLabel('left', 'Amplitude', units='dB SPL')
            self.pw_waterfall.setTitle("Waterfall Spectrogram")
            self.pw_waterfall.setLabel('bottom', 'Frequency', units='Hz')
            self.pw_waterfall.setLabel('left', 'Time History')
            self.pw_history.setTitle("SPL History")
            self.pw_history.setLabel('bottom', 'Time', units='s')
            self.pw_history.setLabel('left', 'SPL', units='dB')
            
            self.btn_lang.setText("EN / ZH")
            self.btn_octave.setText("1/3 Octave")
            self.btn_hide.setText("Hide Menu ➡" if self.settings_widget.isVisible() else "Show Menu ⬅")
            self.btn_screenshot.setText("Screenshot")
            self.btn_record.setText("⏹ Stop Recording" if self.is_recording else "⏺ Start Recording")
            self.spin_record_time.setPrefix("Duration (s): ")
            self.spin_record_time.setSpecialValueText("Infinite")
            
            self.btn_load_cal.setText("Load UMIK-1 Calib File...")
            self.chk_dc.setText("Subtract DC")
            self.chk_maxhold.setText("Max-Hold Trace")
            self.chk_alarm.setText("Enable SPL Alarm")
            self.chk_buzzer.setText("Sound Buzzer")
            self.lbl_fft_param.setText("FFT Size:")
            self.lbl_instructions.setText("Note:\nLower FFT sizes are much faster (low latency) but raise the minimum resolving frequency (Resolution = SampleRate / FFT size).\nZooming/Panning the Spectrum Y-Axis directly controls the Waterfall's Color Gain Limits!")
            if self.pw_waterfall.isHidden():
                self.btn_overlay_spec.setText("Show Waterfall")
            else:
                self.btn_overlay_spec.setText("Hide")
            if self.pw_spectrum.isHidden():
                self.btn_overlay_wf.setText("Show Spectrum")
            else:
                self.btn_overlay_wf.setText("Hide")

    def toggle_menu(self):
        if self.settings_widget.isVisible():
            self.settings_widget.setVisible(False)
            self.btn_hide.setText("Show Menu ⬅")
        else:
            self.settings_widget.setVisible(True)
            self.btn_hide.setText("Hide Menu ➡")
            
    def eventFilter(self, source, event):
        if event.type() == QEvent.Enter:
            if source == self.pw_spectrum:
                if self.pw_waterfall.isHidden():
                    self.btn_overlay_spec.setText("Show Waterfall" if self.current_lang=="EN" else "显示瀑布图")
                else:
                    self.btn_overlay_spec.setText("Hide" if self.current_lang=="EN" else "隐藏")
                self.btn_overlay_spec.show()
                self.btn_overlay_spec.raise_()
            elif source == self.pw_waterfall:
                if self.pw_spectrum.isHidden():
                    self.btn_overlay_wf.setText("Show Spectrum" if self.current_lang=="EN" else "显示频谱图")
                else:
                    self.btn_overlay_wf.setText("Hide" if self.current_lang=="EN" else "隐藏")
                self.btn_overlay_wf.show()
                self.btn_overlay_wf.raise_()
        elif event.type() == QEvent.Leave:
            if source == self.pw_spectrum:
                self.btn_overlay_spec.hide()
            elif source == self.pw_waterfall:
                self.btn_overlay_wf.hide()
        return super().eventFilter(source, event)

    def toggle_spectrum(self):
        if self.pw_waterfall.isHidden():
            self.pw_waterfall.setVisible(True)
            self.btn_overlay_spec.hide()
        else:
            self.pw_spectrum.setVisible(False)

    def toggle_waterfall(self):
        if self.pw_spectrum.isHidden():
            self.pw_spectrum.setVisible(True)
            self.btn_overlay_wf.hide()
        else:
            self.pw_waterfall.setVisible(False)
                    
    def update_waterfall_gain(self, vb, viewRange):
        # viewRange contains the Y-axis min and max
        self.img_waterfall.setLevels([viewRange[0], viewRange[1]])
        
    def reset_minmax(self):
        self.primary_spl_min = 999.0
        self.primary_spl_max = -999.0
        self.spl_history = np.full(100, -20.0) # Reset history trace too
        
    # -- GUI Callbacks --
    def load_cal_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Calibration File", "d:\\script\\FFT", "Text files (*.txt);;All files (*.*)")
        if filepath:
            success, msg = self.engine.parse_calibration(filepath)
            self.lbl_cal_status.setText(msg)
            if success:
                self.lbl_cal_status.setStyleSheet("color: #4CAF50;")
            else:
                self.lbl_cal_status.setStyleSheet("color: #F44336;")

    def update_band_selection(self):
        if self.engine is None:
            return
        minX, maxX = self.region.getRegion()

        # Snap logic if octave is checked
        if self.btn_octave.isChecked() and hasattr(self.engine, 'oct_lower'):
            act_min = 10**minX
            act_max = 10**maxX
            
            idx_min = (np.abs(self.engine.oct_lower - act_min)).argmin()
            snap_min_log = np.log10(self.engine.oct_lower[idx_min])
            
            idx_max = (np.abs(self.engine.oct_upper - act_max)).argmin()
            snap_max_log = np.log10(self.engine.oct_upper[idx_max])
            
            # Avoid recursive signals
            if not getattr(self, '_is_snapping', False):
                self._is_snapping = True
                self.region.setRegion([snap_min_log, snap_max_log])
                self._is_snapping = False
                
            minX, maxX = snap_min_log, snap_max_log

        # Region returns log10 values, so convert back
        actual_min = 10**minX
        actual_max = 10**maxX
        self.engine.set_band(actual_min, actual_max)

    def change_device(self, idx):
        dev_id = self.combo_device.itemData(idx)
        self.engine.device_id = dev_id
        self.engine.restart_stream()
        self.max_hold_trace = None

    def change_fft_size(self, idx):
        self.engine.set_block_size(int(self.combo_fft.currentText()))
        self.max_hold_trace = None
        self.waterfall_data = None # Reset waterfall matrix shape

    def change_window(self, idx):
        self.engine.set_window(self.combo_win.currentText())

    def change_dc(self, state):
        self.engine.subtract_dc = (state == Qt.CheckState.Checked.value if hasattr(Qt.CheckState, 'Checked') else state == 2)

    def reset_max_hold(self):
        self.max_hold_trace = None
        if not self.chk_maxhold.isChecked():
            self.curve_maxhold.setData([], [])
            
    def change_smoothing(self, val):
        # Higher slider value = more smoothness (slower response)
        # alpha goes from ~0.99 (slider 1) down to ~0.01 (slider 99)
        self.engine.smoothing_factor = (100.0 - val) / 100.0
        
    def take_screenshot(self):
        # To avoid scaling issues, just grab the whole screen widget
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "d:\\script\\FFT\\screenshot.png", "Images (*.png)")
        if filepath:
            pixmap = self.centralWidget().grab()
            pixmap.save(filepath, "PNG")
            
    def load_settings(self):
        import json, os
        conf_path = "d:\\script\\FFT\\settings.json"
        data = {}
        if os.path.exists(conf_path):
            try:
                with open(conf_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"[Settings] Loaded from {conf_path}")
            except Exception as e:
                print(f"[Settings] Error reading file: {e}")
        else:
            print("[Settings] No settings file found — using defaults, will create on close.")

        # ---- Apply UI widget values (block signals to avoid duplicate engine calls) ----
        def _set_combo(combo, key, default):
            combo.blockSignals(True)
            combo.setCurrentText(str(data.get(key, default)))
            combo.blockSignals(False)

        def _set_spin(spin, key, default):
            spin.blockSignals(True)
            spin.setValue(int(data.get(key, default)))
            spin.blockSignals(False)

        def _set_check(chk, key, default):
            chk.blockSignals(True)
            chk.setChecked(bool(data.get(key, default)))
            chk.blockSignals(False)

        def _set_slider(slider, key, default):
            slider.blockSignals(True)
            slider.setValue(int(data.get(key, default)))
            slider.blockSignals(False)

        _set_combo(self.combo_fft,  'fft', '65536')
        _set_combo(self.combo_win,  'win', 'blackman-harris')
        _set_check(self.chk_dc,     'dc',  True)
        _set_check(self.chk_maxhold,'maxhold', True)
        _set_slider(self.slider_smooth,   'smooth',  80)
        _set_slider(self.slider_wf_speed, 'wfspeed', 10)
        _set_spin(self.spin_peaks,        'peaks',   3)

        self.combo_primary.blockSignals(True)
        self.combo_primary.setCurrentIndex(int(data.get('primary_spl', 0)))
        self.combo_primary.blockSignals(False)

        _set_check(self.chk_alarm,    'alarm',    False)
        self.combo_alarm_type.blockSignals(True)
        self.combo_alarm_type.setCurrentIndex(int(data.get('alarm_type', 0)))
        self.combo_alarm_type.blockSignals(False)
        _set_spin(self.spin_alarm_thresh, 'alarm_thresh', 85)
        _set_check(self.chk_buzzer,   'buzzer',   False)
        _set_check(self.chk_alarm_rec,'alarm_rec',False)

        if data.get('alarm_rec_dir'):
            self.alarm_rec_dir = data['alarm_rec_dir']
            self.lbl_alarm_rec_dir.setText(data['alarm_rec_dir'])

        # Band region
        band_region = data.get('band_region')
        if band_region and len(band_region) == 2:
            self.region.blockSignals(True)
            self.region.setRegion(band_region)
            self.region.blockSignals(False)

        # Restore audio device by name
        saved_device = data.get('device_name', '')
        if saved_device:
            for i in range(self.combo_device.count()):
                if self.combo_device.itemText(i) == saved_device:
                    self.combo_device.blockSignals(True)
                    self.combo_device.setCurrentIndex(i)
                    self.combo_device.blockSignals(False)
                    self.engine.device_id = self.combo_device.itemData(i)
                    break

        # ---- Sync engine directly (one call each, no signal cascades) ----
        self.engine.set_block_size(int(self.combo_fft.currentText()))
        self.engine.set_window(self.combo_win.currentText())
        self.engine.subtract_dc = self.chk_dc.isChecked()
        self.engine.smoothing_factor = (100.0 - self.slider_smooth.value()) / 100.0
        minX, maxX = self.region.getRegion()
        self.engine.set_band(10 ** minX, 10 ** maxX)

        # ---- Load calibration ----
        cal_file = data.get('cal_file', '')
        cal_loaded = False
        if cal_file and os.path.exists(cal_file):
            success, msg = self.engine.parse_calibration(cal_file)
            print(f"[Settings] Calibration: {msg}")
            self.lbl_cal_status.setText(msg)
            if success:
                self.lbl_cal_status.setStyleSheet("color: #4CAF50;")
                cal_loaded = True
            else:
                self.lbl_cal_status.setStyleSheet("color: #F44336;")

        if not cal_loaded:
            # Fallback: try the standard local calibration file
            default_cal = r'd:\script\FFT\calibration\7189949_90deg.txt'
            if os.path.exists(default_cal):
                success, msg = self.engine.parse_calibration(default_cal)
                print(f"[Settings] Calibration (fallback): {msg}")
                if success:
                    self.lbl_cal_status.setText(msg)
                    self.lbl_cal_status.setStyleSheet("color: #4CAF50;")

        print(f"[Settings] Engine state — FFT={self.engine.block_size}, "
              f"win={self.engine.window_type}, smooth={self.engine.smoothing_factor:.3f}, "
              f"dc={self.engine.subtract_dc}")

        # ---- Restore spectrum view range ----
        view_x = data.get('view_x')
        view_y = data.get('view_y')
        if view_x and len(view_x) == 2:
            self.pw_spectrum.setXRange(view_x[0], view_x[1], padding=0)
        if view_y and len(view_y) == 2:
            self.pw_spectrum.setYRange(view_y[0], view_y[1], padding=0)

    def save_settings(self):
        import json, os
        conf_path = "d:\\script\\FFT\\settings.json"
        try:
            os.makedirs(os.path.dirname(conf_path), exist_ok=True)
            band_region = list(self.region.getRegion())
            data = {
                'fft':          self.combo_fft.currentText(),
                'win':          self.combo_win.currentText(),
                'dc':           self.chk_dc.isChecked(),
                'maxhold':      self.chk_maxhold.isChecked(),
                'smooth':       self.slider_smooth.value(),
                'wfspeed':      self.slider_wf_speed.value(),
                'peaks':        self.spin_peaks.value(),
                'primary_spl':  self.combo_primary.currentIndex(),
                'alarm':        self.chk_alarm.isChecked(),
                'alarm_type':   self.combo_alarm_type.currentIndex(),
                'alarm_thresh': self.spin_alarm_thresh.value(),
                'buzzer':       self.chk_buzzer.isChecked(),
                'alarm_rec':    self.chk_alarm_rec.isChecked(),
                'alarm_rec_dir':self.alarm_rec_dir,
                'cal_file':     self.engine.cal_file if self.engine else "",
                'band_region':  band_region,
                'device_name':  self.combo_device.currentText(),
            }
            try:
                vb = self.pw_spectrum.getViewBox()
                rng = vb.viewRange()
                data['view_x'] = rng[0]
                data['view_y'] = rng[1]
            except Exception:
                pass
            with open(conf_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"[Settings] Saved to {conf_path}")
        except Exception as e:
            print(f"[Settings] Error saving: {e}")

    def closeEvent(self, event):
        self.save_settings()
        if self.is_recording:
            self.stop_and_save_recording(blocking=True)
        elif hasattr(self, '_save_thread') and self._save_thread.is_alive():
            self._save_thread.join(timeout=60)
        if self.engine:
            self.engine.stop_stream()
        super().closeEvent(event)
            
    def toggle_recording(self):
        should_record = self.btn_record.isChecked()
        if should_record:
            filepath, _ = QFileDialog.getSaveFileName(self, "Save Data Package", "d:\\script\\FFT\\record", "NPZ Archive (*.npz)")
            if filepath:
                if not filepath.endswith('.npz'):
                    filepath += '.npz'
                self.record_filepath = filepath
                
                # Setup internal data buffers
                self.record_data_dict = {
                    'timestamps': [],
                    'lz': [], 'la': [], 'lc': [], 'band': [],
                    'freqs': None, # We will save this once
                    'spectra': [],
                    'waterfall': [] # Can be massive, so we append the history row
                }
                self.record_start_time = time.time()
                self.record_duration = self.spin_record_time.value()
                
                self.btn_record.setText("⏹ 停止记录" if self.current_lang == 'ZH' else "⏹ Stop Recording")
                self.btn_record.setStyleSheet("background-color: #F44336; color: white; padding: 5px; font-weight: bold;")
                
                # Activate flag LAST to prevent thread race condition
                self.is_recording = True
            else:
                self.btn_record.setChecked(False)
                self.is_recording = False
        else:
            self.stop_and_save_recording()
            
    def stop_and_save_recording(self, blocking=False):
        self.is_recording = False
        self.btn_record.setChecked(False)
        self.spin_record_time.setEnabled(True)
        self.btn_record.setText("⏺ 开始记录" if self.current_lang == 'ZH' else "⏺ Start Recording")
        self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px; font-weight: bold;")

        # Grab data immediately and clear shared dict so recording can restart
        data_to_save = self.record_data_dict
        filepath = getattr(self, 'record_filepath', None)
        self.record_data_dict = {}

        if not data_to_save or not filepath or len(data_to_save.get('timestamps', [])) == 0:
            return

        def _do_save():
            try:
                np.savez_compressed(
                    filepath,
                    timestamps=np.array(data_to_save['timestamps']),
                    lz=np.array(data_to_save['lz']),
                    la=np.array(data_to_save['la']),
                    lc=np.array(data_to_save['lc']),
                    band=np.array(data_to_save['band']),
                    freqs=data_to_save['freqs'],
                    spectra=np.array(data_to_save['spectra'])
                )
                print(f"Saved recording to {filepath}")
            except Exception as e:
                print(f"Error saving NPZ record: {e}")

        t = threading.Thread(target=_do_save, daemon=True)
        t.start()
        self._save_thread = t
        if blocking:
            t.join(timeout=60)
            
    def pick_alarm_rec_dir(self):
        from PyQt5.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self, "Select Alarm Recording Folder", self.alarm_rec_dir)
        if folder:
            self.alarm_rec_dir = folder
            self.lbl_alarm_rec_dir.setText(folder)
            
    def start_alarm_recording(self):
        """Auto-trigger recording when alarm fires. Uses an auto-timestamped path."""
        import os
        os.makedirs(self.alarm_rec_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.alarm_rec_dir, f"alarm_{ts}.npz")
        self.record_filepath = filepath
        self.record_data_dict = {
            'timestamps': [],
            'lz': [], 'la': [], 'lc': [], 'band': [],
            'freqs': None,
            'spectra': []
        }
        self.record_start_time = time.time()
        self.record_duration = 0  # No manual cap — controlled by alarm state machine
        self.spin_record_time.setEnabled(False)
        self.btn_record.setChecked(True)
        self.btn_record.setText("⏹ 报警录制中..." if self.current_lang == 'ZH' else "⏹ Alarm Recording...")
        self.btn_record.setStyleSheet("background-color: #E65100; color: white; padding: 5px; font-weight: bold;")
        self.is_recording = True
        print(f"[Auto-Record] Alarm triggered — recording started: {filepath}")
            
    def on_plot_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            log_x = None
            if self.pw_spectrum.sceneBoundingRect().contains(pos):
                mousePoint = self.pw_spectrum.plotItem.vb.mapSceneToView(pos)
                log_x = mousePoint.x()
            elif self.pw_waterfall.sceneBoundingRect().contains(pos):
                mousePoint = self.pw_waterfall.plotItem.vb.mapSceneToView(pos)
                log_x = mousePoint.x()
                
            if log_x is not None:
                self.current_crosshair_logx = log_x
                self.vLine_spec.setPos(log_x)
                self.vLine_wf.setPos(log_x)
                self.vLine_spec.show()
                self.vLine_wf.show()
                # Text will update accurately on the next data render callback

        
    def change_waterfall_levels(self):
        min_v = self.slider_heat_min.value()
        max_v = self.slider_heat_max.value()
        if min_v >= max_v: return
        self.img_waterfall.setLevels([min_v, max_v])

    # -- Realtime Render Callback from Thread --
    def on_data_ready(self, freqs, spectrum_spl, lz, la, lc, l_band, is_clipping, oct_freqs, oct_spl):
        try:
            # Prevent 0 indexing of log
            valid_idx = freqs > 0
            log_f = np.log10(freqs[valid_idx])
            valid_spl = spectrum_spl[valid_idx]
            
            # Render Top Spectrum
            if self.btn_octave.isChecked():
                log_oct_f = np.log10(oct_freqs)
                # width ~0.1 on log10 scale represents exactly 1/3 octave
                self.bar_octave.setOpts(x=log_oct_f, height=oct_spl, width=0.09)
                self.bar_octave.show()
                self.curve_realtime.hide()
            else:
                self.curve_realtime.setData(log_f, valid_spl)
                self.curve_realtime.show()
                self.bar_octave.hide()
            
            # Max Hold Logic
            if self.chk_maxhold.isChecked():
                if self.btn_octave.isChecked():
                    if getattr(self, 'max_hold_octave', None) is None or len(self.max_hold_octave) != len(oct_spl):
                        self.max_hold_octave = oct_spl.copy()
                    else:
                        self.max_hold_octave = np.maximum(self.max_hold_octave, oct_spl)
                    self.curve_maxhold.setData(np.log10(oct_freqs), self.max_hold_octave)
                else:
                    if getattr(self, 'max_hold_trace', None) is None or len(self.max_hold_trace) != len(valid_spl):
                        self.max_hold_trace = valid_spl.copy()
                    else:
                        self.max_hold_trace = np.maximum(self.max_hold_trace, valid_spl)
                    self.curve_maxhold.setData(log_f, self.max_hold_trace)
                self.curve_maxhold.show()
            else:
                self.curve_maxhold.hide()
                
            # Update Primary Readout & History
            primary_idx = self.combo_primary.currentIndex()
            if primary_idx == 0:
                current_spl = lz
                unit = "dB (Z)"
            elif primary_idx == 1:
                current_spl = la
                unit = "dB (A)"
            else:
                current_spl = lc
                unit = "dB (C)"
            
            # Min/Max Tracking
            if current_spl < self.primary_spl_min: self.primary_spl_min = current_spl
            if current_spl > self.primary_spl_max: self.primary_spl_max = current_spl
            
            # Clipping / Overload
            if is_clipping:
                self.lbl_overload.setStyleSheet("font-family: 'Courier New', monospace; font-size: 20px; font-weight: bold; color: white; background-color: #F44336; border: 2px solid red; border-radius: 5px; padding: 5px;")
            else:
                self.lbl_overload.setStyleSheet("font-family: 'Courier New', monospace; font-size: 20px; font-weight: bold; color: #555; background-color: #111; border: 2px solid #333; border-radius: 5px; padding: 5px;")
                
            self.lbl_lz.setText(f"Total SPL\n{current_spl:.1f} {unit}")
            self.lbl_minmax.setText(f"Min: {self.primary_spl_min:.1f} | Max: {self.primary_spl_max:.1f}")
            
            # Record Data
            if self.is_recording:
                try:
                    current_time = time.time()
                    elapsed = current_time - self.record_start_time
                    
                    # Check auto-stop timer
                    if self.record_duration > 0 and elapsed >= self.record_duration:
                        self.stop_and_save_recording()
                    else:
                        self.record_data_dict['timestamps'].append(current_time)
                        self.record_data_dict['lz'].append(lz)
                        self.record_data_dict['la'].append(la)
                        self.record_data_dict['lc'].append(lc)
                        self.record_data_dict['band'].append(l_band)
                        self.record_data_dict['spectra'].append(valid_spl.astype(np.float32))
                        if self.record_data_dict['freqs'] is None:
                            self.record_data_dict['freqs'] = freqs[valid_idx].astype(np.float32)
                except KeyError:
                    pass # Wait for dictionary to finish initializing

            # Update History Curve
            self.spl_history[:-1] = self.spl_history[1:]
            self.spl_history[-1] = current_spl
            
            x_data = np.arange(-99, 1)
            self.curve_history.setData(x_data, self.spl_history)
            
            # Show track lines
            if self.primary_spl_max != -999.0:
                self.line_hist_max.setValue(self.primary_spl_max)
            if self.primary_spl_min != 999.0:
                self.line_hist_min.setValue(self.primary_spl_min)
                
            # Auto-Scale Y Axis robustly
            if self.primary_spl_min != 999.0 and self.primary_spl_max != -999.0:
                padding = max((self.primary_spl_max - self.primary_spl_min) * 0.1, 2)
                self.pw_history.setYRange(self.primary_spl_min - padding, self.primary_spl_max + padding)
            
            # Region label formatter
            minX, maxX = self.region.getRegion()
            fmin = 10**minX
            fmax = 10**maxX
            self.lbl_band.setText(f"Band SPL\n{l_band:.1f} dB\n({fmin:.1f}-{fmax:.1f}Hz)")
            
            self.lbl_la.setText(f"LAeq: {la:.1f} dB(A)")
            self.lbl_lc.setText(f"LCeq: {lc:.1f} dB(C)")
            
            # Peaks Detection
            num_peaks = self.spin_peaks.value()
            if num_peaks > 0:
                if self.btn_octave.isChecked():
                    log_target_f = np.log10(oct_freqs)
                    target_spl = oct_spl
                    peaks_idx, _ = find_peaks(target_spl, prominence=1, distance=1)
                else:
                    log_target_f = log_f
                    target_spl = valid_spl
                    peaks_idx, _ = find_peaks(target_spl, prominence=3, distance=5)
                    
                # Sort by strength
                if len(peaks_idx) > 0:
                    peak_mags = target_spl[peaks_idx]
                    sorted_indices = np.argsort(peak_mags)[::-1]
                    top_peaks = peaks_idx[sorted_indices[:num_peaks]]
                    
                    # Update Scatter
                    self.peak_scatter.setData(log_target_f[top_peaks], target_spl[top_peaks])
                    
                    # Update Texts
                    for i in range(10):
                        if i < len(top_peaks):
                            idx = top_peaks[i]
                            self.peak_texts[i].setText(f"{10**log_target_f[idx]:.0f}Hz\n{target_spl[idx]:.1f}")
                            self.peak_texts[i].setPos(log_target_f[idx], target_spl[idx])
                            self.peak_texts[i].show()
                        else:
                            self.peak_texts[i].hide()
                else:
                    self.peak_scatter.setData([], [])
                    for t in self.peak_texts: t.hide()
            else:
                self.peak_scatter.setData([], [])
                for t in self.peak_texts: t.hide()
                
            # Alarm Logic
            if self.chk_alarm.isChecked():
                idx_alarm = self.combo_alarm_type.currentIndex()
                val_alarm = l_band if idx_alarm == 0 else (lz if idx_alarm == 1 else (la if idx_alarm == 2 else lc))
                thresh = self.spin_alarm_thresh.value()
                
                # Highlight the correct label based on target
                target_label = self.lbl_band if idx_alarm == 0 else self.lbl_lz
                other_label = self.lbl_lz if idx_alarm == 0 else self.lbl_band
                
                # Reset the other label to normal
                if other_label == self.lbl_band:
                    other_label.setStyleSheet("font-family: 'Courier New', monospace; font-size: 24px; font-weight: bold; color: #FFC107; background-color: #111; border: 2px solid #333; border-radius: 5px; padding: 10px;")
                else:
                    other_label.setStyleSheet("font-family: 'Courier New', monospace; font-size: 28px; font-weight: bold; color: #00FF00; background-color: #111; border: 2px solid #333; border-radius: 5px; padding: 10px;")

                if val_alarm >= thresh:
                    target_label.setStyleSheet("font-family: 'Courier New', monospace; font-size: 24px; font-weight: bold; color: white; background-color: #D32F2F; border: 2px solid red; border-radius: 5px; padding: 10px;")
                    if self.chk_buzzer.isChecked() and _winsound is not None:
                        now = time.time()
                        if now - getattr(self, '_last_beep_time', 0) >= 0.4:
                            self._last_beep_time = now
                            threading.Thread(
                                target=lambda: _winsound.Beep(2000, 200),
                                daemon=True).start()
                    
                    # Auto-record state machine: alarm active
                    if self.chk_alarm_rec.isChecked():
                        self.alarm_last_triggered_time = time.time()
                        if not self.is_recording and not getattr(self, '_alarm_rec_started', False):
                            self._alarm_rec_started = True
                            self.start_alarm_recording()
                else:
                    if target_label == self.lbl_band:
                        target_label.setStyleSheet("font-family: 'Courier New', monospace; font-size: 24px; font-weight: bold; color: #FFC107; background-color: #111; border: 2px solid #333; border-radius: 5px; padding: 10px;")
                    else:
                        target_label.setStyleSheet("font-family: 'Courier New', monospace; font-size: 28px; font-weight: bold; color: #00FF00; background-color: #111; border: 2px solid #333; border-radius: 5px; padding: 10px;")
                    
                    # Auto-record state machine: alarm cleared — wait 30s then stop
                    if self.chk_alarm_rec.isChecked() and getattr(self, '_alarm_rec_started', False):
                        if self.alarm_last_triggered_time is not None:
                            post_alarm_elapsed = time.time() - self.alarm_last_triggered_time
                            if post_alarm_elapsed >= 30.0:
                                self._alarm_rec_started = False
                                self.alarm_last_triggered_time = None
                                if self.is_recording:
                                    self.stop_and_save_recording()
            else:
                self.lbl_band.setStyleSheet("font-family: 'Courier New', monospace; font-size: 24px; font-weight: bold; color: #FFC107; background-color: #111; border: 2px solid #333; border-radius: 5px; padding: 10px;")
                self.lbl_lz.setStyleSheet("font-family: 'Courier New', monospace; font-size: 28px; font-weight: bold; color: #00FF00; background-color: #111; border: 2px solid #333; border-radius: 5px; padding: 10px;")
            
            # Crosshair update
            if self.current_crosshair_logx is not None:
                if self.btn_octave.isChecked():
                    log_target_f = np.log10(oct_freqs)
                    target_spl = oct_spl
                else:
                    log_target_f = log_f
                    target_spl = valid_spl
                    
                idx = (np.abs(log_target_f - self.current_crosshair_logx)).argmin()
                exact_freq = 10 ** log_target_f[idx]
                spl_at_freq = target_spl[idx]
                
                # Snap visual line to exact bin
                self.vLine_spec.setPos(log_target_f[idx])
                
                self.lbl_crosshair_readout.setText(f"Marker:\n{exact_freq:.1f} Hz | {spl_at_freq:.1f} dB")
                self.crosshair_text.setText(f"{exact_freq:.1f} Hz\n{spl_at_freq:.1f} dB")
                self.crosshair_text.setPos(log_target_f[idx], spl_at_freq)
            
            # Waterfall Speed Decimation
            self.waterfall_skip_counter += 1
            skip_frames = 11 - self.slider_wf_speed.value()
            if self.waterfall_skip_counter < skip_frames:
                return
            self.waterfall_skip_counter = 0
            
            # To make Waterfall Image exactly match a LogX PlotWidget above, we must physically
            # remap the linear FFT bins into logarithmic bins before calling setImage, since
            # ImageItems are inherently linear grids.
            
            if not hasattr(self, 'log_mapping_initialized') or self.log_mapping_initialized != len(valid_spl):
                # Precompute a logarithmic pixel map (e.g. 500 pixels wide)
                self.log_pixel_width = 800
                self.log_x_bounds = np.linspace(log_f[0], log_f[-1], self.log_pixel_width)
                
                # Create interpolation function from [Linear Freq Array] -> [SPL]
                # Then query it at [Logarithmic Freq Array]
                self.log_mapping_initialized = len(valid_spl)
                self.waterfall_data_log = np.full((self.log_pixel_width, self.waterfall_history), -100.0)
                
            # Compute current frame in Log space
            interp_func = interp1d(log_f, valid_spl, kind='linear', fill_value='extrapolate')
            log_frame_spl = interp_func(self.log_x_bounds)
            
            # Shift history
            self.waterfall_data_log[:, :-1] = self.waterfall_data_log[:, 1:]
            self.waterfall_data_log[:, -1] = log_frame_spl
            
            # Apply Transform
            xscale = (log_f[-1] - log_f[0]) / self.log_pixel_width
            tr = pg.QtGui.QTransform()
            tr.translate(log_f[0], 0)
            tr.scale(xscale, 1.0)
            self.img_waterfall.setTransform(tr)
            
            self.img_waterfall.setImage(self.waterfall_data_log, autoLevels=False)
        except Exception as e:
            print(f"Exception in on_data_ready: {e}")


def main():
    try:
        from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor
        from PyQt5.QtWidgets import QSplashScreen

        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        # Splash Screen
        pixmap = QPixmap(1200, 675)
        pixmap.fill(QColor("#0d0d0d"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("#e0e0e0"))
        painter.setFont(QFont("Microsoft YaHei", 32, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "UMIK-1 Spectrum Analyzer\n\nStarting DSP Engine...")
        painter.end()
        
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()
        
        time.sleep(1.2) # Let the user see the branding
        
        window = MainWindow()
        window.show()
        splash.finish(window)
        
        app.exec_()
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
