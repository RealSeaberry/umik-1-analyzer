import sys
import numpy as np
import pyqtgraph as pg
from scipy.interpolate import interp1d
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, QSlider)
from PyQt5.QtCore import Qt
import datetime


class LogFreqAxis(pg.AxisItem):
    """Axis that displays log10 internal values as real Hz labels."""
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

class OffsetTimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = 0.0

    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            try:
                # Add offset back properly to format absolute local time
                dt = datetime.datetime.fromtimestamp(self.offset + v)
                strings.append(dt.strftime('%H:%M:%S'))
            except:
                strings.append("")
        return strings

class UmikDataViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UMIK-1 Spectrum Data Viewer / UMIK-1 频谱数据回放与分析器")
        self.resize(1200, 900)
        self.setStyleSheet("background-color: #0d0d0d; color: #e0e0e0;")
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.current_lang = 'ZH'
        
        # Top Panel
        self.top_panel = QHBoxLayout()
        self.btn_lang = QPushButton("EN / 中文")
        self.btn_lang.setStyleSheet("background-color: #333; color: white; padding: 10px; font-weight: bold; font-size: 14px;")
        self.btn_lang.clicked.connect(self.toggle_lang)
        self.btn_load = QPushButton("Load NPZ Record / 加载存档数据包...")
        self.btn_load.clicked.connect(self.load_npz)
        self.btn_load.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; font-weight: bold; font-size: 14px;")
        
        self.lbl_info = QLabel("No Data Loaded")
        
        self.top_panel.addWidget(self.btn_lang)
        self.top_panel.addWidget(self.btn_load)
        self.top_panel.addWidget(self.lbl_info)
        self.top_panel.addStretch(1)
        self.layout.addLayout(self.top_panel)
        
        # Main Plots Area (Split)
        self.plot_layout = QVBoxLayout()
        self.layout.addLayout(self.plot_layout, stretch=3)
        
        # 1. SPL Time Series Plot
        self.date_axis_spl = OffsetTimeAxisItem(orientation='bottom')
        self.pw_spl = pg.PlotWidget(title="SPL Time Series (声压级全程走势)", axisItems={'bottom': self.date_axis_spl})
        self.pw_spl.showGrid(x=True, y=True, alpha=0.3)
        self.pw_spl.setLabel('bottom', 'Local Time (本地时间)')
        self.pw_spl.setLabel('left', 'SPL', units='dB')
        self.pw_spl.addLegend()
        self.plot_layout.addWidget(self.pw_spl, stretch=1)
        
        # Interactive Time Line Marker
        self.time_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('yellow', width=2))
        self.pw_spl.addItem(self.time_line)
        self.time_line.sigPositionChanged.connect(self.on_time_scrub)
        
        # Bottom Plot Area (Tabbed to save screen space)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { height: 40px; width: 250px; font-size: 16px; font-weight: bold; background: #333; color: #ccc; padding: 5px; } QTabBar::tab:selected { background: #2196F3; color: white; }")
        self.plot_layout.addWidget(self.tabs, stretch=2)
        
        # Tab 1: Historical Waterfall Plot
        self.date_axis_wf = OffsetTimeAxisItem(orientation='bottom')
        self.log_freq_axis_wf = LogFreqAxis(orientation='left')
        self.pw_waterfall = pg.PlotWidget(title="Full Waterfall (全量程瀑布图)",
                                          axisItems={'bottom': self.date_axis_wf,
                                                     'left': self.log_freq_axis_wf})
        self.pw_waterfall.setLabel('bottom', 'Local Time (本地时间)')
        self.pw_waterfall.setLabel('left', 'Frequency', units='Hz')
        self.pw_waterfall.setMouseEnabled(x=True, y=True)
        self.img_waterfall = pg.ImageItem()
        self.pw_waterfall.addItem(self.img_waterfall)
        colormap = pg.colormap.get('inferno')
        self.img_waterfall.setLookupTable(colormap.getLookupTable())
        self.pw_waterfall.setXLink(self.pw_spl)  # Link Time Axes
        self.tabs.addTab(self.pw_waterfall, "Waterfall 瀑布图")
        
        # Tab 2: Snapshot Spectrum Plot (Controlled by Time Marker)
        spec_tab_widget = QWidget()
        spec_layout = QVBoxLayout(spec_tab_widget)
        
        self.pw_spec = pg.PlotWidget(title="Spectrum Snapshot at Cursor (游标处截面频谱)",
                                     axisItems={'bottom': LogFreqAxis(orientation='bottom')})
        self.pw_spec.showGrid(x=True, y=True, alpha=0.3)
        self.pw_spec.setLabel('bottom', 'Frequency', units='Hz')
        self.pw_spec.setLabel('left', 'SPL', units='dB')
        self.pw_spec.setYRange(0, 130)
        self.curve_spec = self.pw_spec.plot(pen=pg.mkPen('cyan', width=1.5))
        spec_layout.addWidget(self.pw_spec, stretch=1)
        
        slider_layout = QHBoxLayout()
        self.lbl_slider_time = QLabel("Local Time: --")
        self.lbl_slider_time.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffeb3b;")
        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setEnabled(False)
        self.slider_time.valueChanged.connect(self.on_slider_scrub)
        
        slider_layout.addWidget(self.lbl_slider_time)
        slider_layout.addWidget(self.slider_time, stretch=1)
        spec_layout.addLayout(slider_layout)
        
        self.tabs.addTab(spec_tab_widget, "Spectrum Snapshot 截面分析")
        
        # Stats Table
        self.table = QTableWidget(4, 4)
        self.table.setHorizontalHeaderLabels(["Metric (dB)", "Min", "Max", "Mean"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet("background-color: #222; color: white;")
        self.layout.addWidget(self.table, stretch=1)
        
        self.loaded_data = None
        self.update_lang_text()
        
    def toggle_lang(self):
        self.current_lang = 'EN' if self.current_lang == 'ZH' else 'ZH'
        self.update_lang_text()
        
    def update_lang_text(self):
        if self.current_lang == 'ZH':
            self.pw_spl.setTitle("实时声压级全程走势")
            self.pw_spl.setLabel('bottom', '本地时间')
            self.pw_spl.setLabel('left', '声压级', units='dB')
            self.tabs.setTabText(0, "历史瀑布图")
            self.tabs.setTabText(1, "游标截面分析")
            self.pw_waterfall.setTitle("全量程瀑布图")
            self.pw_waterfall.setLabel('bottom', '本地时间')
            self.pw_waterfall.setLabel('left', '频率', units='Hz')
            self.pw_spec.setTitle("当前游标处截面频谱")
            self.pw_spec.setLabel('bottom', '频率', units='Hz')
            self.pw_spec.setLabel('left', '声压级', units='dB')
            self.btn_load.setText("加载录音数据包...")
            self.btn_lang.setText("EN / ZH")
            self.table.setHorizontalHeaderLabels(["指标 (dB)", "最小", "最大", "平均"])
            if self.loaded_data is None:
                self.lbl_info.setText("空闲中，未加载数据")
            else:
                self.lbl_info.setText("数据包已成功加载")
            
            time_str = self.lbl_slider_time.text().split(":")[-1].strip() if ":" in self.lbl_slider_time.text() else "--"
            self.lbl_slider_time.setText(f"本地时间: {time_str}")
        else:
            self.pw_spl.setTitle("SPL Time Series")
            self.pw_spl.setLabel('bottom', 'Local Time')
            self.pw_spl.setLabel('left', 'SPL', units='dB')
            self.tabs.setTabText(0, "Waterfall")
            self.tabs.setTabText(1, "Snapshot")
            self.pw_waterfall.setTitle("Full Waterfall")
            self.pw_waterfall.setLabel('bottom', 'Local Time')
            self.pw_waterfall.setLabel('left', 'Frequency', units='Hz')
            self.pw_spec.setTitle("Spectrum Snapshot at Cursor")
            self.pw_spec.setLabel('bottom', 'Frequency', units='Hz')
            self.pw_spec.setLabel('left', 'SPL', units='dB')
            self.btn_load.setText("Load Record Archive...")
            self.btn_lang.setText("EN / ZH")
            self.table.setHorizontalHeaderLabels(["Metric (dB)", "Min", "Max", "Mean"])
            if self.loaded_data is None:
                self.lbl_info.setText("Idle, No Data Loaded")
            else:
                self.lbl_info.setText("Data Loaded Successfully")
                
            time_str = self.lbl_slider_time.text().split(":")[-1].strip() if ":" in self.lbl_slider_time.text() else "--"
            self.lbl_slider_time.setText(f"Local Time: {time_str}")
        
    def load_npz(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select NPZ Record", "d:\\script\\FFT", "NPZ Archives (*.npz)")
        if not filepath: return
        
        try:
            data = np.load(filepath)
            timestamps = data['timestamps']
            
            # Extract absolute start time
            t_min = timestamps[0]
            
            # Set offset for our custom axes
            self.date_axis_spl.offset = t_min
            self.date_axis_wf.offset = t_min
            
            # Normalize to 0 to prevent PyqtGraph float32 overflow limit
            timestamps = timestamps - t_min 
            
            lz = data['lz']
            la = data['la']
            lc = data['lc']
            band = data['band']
            freqs = data['freqs']
            spectra = data['spectra'] # Shape (Time, Frequencies)
            
            self.loaded_data = {
                't': timestamps,
                'f': freqs,
                's': spectra
            }
            
            # Clear & Plot SPLs
            self.pw_spl.clear()
            self.pw_spl.addItem(self.time_line)
            self.pw_spl.plot(timestamps, lz, pen=pg.mkPen('g', width=2), name="Total LZeq")
            self.pw_spl.plot(timestamps, la, pen=pg.mkPen('y', width=1.5), name="Total LAeq")
            self.pw_spl.plot(timestamps, lc, pen=pg.mkPen('c', width=1.5), name="Total LCeq")
            self.pw_spl.plot(timestamps, band, pen=pg.mkPen('r', width=3), name="Band SPL")
            
            # ---- Waterfall: remap linear-freq spectra to log-freq grid ----
            valid_mask = freqs > 0
            valid_freqs = freqs[valid_mask]
            log_f_min = np.log10(valid_freqs[0])
            log_f_max = np.log10(valid_freqs[-1])

            # Build a log-spaced frequency grid (256 bins looks good)
            N_LOG_BINS = 256
            log_freq_grid = np.linspace(log_f_min, log_f_max, N_LOG_BINS)
            actual_freq_grid = 10 ** log_freq_grid

            # Remap each time frame from linear freq bins to log freq bins
            n_frames = spectra.shape[0]
            log_spectra = np.empty((n_frames, N_LOG_BINS), dtype=np.float32)
            for i in range(n_frames):
                row = spectra[i][valid_mask]
                interp_fn = interp1d(valid_freqs, row, kind='linear',
                                     bounds_error=False, fill_value=(row[0], row[-1]))
                log_spectra[i] = interp_fn(actual_freq_grid)

            # Downsample time axis for very long recordings (max 2000 columns)
            MAX_TIME_COLS = 2000
            if n_frames > MAX_TIME_COLS:
                step = n_frames // MAX_TIME_COLS
                disp_spectra = log_spectra[::step]
                disp_timestamps = timestamps[::step]
            else:
                disp_spectra = log_spectra
                disp_timestamps = timestamps

            t_start = timestamps[0]   # = 0 after normalization
            t_max = timestamps[-1]
            t_duration = t_max - t_start
            disp_t_start = disp_timestamps[0]
            disp_t_dur = disp_timestamps[-1] - disp_t_start

            # ImageItem: shape (Time, Freq) → X=Time, Y=Freq
            self.img_waterfall.setImage(disp_spectra)
            self.img_waterfall.setRect(pg.QtCore.QRectF(
                disp_t_start, log_f_min, disp_t_dur, log_f_max - log_f_min))
            spl_min = float(np.percentile(log_spectra, 5))
            spl_max = float(np.percentile(log_spectra, 99))
            self.img_waterfall.setLevels([spl_min, spl_max])
            self.pw_waterfall.setYRange(log_f_min, log_f_max)
            
            # Auto range
            self.pw_spl.setXRange(t_start, t_max)
            self.pw_spl.autoRange()

            # Set spectrum snapshot X range to match frequency data
            self.pw_spec.setXRange(log_f_min, log_f_max)
            
            # Populate Table
            metrics = [("Total LZeq", lz), ("Total LAeq", la), ("Total LCeq", lc), ("Target Band SPL", band)]
            for row, (name, arr) in enumerate(metrics):
                self.table.setItem(row, 0, QTableWidgetItem(name))
                self.table.setItem(row, 1, QTableWidgetItem(f"{np.min(arr):.2f}"))
                self.table.setItem(row, 2, QTableWidgetItem(f"{np.max(arr):.2f}"))
                self.table.setItem(row, 3, QTableWidgetItem(f"{np.mean(arr):.2f}"))
                
            self.lbl_info.setText(f"Loaded {len(timestamps)} frames. Total Duration: {t_duration:.1f} s. Spectral Bins: {len(freqs)}")
            
            # Init Scrub
            self.slider_time.setEnabled(True)
            self.slider_time.setRange(0, len(timestamps) - 1)
            
            # Will automatically trigger update_spectrum_for_index on change
            self.time_line.setValue(t_start + t_duration / 2.0)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.lbl_info.setText(f"Error loading NPZ: {e}")

    def on_time_scrub(self, line):
        if self.loaded_data is None: return
        t_val = line.value()
        
        # Find closest time index
        idx = (np.abs(self.loaded_data['t'] - t_val)).argmin()
        actual_t = self.loaded_data['t'][idx]
        
        # Update Slider silently
        self.slider_time.blockSignals(True)
        self.slider_time.setValue(idx)
        self.slider_time.blockSignals(False)
        
        self.update_spectrum_for_index(idx, actual_t)

    def on_slider_scrub(self, value):
        if self.loaded_data is None: return
        idx = value
        actual_t = self.loaded_data['t'][idx]
        
        # Update time line silently
        self.time_line.blockSignals(True)
        self.time_line.setValue(actual_t)
        self.time_line.blockSignals(False)
        
        self.update_spectrum_for_index(idx, actual_t)

    def update_spectrum_for_index(self, idx, t_val):
        # t_val here is relative. Add offset back to display local time label!
        abs_t = self.date_axis_spl.offset + t_val
        dt = datetime.datetime.fromtimestamp(abs_t)
        time_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        if self.current_lang == 'ZH':
            self.lbl_slider_time.setText(f"本地时间: {time_str}")
        else:
            self.lbl_slider_time.setText(f"Local Time: {time_str}")

        freqs = self.loaded_data['f']
        valid_idx = freqs > 0
        log_f = np.log10(freqs[valid_idx])
        spl_slice = self.loaded_data['s'][idx][valid_idx]
        
        self.curve_spec.setData(log_f, spl_slice)

if __name__ == "__main__":
    try:
        from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor
        from PyQt5.QtWidgets import QSplashScreen
        import time
        
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        pixmap = QPixmap(1200, 675)
        pixmap.fill(QColor("#0d0d0d"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("#e0e0e0"))
        painter.setFont(QFont("Microsoft YaHei", 32, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "UMIK-1 Spectrum Data Viewer\n\nLoading Data Core...")
        painter.end()
        
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()
        
        time.sleep(1.2)
        
        viewer = UmikDataViewer()
        viewer.show()
        splash.finish(viewer)
        
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        traceback.print_exc()
