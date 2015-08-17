__author__ = 'alansanders'

import numpy as np
import numpy.ma as ma
from nplab.utils.gui import *
from PyQt4 import uic
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from nplab.ui.ui_tools import UiTools
import h5py
from multiprocessing.pool import ThreadPool
from threading import Thread
import time
import h5py
import os
import inspect
import datetime
from nplab.instrument import Instrument


class Spectrometer(Instrument):

    metadata_property_names = ('model_name', 'serial_number', 'integration_time',
                               'reference', 'background', 'wavelengths')

    def __init__(self):
        super(Spectrometer, self).__init__()
        self._model_name = None
        self._serial_number = None
        self._wavelengths = None
        self.reference = None
        self.background = None
        self.latest_raw_spectrum = None
        self.latest_spectrum = None
        self._config_file = None

    def __del__(self):
        try:
            self._config_file.close()
        except AttributeError:
            pass #if it's not present, we get an exception - which doesn't matter.

    def open_config_file(self):
        """Open the config file for the current spectrometer and return it, creating if it's not there"""
        if self._config_file is None:
            f = inspect.getfile(self.__class__)
            d = os.path.dirname(f)
            self._config_file = h5py.File(os.path.join(d, 'config.h5'))
            self._config_file.attrs['date'] = datetime.datetime.now().strftime("%H:%M %d/%m/%y")
        return self._config_file

    config_file = property(open_config_file)

    def update_config(self, name, data):
        f = self.config_file
        if name not in f:
            f.create_dataset(name, data=data)
        else:
            dset = f[name]
            dset[:] = data
            f.flush()

    def get_model_name(self):
        if self._model_name is None:
            self._model_name = 'model_name'
        return self._model_name

    model_name = property(get_model_name)

    def get_serial_number(self):
        if self._serial_number is None:
            self._serial_number = 'serial_number'
        return self._serial_number

    serial_number = property(get_serial_number)

    def get_integration_time(self):
        return 0

    def set_integration_time(self, value):
        print 'setting 0'

    integration_time = property(get_integration_time, set_integration_time)

    def get_wavelengths(self):
        if self._wavelengths is None:
            self._wavelengths = np.arange(400,1200,1)
        return self._wavelengths

    wavelengths = property(get_wavelengths)

    def read_spectrum(self):
        self.latest_raw_spectrum = np.zeros(0)
        return self.latest_raw_spectrum

    def read_background(self):
        """Acquire a new spectrum and use it as a background measurement."""
        self.background = self.read_spectrum()
        self.update_config('background', self.background)

    def clear_background(self):
        """Clear the current background reading."""
        self.background = None

    def read_reference(self):
        """Acquire a new spectrum and use it as a reference."""
        self.reference = self.read_spectrum()
        self.update_config('reference', self.reference)

    def clear_reference(self):
        """Clear the current reference spectrum"""
        self.reference = None

    def is_background_compensated(self):
        """Return whether there's currently a valid background spectrum"""
        return len(self.background)==len(self.latest_raw_spectrum) and \
            sum(self.background)>0

    def is_referenced(self):
        """Check whether there's currently a valid background and reference spectrum"""
        return self.is_background_compensated and \
            len(self.reference)==len(self.latest_raw_spectrum) and \
            sum(self.reference)>0

    def process_spectrum(self, spectrum):
        """Subtract the background and divide by the reference, if possible"""
        if self.background is not None:
            if self.reference is not None:
                old_error_settings = np.seterr(all='ignore')
                new_spectrum = (spectrum - self.background)/(self.reference - self.background)
                np.seterr(**old_error_settings)
                new_spectrum[np.isinf(new_spectrum)] = np.NaN #if the reference is nearly 0, we get infinities - just make them all NaNs.
            else:
                new_spectrum = spectrum - self.background
        else:
            new_spectrum = spectrum
        return new_spectrum

    def read_processed_spectrum(self):
        """Acquire a new spectrum and return a processed (referenced/background-subtracted) spectrum.
        
        NB if saving data to file, it's best to save raw spectra along with metadata - this is a
        convenience method for display purposes."""
        spectrum = self.read_spectrum()
        self.latest_spectrum = self.process_spectrum(spectrum)
        return self.latest_spectrum

    def read(self):
        """Acquire a new spectrum and return a tuple of wavelengths, spectrum"""
        return self.wavelengths, self.read_processed_spectrum()

    def mask_spectrum(self, spectrum, threshold):
        """Return a masked array of the spectrum, showing only points where the reference
        is bright enough to be useful."""
        if self.reference is not None and self.background is not None:
            reference = self.reference - self.background
            mask = reference < reference.max() * threshold
            if len(spectrum.shape)>1:
                mask = np.tile(mask, spectrum.shape[:-1]+(1,))
            return ma.array(spectrum, mask=mask)
        else:
            return spectrum

    def get_qt_ui(self, control_only=False):
        """Create a Qt interface for the spectrometer"""
        if control_only:
            return SpectrometerControlUI(self)
        else:
            return SpectrometerUI(self)

    def save_spectrum(self, spectrum=None, attrs={}):
        """Save a spectrum to the current datafile, creating if necessary.
        
        If no spectrum is passed in, a new spectrum is taken.  The convention
        is to save raw spectra only, along with reference/background to allow
        later processing.
        
        The attrs dictionary allows extra metadata to be saved in the HDF5 file."""
        spectrum = self.read_processed_spectrum() if spectrum is None else spectrum
        metadata = self.metadata
        metadata.update(attrs) #allow extra metadata to be passed in
        self.create_dataset("spectrum", data=spectrum, attrs=metadata) 
        #save data in the default place (see nplab.instrument.Instrument)

    def save_reference_to_file(self):
        pass

    def load_reference_from_file(self):
        pass


class Spectrometers(Instrument):
    def __init__(self, spectrometer_list):
        assert False not in [isinstance(s, Spectrometer) for s in spectrometer_list],\
            'an invalid spectrometer was supplied'
        super(Spectrometers, self).__init__()
        self.spectrometers = spectrometer_list
        self.num_spectrometers = len(spectrometer_list)
        self._pool = ThreadPool(processes=self.num_spectrometers)
        self._wavelengths = None

    def __del__(self):
        self._pool.close()

    def add_spectrometer(self, spectrometer):
        assert isinstance(spectrometer, Spectrometer), 'spectrometer must be an instance of Spectrometer'
        if spectrometer not in self.spectrometers:
            self.spectrometers.append(spectrometer)
            self.num_spectrometers = len(self.spectrometers)

    def get_wavelengths(self):
        if self._wavelengths is None:
            self._wavelengths = [s.wavelengths for s in self.spectrometers]
        return self._wavelengths

    wavelengths = property(get_wavelengths)

    def read_spectra(self):
        """Acquire spectra from all spectrometers and return as a list."""
        return self._pool.map(lambda s: s.read_spectrum(), self.spectrometers)

    def read_processed_spectra(self):
        """Acquire a list of processed (referenced, background subtracted) spectra."""
        return self._pool.map(lambda s: s.read_processed_spectrum(), self.spectrometers)

    def process_spectra(self, spectra):
        pairs = zip(self.spectrometers, spectra)
        return self._pool.map(lambda (s, spectrum): s.process_spectrum(spectrum), pairs)

    def get_metadata_list(self):
        """Return a list of metadata for each spectrometer."""
        return self._pool.map(lambda s: s.get_metadata(), self.spectrometers)

    def mask_spectra(self, spectra, threshold):
        return [spectrometer.mask_spectrum(spectrum, threshold) for (spectrometer, spectrum) in zip(self.spectrometers, spectra)]

    def get_qt_ui(self):
        return SpectrometersUI(self)

    def save_spectra(self, spectra=None, attrs={}):
        """Save spectra from all the spectrometers, in a folder in the current
        datafile, creating the file if needed.

        If no spectra are given, new ones are acquired - NB you should pass
        raw spectra in - metadata will be saved along with the spectra.
        """
        spectra = self.read_processed_spectra() if spectra is None else spectra
        metadata_list = self.get_metadata_list()
        g = self.create_data_group('spectra',attrs=attrs) # create a uniquely numbered group in the default place
        for spectrum,metadata in zip(spectra,metadata_list):
            g.create_dataset('spectrum_%d',data=spectrum,attrs=metadata)
            
    def get_metadata(self):
        """
        Returns a list of dictionaries containing relevant spectrometer properties
        for each spectrometer.
        """
        return [spectrometer.metadata for spectrometer in self.spectrometers]

    metadata = property(get_metadata)


controls_base, controls_widget = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'spectrometer_controls.ui'))
display_base, display_widget = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'spectrometer_view.ui'))


class SpectrometerControlUI(UiTools, controls_base, controls_widget):
    def __init__(self, spectrometer, parent=None):
        assert isinstance(spectrometer, Spectrometer), "instrument must be a Spectrometer"
        super(SpectrometerControlUI, self).__init__()
        self.spectrometer = spectrometer
        self.setupUi(self)

        self.integration_time.setValidator(QtGui.QDoubleValidator())
        self.integration_time.textChanged.connect(self.check_state)
        self.integration_time.textChanged.connect(self.update_param)

        self.read_background_button.clicked.connect(self.button_pressed)
        self.read_reference_button.clicked.connect(self.button_pressed)
        self.clear_background_button.clicked.connect(self.button_pressed)
        self.clear_reference_button.clicked.connect(self.button_pressed)
        self.load_state_button.clicked.connect(self.button_pressed)

        self.background_subtracted.stateChanged.connect(self.state_changed)
        self.referenced.stateChanged.connect(self.state_changed)

        self.id_string.setText('{0} {1}'.format(self.spectrometer.model_name, self.spectrometer.serial_number))
        self.id_string.resize(self.id_string.sizeHint())

        self.integration_time.setText(str(spectrometer.integration_time))

    def update_param(self, *args, **kwargs):
        sender = self.sender()
        if sender is self.integration_time:
            try:
                self.spectrometer.integration_time = float(args[0])
            except ValueError:
                pass

    def button_pressed(self, *args, **kwargs):
        sender = self.sender()
        if sender is self.read_background_button:
            self.spectrometer.read_background()
            self.background_subtracted.blockSignals(True)
            self.background_subtracted.setCheckState(QtCore.Qt.Checked)
            self.background_subtracted.blockSignals(False)
        elif sender is self.clear_background_button:
            self.spectrometer.clear_background()
            self.background_subtracted.blockSignals(True)
            self.background_subtracted.setCheckState(QtCore.Qt.Unchecked)
            self.background_subtracted.blockSignals(False)
        elif sender is self.read_reference_button:
            self.spectrometer.read_reference()
            self.referenced.blockSignals(True)
            self.referenced.setCheckState(QtCore.Qt.Checked)
            self.referenced.blockSignals(False)
        elif sender is self.clear_reference_button:
            self.spectrometer.clear_reference()
            self.referenced.blockSignals(True)
            self.referenced.setCheckState(QtCore.Qt.Unchecked)
            self.referenced.blockSignals(False)
        elif sender is self.load_state_button:
            if 'background' in self.spectrometer.config_file:
                self.spectrometer.background = self.spectrometer.config_file['background'][:]
                self.background_subtracted.blockSignals(True)
                self.background_subtracted.setCheckState(QtCore.Qt.Checked)
                self.background_subtracted.blockSignals(False)
            else:
                print 'background not found in config file'
            if 'reference' in self.spectrometer.config_file:
                self.spectrometer.reference = self.spectrometer.config_file['reference'][:]
                self.referenced.blockSignals(True)
                self.referenced.setCheckState(QtCore.Qt.Checked)
                self.referenced.blockSignals(False)
            else:
                print 'reference not found in config file'

    def state_changed(self, state):
        sender = self.sender()
        if sender is self.background_subtracted and state == QtCore.Qt.Checked:
            self.spectrometer.read_background()
        elif sender is self.background_subtracted and state == QtCore.Qt.Unchecked:
            self.spectrometer.clear_background()
        if sender is self.referenced and state == QtCore.Qt.Checked:
            self.spectrometer.read_reference()
        elif sender is self.referenced and state == QtCore.Qt.Unchecked:
            self.spectrometer.clear_reference()


class DisplayThread(QtCore.QThread):
    spectrum_ready = QtCore.pyqtSignal(np.ndarray)
    spectra_ready = QtCore.pyqtSignal(list)

    def __init__(self, parent):
        super(DisplayThread, self).__init__()
        self.parent = parent
        self.single_shot = False
        self.refresh_rate = 30.

    def run(self):
        t0 = time.time()
        while self.parent.live_button.isChecked() or self.single_shot:
            read_processed_spectrum = self.parent.spectrometer.read_processed_spectra \
                if isinstance(self.parent.spectrometer, Spectrometers) \
                else self.parent.spectrometer.read_processed_spectrum
            spectrum = read_processed_spectrum()
            if time.time()-t0 < 1./self.refresh_rate:
                continue
            else:
                t0 = time.time()
            if type(spectrum) == np.ndarray:
                self.spectrum_ready.emit(spectrum)
            elif type(spectrum) == list:
                self.spectra_ready.emit(spectrum)
            if self.single_shot:
                break
        self.finished.emit()


class SpectrometerDisplayUI(UiTools, display_base, display_widget):
    def __init__(self, spectrometer, parent=None):
        assert isinstance(spectrometer, Spectrometer) or isinstance(spectrometer, Spectrometers),\
            "instrument must be a Spectrometer or an instance of Spectrometers"
        super(SpectrometerDisplayUI, self).__init__()
        if isinstance(spectrometer, Spectrometers) and spectrometer.num_spectrometers == 1:
            spectrometer = spectrometer.spectrometers[0]
        self.spectrometer = spectrometer
        print self.spectrometer
        self.setupUi(self)
        self.fig = Figure()
        self.figure_widget = self.replace_widget(self.display_layout,
                                                 self.figure_widget, FigureCanvas(self.fig))

        self.take_spectrum_button.clicked.connect(self.button_pressed)
        self.live_button.clicked.connect(self.button_pressed)
        self.save_button.clicked.connect(self.button_pressed)
        self.threshold.setValidator(QtGui.QDoubleValidator())
        self.threshold.textChanged.connect(self.check_state)

        for text_field in [self.x_max, self.x_min, self.y_max, self.y_min]:
            text_field.setValidator(QtGui.QDoubleValidator())
            text_field.textChanged.connect(self.update_limits)
        for checkbox in [self.autoscale_x, self.autoscale_y]:
            checkbox.stateChanged.connect(self.update_limits)

        #self._display_thread = Thread(target=self.update_spectrum)
        self._display_thread = DisplayThread(self)
        self._display_thread.spectrum_ready.connect(self.update_display)
        self._display_thread.spectra_ready.connect(self.update_display)

        self.period = 0.2

    def button_pressed(self, *args, **kwargs):
        sender = self.sender()
        if sender is self.take_spectrum_button:
            #if self._display_thread.is_alive():
            if self._display_thread.isRunning():
                print 'already acquiring'
                return
            #self._display_thread = Thread(target=self.update_spectrum)
            self._display_thread.single_shot = True
            self._display_thread.start()
            #self.update_spectrum()
        elif sender is self.save_button:
            save_spectrum = self.spectrometer.save_spectra \
                if isinstance(self.spectrometer, Spectrometers) \
                else self.spectrometer.save_spectrum
            save_spectrum(attrs={'description':str(self.description.text())})
        elif sender is self.live_button:
            if self.live_button.isChecked():
                #if self._display_thread.is_alive():
                if self._display_thread.isRunning():
                    print 'already acquiring'
                    return
                #self._display_thread = Thread(target=self.continuously_update_spectrum)
                self._display_thread.single_shot = False
                self._display_thread.start()

    def update_spectrum(self):
        read_processed_spectrum = self.spectrometer.read_processed_spectra \
            if isinstance(self.spectrometer, Spectrometers) \
            else self.spectrometer.read_processed_spectrum
        spectrum = read_processed_spectrum()
        self.update_display(spectrum)

    def update_limits(self, *args, **kwargs):
        """Handle autoscaling/limit related parameter changes"""
        try:
            for ax in self.fig.axes:
                if self.autoscale_x.checkState():
                    ax.set_xlim(auto=True)
                else:
                    ax.set_xlim(float(self.x_min.text()),float(self.x_max.text()))
                if self.autoscale_y.checkState():
                    ax.set_ylim(auto=True)
                else:
                    ax.set_ylim(float(self.y_min.text()),float(self.y_max.text()))
        except:
            print "Uh oh, something went wrong setting the graph limits."


    def continuously_update_spectrum(self):
        t0 = time.time()
        while self.live_button.isChecked():
            if time.time()-t0 < 1./30.:
                continue
            else:
                t0 = time.time()
            self.update_spectrum()

    def update_display(self, spectrum):
        if self.enable_threshold.checkState() == QtCore.Qt.Checked:
            threshold = float(self.threshold.text())
            if isinstance(self.spectrometer, Spectrometers):
                spectrum = [spectrometer.mask_spectrum(s, threshold) for (spectrometer, s) in zip(self.spectrometer.spectrometers, spectrum)]
            else:
                spectrum = self.spectrometer.mask_spectrum(spectrum, threshold)
        if not self.fig.axes:
            if isinstance(self.spectrometer, Spectrometers):
                ax = self.fig.add_subplot(111)
                ax.plot(self.spectrometer.wavelengths[0], spectrum[0], 'r-')
                ax2 = ax.twinx()
                ax2.plot(self.spectrometer.wavelengths[1], spectrum[1], 'b-')
            else:
                ax = self.fig.add_subplot(111)
                ax.plot(self.spectrometer.wavelengths, spectrum, 'r-')
            self.update_limits()
        else:
            if isinstance(self.spectrometer, Spectrometers):
                for i, axis in enumerate(self.fig.axes):
                    l, = axis.lines
                    l.set_data(self.spectrometer.wavelengths[i], spectrum[i])
                    axis.relim()
                    axis.autoscale_view()
            else:
                ax, = self.fig.axes
                l, = ax.lines
                l.set_data(self.spectrometer.wavelengths, spectrum)
                ax.relim()
                ax.autoscale_view()
        self.fig.canvas.draw()


class SpectrometerUI(QtGui.QWidget):
    """
    Joins together the control and display UIs into a single spectrometer UI.
    """

    def __init__(self, spectrometer):
        assert isinstance(spectrometer, Spectrometer), "instrument must be a Spectrometer"
        super(SpectrometerUI, self).__init__()
        self.spectrometer = spectrometer
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle(self.spectrometer.__class__.__name__)
        self.controls = self.spectrometer.get_qt_ui(control_only=True)
        self.display = SpectrometerDisplayUI(self.spectrometer)
        layout = QtGui.QVBoxLayout()
        controls_layout = QtGui.QVBoxLayout()
        controls_layout.addWidget(self.controls)
        controls_layout.setContentsMargins(0,0,0,0)
        controls_group = QtGui.QGroupBox()
        controls_group.setTitle('Spectrometer')
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        layout.addWidget(self.display)
        layout.setContentsMargins(5,5,5,5)
        layout.setSpacing(5)
        self.setLayout(layout)


class SpectrometersUI(QtGui.QWidget):
    def __init__(self, spectrometers):
        assert isinstance(spectrometers, Spectrometers), "instrument must be an instance of Spectrometers"
        super(SpectrometersUI, self).__init__()
        self.spectrometers = spectrometers
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle('Spectrometers')
        self.controls_layout = QtGui.QHBoxLayout()
        controls_group = QtGui.QGroupBox()
        controls_group.setTitle('Spectrometers')
        controls_group.setLayout(self.controls_layout)
        self.controls = []
        for spectrometer in self.spectrometers.spectrometers:
            control = spectrometer.get_qt_ui(control_only=True)
            self.controls_layout.addWidget(control)
            self.controls.append(control)
        self.display = SpectrometerDisplayUI(self.spectrometers)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(controls_group)
        layout.addWidget(self.display)
        self.setLayout(layout)


class DummySpectrometer(Spectrometer):
    def __init__(self):
        super(DummySpectrometer, self).__init__()
        self._integration_time = 10

    def get_integration_time(self):
        return self._integration_time

    def set_integration_time(self, value):
        self._integration_time = value

    integration_time = property(get_integration_time, set_integration_time)

    def get_wavelengths(self):
        return np.arange(400,1200,1)

    wavelengths = property(get_wavelengths)

    def read_spectrum(self):
        from time import sleep
        sleep(self.integration_time/1000.)
        return np.array([np.random.random() for wl in self.wavelengths])


if __name__ == '__main__':
    import sys
    from nplab.utils.gui import get_qt_app
    s1 = DummySpectrometer()
    s2 = DummySpectrometer()
    spectrometers = Spectrometers([s1, s2])
    for spectrometer in spectrometers.spectrometers:
        spectrometer.integration_time = 100
    import timeit
    print '{0:.2f} ms'.format(1000*timeit.Timer(spectrometers.read_spectra).timeit(number=10)/10)
    app = get_qt_app()
    ui = SpectrometersUI(spectrometers)
    ui.show()
    sys.exit(app.exec_())
