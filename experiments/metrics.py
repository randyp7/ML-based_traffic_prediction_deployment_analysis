"""
Measurement infrastructure for the benchmark system.

Contains:
- PhaseTimer: context manager for wall-clock, CPU time, and memory tracking
- CPUMonitor: background thread for CPU utilisation, frequency, temperature, power
- GPUMonitor: background thread for GPU utilisation via nvidia-smi
- get_system_info(): platform metadata collection

CPUMonitor and GPUMonitor are extracted from model/benchmark_runner.py.
"""

import os
import time
import platform
import subprocess
import tracemalloc
import threading

import numpy as np
import psutil

from .config import MONITOR_INTERVAL


# ============================================================================
# PhaseTimer — context manager for per-phase timing and memory
# ============================================================================

class PhaseTimer:
    """
    Context manager that measures wall-clock time, CPU time, and peak memory
    for a benchmark phase.

    Usage:
        with PhaseTimer('training') as timer:
            # ... training code ...
        print(timer.wall_clock_sec, timer.peak_memory_mb)
    """

    def __init__(self, phase_name):
        self.phase_name = phase_name
        self.wall_clock_sec = 0.0
        self.cpu_time_sec = 0.0
        self.peak_memory_mb = 0.0
        self.rss_before_mb = 0.0
        self.rss_after_mb = 0.0
        self.gpu_memory_peak_mb = None

    def __enter__(self):
        # RSS before
        self.rss_before_mb = psutil.Process().memory_info().rss / (1024 ** 2)

        # Start tracemalloc for Python-level memory tracking
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()

        # Reset GPU memory stats if available
        self._reset_gpu_memory()

        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wall_clock_sec = round(time.perf_counter() - self._wall_start, 4)
        self.cpu_time_sec = round(time.process_time() - self._cpu_start, 4)

        # tracemalloc peak
        _, peak = tracemalloc.get_traced_memory()
        self.peak_memory_mb = round(peak / (1024 ** 2), 2)
        tracemalloc.stop()

        # RSS after
        self.rss_after_mb = round(psutil.Process().memory_info().rss / (1024 ** 2), 2)

        # GPU memory peak
        self.gpu_memory_peak_mb = self._get_gpu_memory_peak()

        return False  # don't suppress exceptions

    def _reset_gpu_memory(self):
        """Reset TensorFlow GPU memory stats if GPU is available."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.reset_memory_stats('GPU:0')
        except Exception:
            pass

    def _get_gpu_memory_peak(self):
        """Get TensorFlow GPU peak memory if available."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info = tf.config.experimental.get_memory_info('GPU:0')
                return round(info.get('peak', 0) / (1024 ** 2), 2)
        except Exception:
            pass
        return None

    def to_dict(self):
        """Return phase metrics as a dictionary."""
        result = {
            "wall_clock_sec": self.wall_clock_sec,
            "cpu_time_sec": self.cpu_time_sec,
            "peak_memory_mb": self.peak_memory_mb,
            "rss_before_mb": round(self.rss_before_mb, 2),
            "rss_after_mb": self.rss_after_mb,
        }
        if self.gpu_memory_peak_mb is not None:
            result["gpu_memory_peak_mb"] = self.gpu_memory_peak_mb
        return result


# ============================================================================
# CPUMonitor — background CPU monitoring
# Extracted from model/benchmark_runner.py
# ============================================================================

class CPUMonitor:
    """Monitor CPU utilization, frequency, temperature, and power during training."""

    def __init__(self, interval=MONITOR_INTERVAL):
        self.interval = interval
        self.system_cpu_readings = []
        self.process_cpu_readings = []
        self.frequency_readings = []
        self.temperature_readings = []
        self.power_readings = []
        self._stop_event = threading.Event()
        self._thread = None
        self._process = psutil.Process()
        self._wmi = None
        self._wmi_cimv2 = None
        self._temp_available = False
        self._power_available = False
        self._power_method = None
        self._rapl_path = None
        self._libre_hardware_monitor = None

        # Windows: try WMI for temperature
        if platform.system() == 'Windows':
            try:
                import wmi
                self._wmi = wmi.WMI(namespace=r'root\wmi')
                self._wmi_cimv2 = wmi.WMI()
                try:
                    temps = self._wmi.MSAcpi_ThermalZoneTemperature()
                    if temps:
                        self._temp_available = True
                except Exception:
                    pass
            except Exception:
                pass
            self._init_libre_hardware_monitor()

        # Linux: try RAPL for power
        if platform.system() == 'Linux':
            rapl_paths = [
                '/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj',
                '/sys/class/powercap/intel-rapl:0/energy_uj',
            ]
            for path in rapl_paths:
                if os.path.exists(path):
                    self._rapl_path = path
                    self._power_available = True
                    self._power_method = 'RAPL'
                    self._last_energy_uj = None
                    self._last_energy_time = None
                    break

    def _init_libre_hardware_monitor(self):
        """Initialize LibreHardwareMonitor for accurate CPU power on Windows."""
        try:
            import clr
            dll_paths = [
                r'C:\Program Files\LibreHardwareMonitor\LibreHardwareMonitorLib.dll',
                r'C:\LibreHardwareMonitor\LibreHardwareMonitorLib.dll',
                os.path.join(os.path.dirname(__file__), 'LibreHardwareMonitorLib.dll'),
            ]
            dll_found = None
            for path in dll_paths:
                if os.path.exists(path):
                    dll_found = path
                    break

            if dll_found:
                clr.AddReference(dll_found)
                from LibreHardwareMonitor.Hardware import Computer, HardwareType, SensorType
                self._lhm_computer = Computer()
                self._lhm_computer.IsCpuEnabled = True
                self._lhm_computer.Open()
                self._lhm_HardwareType = HardwareType
                self._lhm_SensorType = SensorType
                self._libre_hardware_monitor = True
                self._power_available = True
                self._power_method = 'LibreHardwareMonitor'
            else:
                self._power_method = 'TDP_estimation'
        except ImportError:
            self._power_method = 'TDP_estimation'
        except Exception:
            self._power_method = 'TDP_estimation'

    def _get_cpu_power_libre(self):
        """Get CPU power using LibreHardwareMonitor (Windows)."""
        if not self._libre_hardware_monitor:
            return None
        try:
            for hardware in self._lhm_computer.Hardware:
                if hardware.HardwareType == self._lhm_HardwareType.Cpu:
                    hardware.Update()
                    for sensor in hardware.Sensors:
                        if sensor.SensorType == self._lhm_SensorType.Power:
                            if 'Package' in sensor.Name or 'CPU Package' in sensor.Name:
                                if sensor.Value is not None:
                                    return float(sensor.Value)
                    total_power = 0
                    for sensor in hardware.Sensors:
                        if sensor.SensorType == self._lhm_SensorType.Power:
                            if sensor.Value is not None:
                                total_power += float(sensor.Value)
                    if total_power > 0:
                        return total_power
        except Exception:
            pass
        return None

    def _get_cpu_temperature(self):
        """Get CPU temperature if available."""
        if self._wmi and self._temp_available:
            try:
                temps = self._wmi.MSAcpi_ThermalZoneTemperature()
                if temps:
                    return (temps[0].CurrentTemperature / 10) - 273.15
            except Exception:
                pass

        if platform.system() == 'Linux':
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name in ['coretemp', 'k10temp', 'cpu_thermal', 'acpitz']:
                        if name in temps:
                            return temps[name][0].current
            except Exception:
                pass

        return None

    def _get_cpu_power(self):
        """Get CPU power consumption in watts if available."""
        # Linux RAPL
        if self._rapl_path and platform.system() == 'Linux':
            try:
                with open(self._rapl_path, 'r') as f:
                    energy_uj = int(f.read().strip())
                current_time = time.time()

                if self._last_energy_uj is not None and self._last_energy_time is not None:
                    energy_diff = energy_uj - self._last_energy_uj
                    time_diff = current_time - self._last_energy_time
                    if energy_diff < 0:
                        energy_diff += 2 ** 32
                    if time_diff > 0:
                        power_watts = (energy_diff / time_diff) / 1_000_000
                        self._last_energy_uj = energy_uj
                        self._last_energy_time = current_time
                        return power_watts

                self._last_energy_uj = energy_uj
                self._last_energy_time = current_time
            except Exception:
                pass

        # Windows: LibreHardwareMonitor
        if platform.system() == 'Windows' and self._libre_hardware_monitor:
            power = self._get_cpu_power_libre()
            if power is not None:
                return power

        # Windows fallback: TDP estimation
        if platform.system() == 'Windows' and self.system_cpu_readings:
            try:
                cpu_pct = self.system_cpu_readings[-1] if self.system_cpu_readings else 0
                idle_power = 15
                max_power = 115
                estimated_power = idle_power + (max_power - idle_power) * (cpu_pct / 100)
                return estimated_power
            except Exception:
                pass

        return None

    def _monitor(self):
        """Background thread to collect CPU readings."""
        self._process.cpu_percent()
        while not self._stop_event.is_set():
            system_cpu = psutil.cpu_percent(interval=self.interval)
            self.system_cpu_readings.append(system_cpu)

            try:
                process_cpu = self._process.cpu_percent()
                self.process_cpu_readings.append(process_cpu)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            try:
                freq = psutil.cpu_freq()
                if freq:
                    self.frequency_readings.append(freq.current)
            except Exception:
                pass

            temp = self._get_cpu_temperature()
            if temp is not None:
                self.temperature_readings.append(temp)

            power = self._get_cpu_power()
            if power is not None:
                self.power_readings.append(power)

    def start(self):
        """Start CPU monitoring."""
        self.system_cpu_readings = []
        self.process_cpu_readings = []
        self.frequency_readings = []
        self.temperature_readings = []
        self.power_readings = []
        self._stop_event.clear()
        self._process = psutil.Process()
        if self._rapl_path:
            self._last_energy_uj = None
            self._last_energy_time = None
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop CPU monitoring and return statistics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

        results = {}

        if self.system_cpu_readings:
            results.update({
                'cpu_system_avg_pct': round(np.mean(self.system_cpu_readings), 2),
                'cpu_system_max_pct': round(np.max(self.system_cpu_readings), 2),
                'cpu_system_min_pct': round(np.min(self.system_cpu_readings), 2),
                'cpu_system_readings_count': len(self.system_cpu_readings),
            })

        if self.process_cpu_readings:
            results.update({
                'cpu_process_avg_pct': round(np.mean(self.process_cpu_readings), 2),
                'cpu_process_max_pct': round(np.max(self.process_cpu_readings), 2),
                'cpu_process_min_pct': round(np.min(self.process_cpu_readings), 2),
            })

        if self.frequency_readings:
            results.update({
                'cpu_freq_avg_mhz': round(np.mean(self.frequency_readings), 0),
                'cpu_freq_max_mhz': round(np.max(self.frequency_readings), 0),
                'cpu_freq_min_mhz': round(np.min(self.frequency_readings), 0),
            })

        if self.temperature_readings:
            results.update({
                'cpu_temp_avg_c': round(np.mean(self.temperature_readings), 1),
                'cpu_temp_max_c': round(np.max(self.temperature_readings), 1),
                'cpu_temp_min_c': round(np.min(self.temperature_readings), 1),
            })

        if self.power_readings:
            results.update({
                'cpu_power_avg_w': round(np.mean(self.power_readings), 1),
                'cpu_power_max_w': round(np.max(self.power_readings), 1),
                'cpu_power_min_w': round(np.min(self.power_readings), 1),
            })
            total_energy_wh = (np.sum(self.power_readings) * self.interval) / 3600
            results['cpu_energy_wh'] = round(total_energy_wh, 3)

        if self._power_method:
            results['cpu_power_method'] = self._power_method

        return results


# ============================================================================
# GPUMonitor — background GPU monitoring via nvidia-smi
# Extracted from model/benchmark_runner.py
# ============================================================================

class GPUMonitor:
    """Monitor NVIDIA GPU utilization and memory during training using nvidia-smi."""

    def __init__(self, interval=MONITOR_INTERVAL):
        self.interval = interval
        self.gpu_util_readings = []
        self.gpu_mem_readings = []
        self.gpu_mem_used_readings = []
        self.gpu_temp_readings = []
        self.gpu_power_readings = []
        self._stop_event = threading.Event()
        self._thread = None
        self._gpu_available = self._check_gpu()

    def _check_gpu(self):
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _query_gpu(self):
        """Query GPU metrics using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(',')
                if len(values) >= 6:
                    return {
                        'gpu_util': float(values[0].strip()),
                        'mem_util': float(values[1].strip()),
                        'mem_used': float(values[2].strip()),
                        'mem_total': float(values[3].strip()),
                        'temperature': float(values[4].strip()),
                        'power': float(values[5].strip()) if values[5].strip() != '[N/A]' else None,
                    }
        except Exception:
            pass
        return None

    def _monitor(self):
        """Background thread to collect GPU readings."""
        while not self._stop_event.is_set():
            metrics = self._query_gpu()
            if metrics:
                self.gpu_util_readings.append(metrics['gpu_util'])
                self.gpu_mem_readings.append(metrics['mem_util'])
                self.gpu_mem_used_readings.append(metrics['mem_used'])
                self.gpu_temp_readings.append(metrics['temperature'])
                if metrics['power'] is not None:
                    self.gpu_power_readings.append(metrics['power'])
            time.sleep(self.interval)

    def start(self):
        """Start GPU monitoring."""
        if not self._gpu_available:
            return
        self.gpu_util_readings = []
        self.gpu_mem_readings = []
        self.gpu_mem_used_readings = []
        self.gpu_temp_readings = []
        self.gpu_power_readings = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop GPU monitoring and return statistics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

        results = {'gpu_available': self._gpu_available}

        if not self._gpu_available or not self.gpu_util_readings:
            return results

        results.update({
            'gpu_util_avg_pct': round(np.mean(self.gpu_util_readings), 2),
            'gpu_util_max_pct': round(np.max(self.gpu_util_readings), 2),
            'gpu_util_min_pct': round(np.min(self.gpu_util_readings), 2),
            'gpu_readings_count': len(self.gpu_util_readings),
        })

        if self.gpu_mem_readings:
            results.update({
                'gpu_mem_util_avg_pct': round(np.mean(self.gpu_mem_readings), 2),
                'gpu_mem_util_max_pct': round(np.max(self.gpu_mem_readings), 2),
            })

        if self.gpu_mem_used_readings:
            results.update({
                'gpu_mem_used_avg_mb': round(np.mean(self.gpu_mem_used_readings), 2),
                'gpu_mem_used_max_mb': round(np.max(self.gpu_mem_used_readings), 2),
                'gpu_mem_used_min_mb': round(np.min(self.gpu_mem_used_readings), 2),
            })

        if self.gpu_temp_readings:
            results.update({
                'gpu_temp_avg_c': round(np.mean(self.gpu_temp_readings), 1),
                'gpu_temp_max_c': round(np.max(self.gpu_temp_readings), 1),
            })

        if self.gpu_power_readings:
            results.update({
                'gpu_power_avg_w': round(np.mean(self.gpu_power_readings), 1),
                'gpu_power_max_w': round(np.max(self.gpu_power_readings), 1),
            })

        return results


# ============================================================================
# System Information
# Extracted from model/benchmark_runner.py
# ============================================================================

def get_system_info():
    """Collect system information for the benchmark."""
    info = {
        'platform_name': platform.node(),
        'os': platform.system(),
        'os_version': platform.version(),
        'cpu_cores_physical': psutil.cpu_count(logical=False),
        'cpu_cores_logical': psutil.cpu_count(logical=True),
        'ram_gb': round(psutil.virtual_memory().total / (1024 ** 3), 2),
        'python_version': platform.python_version(),
    }

    # CPU model
    try:
        if platform.system() == 'Windows':
            result = subprocess.run(['wmic', 'cpu', 'get', 'name'],
                                    capture_output=True, text=True, timeout=10)
            cpu_name = result.stdout.strip().split('\n')[-1].strip()
            info['cpu_model'] = cpu_name if cpu_name else platform.processor()
        elif platform.system() == 'Linux':
            result = subprocess.run(['cat', '/proc/cpuinfo'],
                                    capture_output=True, text=True, timeout=10)
            for line in result.stdout.split('\n'):
                if 'model name' in line:
                    info['cpu_model'] = line.split(':')[1].strip()
                    break
        else:
            info['cpu_model'] = platform.processor()
    except Exception:
        info['cpu_model'] = platform.processor()

    # GPU names
    try:
        if platform.system() == 'Windows':
            result = subprocess.run(['wmic', 'path', 'win32_videocontroller', 'get', 'name'],
                                    capture_output=True, text=True, timeout=10)
            gpu_lines = [line.strip() for line in result.stdout.strip().split('\n')
                         if line.strip() and line.strip() != 'Name']
            info['gpu_names'] = gpu_lines
        elif platform.system() == 'Linux':
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                info['gpu_names'] = [line.strip() for line in result.stdout.strip().split('\n')]
            else:
                info['gpu_names'] = []
    except Exception:
        info['gpu_names'] = []

    # GPU memory (NVIDIA only)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            memory_mb = int(result.stdout.strip().split('\n')[0])
            info['gpu_memory_mb'] = memory_mb
            info['gpu_memory_gb'] = round(memory_mb / 1024, 1)
    except Exception:
        pass

    # TensorFlow / GPU info
    try:
        import tensorflow as tf
        info['tensorflow_version'] = tf.__version__
        gpus = tf.config.list_physical_devices('GPU')
        info['tf_gpu_available'] = len(gpus) > 0
        info['tf_gpu_count'] = len(gpus)
        if gpus:
            info['tf_gpu_devices'] = [g.name for g in gpus]
    except ImportError:
        info['tensorflow_version'] = 'not installed'
        info['tf_gpu_available'] = False

    return info
