#!/usr/bin/env python3

import time
import random
import sys
import psutil
import re
import subprocess
import importlib
import importlib.metadata
import argparse
import math
import logging

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def validate_python_version():
    if sys.version_info >= (3, 7):
        LOGGER.info(f'Found python version {sys.version}. OK.')
        return True
    else:
        LOGGER.error(f'Found python version {sys.version} but 3.7 is required. Exiting.')
        return False

def run_for_version(name):
    process = subprocess.Popen(
        [name, '--version'],
        shell=False,
        bufsize=-1,
        universal_newlines=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    return process.stdout.read()

def validate_git():
    success = True
    try:
        out = run_for_version('git')
        parts = out.split('\n')[0].split()
        version_str = parts[-1]
        major_version = int(version_str.split('.')[0])
        success = major_version >= 2
        if success:
            LOGGER.info(f'Found git executable version {version_str}. OK.')
        else:
            LOGGER.error(f'Found git executable version {version_str} but at least 2.0 required. Exiting.')
    except:
        success = False
        LOGGER.error('No git executable found. Exiting.')

    return success

def validate_cmake():
    success = True
    try:
        out = run_for_version('cmake')
        parts = out.split('\n')[0].split()
        version_str = parts[-1]
        major_version = int(version_str.split('.')[0])
        minor_version = int(version_str.split('.')[1])
        success = (major_version, minor_version) >= (3, 4)
        if success:
            LOGGER.info(f'Found cmake executable version {version_str}. OK.')
        else:
            LOGGER.error(f'Found cmake executable version {version_str} but at least 3.4 required. Exiting.')
    except:
        success = False
        LOGGER.error('No cmake executable found. Exiting.')

    return success

def validate_make():
    success = True
    try:
        out = run_for_version('make')
        parts = out.split('\n')[0].split()
        version_str = parts[-1]
        major_version = int(version_str.split('.')[0])
        success = major_version >= 3
        if success:
            LOGGER.info(f'Found make executable version {version_str}. OK.')
        else:
            LOGGER.error(f'Found make executable version {version_str} but at least 3 required. Exiting.')
    except:
        success = False
        LOGGER.error('No make executable found. Exiting.')

    return success

def validate_gcc():
    success = True
    try:
        out = run_for_version('gcc')
        parts = out.split('\n')[0].split()
        version_str = parts[-1]
        major_version = int(version_str.split('.')[0])
        minor_version = int(version_str.split('.')[1])
        success = (major_version, minor_version) >= (9, 2)
        if success:
            LOGGER.info(f'Found gcc executable version {version_str}. OK.')
        else:
            LOGGER.error(f'Found gcc executable version {version_str} but at least 9.2 required. Exiting.')
    except:
        success = False
        LOGGER.error('No gcc executable found. Exiting.')

    return success

def maybe_int(v):
    try:
        return int(v)
    except:
        return v

class PackageInfo:
    def __init__(self, name):
        self._spec = importlib.util.find_spec(name)
        self._version_str = None
        self._version_tup = None
        try:
            if self._spec:
                self._version_str = importlib.metadata.version(name)
                self._version_tup = tuple(maybe_int(v) for v in self._version_str.split('.'))
        except:
            pass

    @property
    def exists(self):
        return self._spec is not None

    def is_version_at_least(self, desired):
        return self._version_tup and self._version_tup >= desired

    @property
    def version(self):
        return self._version_str

def validate_asciimatics():
    pkg = PackageInfo('asciimatics')
    if pkg.exists:
        LOGGER.info('Found asciimatics package. OK.')
        return True
    else:
        LOGGER.error('No asciimatics package found. Run `pip install asciimatics`. Exiting.')
        return False

def validate_pytorch():
    pkg = PackageInfo('torch')
    if pkg.exists:
        if not 'cu' in pkg.version:
            LOGGER.error(f'Found torch without CUDA but CUDA support required. Exiting')
            return False
        elif pkg.is_version_at_least((1, 8)):
            LOGGER.info(f'Found torch version {pkg.version}. OK.')
            return True
        else:
            LOGGER.error(f'Found torch version {pkg.version} but at least 1.8 required. Exiting.')
            return False
    else:
        LOGGER.error('No torch package found. Install at least torch 1.8 with cuda. See https://pytorch.org/. Exiting.')
        return False

def validate_pytorchlightning():
    pkg = PackageInfo('pytorch_lightning')
    if pkg.exists:
        LOGGER.info(f'Found pytorch_lightning version {pkg.version}. OK.')
        return True
    else:
        LOGGER.error('No pytorch_lightning found. Run `pip install pytorch-lightning`. Exiting.')
        return False

def validate_cupy():
    pkg = PackageInfo('cupy')
    if pkg.exists:
        LOGGER.info(f'Found cupy version {pkg.version}. OK.')
        return True
    else:
        LOGGER.error('No cupy found. Install cupy matching cuda version used by pytorch. See https://cupy.dev/. Exiting.')
        return False

def validate_gputil():
    pkg = PackageInfo('GPUtil')
    if pkg.exists:
        LOGGER.info(f'Found GPUtil version {pkg.version}. OK.')
        return True
    else:
        LOGGER.error('No GPUtil found. Run `pip install GPUtil`. Exiting.')
        return False

def validate_imports():
    success = True
    success &= validate_asciimatics()
    success &= validate_pytorch()
    success &= validate_pytorchlightning()
    success &= validate_cupy()
    success &= validate_gputil()
    return success

def validate_environment_requirements():
    success = True
    try:
        success &= validate_python_version()
        success &= validate_git()
        success &= validate_make()
        success &= validate_cmake()
        success &= validate_gcc()
        success &= validate_imports()
    except Exception as e:
        LOGGER.error(e)
        return False
    return success

if not validate_environment_requirements():
    sys.exit(2)

LOGGER.propagate = False

from asciimatics.widgets import Frame, ListBox, Layout, Divider, Text, Button, \
    TextBox, Widget, VerticalDivider, MultiColumnListBox, Label, PopUpDialog
from asciimatics.scene import Scene
from asciimatics.screen import Screen
from asciimatics.exceptions import ResizeScreenError, NextScene, StopApplication
from asciimatics.utilities import BoxTool
from asciimatics.constants import SINGLE_LINE, DOUBLE_LINE
from asciimatics.event import KeyboardEvent, MouseEvent
from threading import Thread, Lock, Event
import GPUtil
import io
import os
import requests
import zipfile
import urllib.request
import urllib.parse
from tqdm.auto import tqdm
from pathlib import Path

ORDO_GIT = ('michiguel/Ordo', '17eec774f2e4b9fdd2b1b38739f55ea221fb851a')
C_CHESS_CLI_GIT = ('lucasart/c-chess-cli', '6d08fee2e95b259c486b21a886f6911b61f676af')

def terminate_process_on_exit(process):
    if sys.platform == "win32":
        try:
            with open('.process_watchdog_helper.bat', 'x') as file:
                file.write(""":waitforpid
tasklist /nh /fi "pid eq %1" 2>nul | find "%1" >nul
if %ERRORLEVEL%==0 (
    timeout /t 5 /nobreak >nul
    goto :waitforpid
) else (
    wmic process where processid="%2" call terminate >nul
)""")
        except:
            pass

        subprocess.Popen(
            ['.process_watchdog_helper.bat', str(os.getpid()), str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    elif sys.platform == "linux":
        # TODO: this
        pass

class DecayingRunningAverage:
    def __init__(self, decay=0.995):
        self._decay = decay
        self._total = 0.0
        self._count = 0.0

    @property
    def decay(self):
        return self._decay

    @property
    def value(self):
        try:
            return self._total / self._count
        except:
            return float('NaN')

    def update(self, value):
        self._total = self._total * self._decay + value
        self._count = self._count * self._decay + 1.0

class SystemResources:
    def __init__(self):
        self._gpus = dict()
        for gpu in GPUtil.getGPUs():
            self._gpus[gpu.id] = gpu
        self._cpu_usage = psutil.cpu_percent() / 100.0
        mem = psutil.virtual_memory()
        self._ram_usage_mb = mem[3] // (1024 * 1024)
        self._ram_max_mb = mem[0] // (1024 * 1024)

    @property
    def gpus(self):
        return self._gpus

    @property
    def cpu_usage(self):
        return self._cpu_usage

    @property
    def ram_usage_mb(self):
        return self._ram_usage_mb

    @property
    def ram_max_mb(self):
        return self._ram_max_mb

class SystemResourcesMonitor(Thread):
    def __init__(self, period_seconds):
        super(SystemResourcesMonitor, self).__init__()

        self._period_seconds = period_seconds
        self._mutex = Lock()
        self._stop_event = Event()

        self._running = True
        self._update()
        self.start()

    def _update(self):
        self._resources = SystemResources()

    def run(self):
        while self._running:
            self._mutex.acquire()
            try:
                self._update()
            finally:
                self._mutex.release()
            self._stop_event.wait(timeout=self._period_seconds)

    @property
    def resources(self):
        self._mutex.acquire()
        try:
            return self._resources
        finally:
            self._mutex.release()

    def stop(self):
        self._running = False
        self._stop_event.set()

def find_latest_checkpoint_in_run(root_dir):
    ckpts = [file for file in Path(root_dir).rglob("*.ckpt")]
    if not ckpts:
        return None

    return str(max(ckpts, key=lambda p: p.stat().st_ctime_ns))

RESOURCE_MONITOR = SystemResourcesMonitor(2)
NUMERIC_CONST_PATTERN = '[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?'

class TrainingRun(Thread):
    ITERATION_PATTERN = re.compile(f'Epoch (\\d+).*?(\\d+)/(\\d+).*?({NUMERIC_CONST_PATTERN})it/s, loss=({NUMERIC_CONST_PATTERN})')
    def __init__(
        self,
        gpu_id,
        run_id,
        nnue_pytorch_directory,
        training_dataset,
        validation_dataset,
        num_data_loader_threads,
        num_pytorch_threads,
        num_epochs,
        batch_size,
        random_fen_skipping,
        smart_fen_skipping,
        wld_fen_skipping,
        features,
        lr,
        gamma,
        network_save_period,
        save_last_network,
        seed,
        root_dir,
        epoch_size,
        validation_size,
        resume_from_model=None,
        resume_training=False,
        additional_args=[]
    ):

        super(TrainingRun, self).__init__()
        self._gpu_id = gpu_id
        self._run_id = run_id
        # use abspaths because we will be running the script from somewhere else
        self._nnue_pytorch_directory = os.path.abspath(nnue_pytorch_directory)
        self._training_dataset = os.path.abspath(training_dataset)
        self._validation_dataset = os.path.abspath(validation_dataset)
        self._num_data_loader_threads = num_data_loader_threads
        self._num_pytorch_threads = num_pytorch_threads
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._random_fen_skipping = random_fen_skipping
        self._smart_fen_skipping = smart_fen_skipping
        self._wld_fen_skipping = wld_fen_skipping
        self._features = features
        self._lr = lr
        self._gamma = gamma
        self._network_save_period = network_save_period
        self._save_last_network = save_last_network
        self._seed = seed
        self._root_dir = os.path.abspath(root_dir)
        self._epoch_size = epoch_size
        self._validation_size = validation_size
        self._resume_from_model = resume_from_model
        self._resume_training = resume_training
        self._additional_args = additional_args

        self._current_step_in_epoch = None
        self._num_steps_in_epoch = None
        self._current_epoch = None
        self._current_loss = None
        self._momentary_iterations_per_second = None
        self._smooth_iterations_per_second = DecayingRunningAverage()
        self._has_finished = False
        self._has_started = False
        self._networks = []
        self._process = None
        self._running = False
        self._has_exited_unexpectedly = False
        self._error = None

        self._last_time = None
        self._last_step = None

    def _get_stringified_args(self):
        args = [
            self._training_dataset,
            self._validation_dataset,
            f'--num-workers={self._num_data_loader_threads}',
            f'--threads={self._num_pytorch_threads}',
            f'--max_epoch={self._num_epochs}',
            f'--batch-size={self._batch_size}',
            f'--random-fen-skipping={self._random_fen_skipping}',
            f'--gpus={self._gpu_id},',
            f'--features={self._features}',
            f'--lr={self._lr}',
            f'--gamma={self._gamma}',
            f'--network-save-period={self._network_save_period}',
            f'--save-last-network={self._save_last_network}',
            f'--seed={self._seed}',
            f'--epoch-size={self._epoch_size}',
            f'--validation-size={self._validation_size}',
            f'--default_root_dir={self._root_dir}',
        ]

        if self._smart_fen_skipping:
            args.append('--smart-fen-skipping')
        else:
            args.append('--no-smart-fen-skipping')

        if not self._wld_fen_skipping:
            args.append('--no-wld-fen-skipping')

        resumed = False
        if self._resume_training:
            ckpt_path = find_latest_checkpoint_in_run(self._root_dir)
            if ckpt_path:
                args.append(f'--resume_from_checkpoint={ckpt_path}')
                resumed = True

        if self._resume_from_model and not resumed:
            args.append(f'--resume-from-model={args._resume_from_model}')

        for arg in self._additional_args:
            args.append(arg)

        return args

    def run(self):
        self._running = True

        cmd = [sys.executable, 'train.py'] + self._get_stringified_args()
        LOGGER.info(f'Running training with command: {cmd}')
        self._process = subprocess.Popen(
            cmd,
            cwd=self._nnue_pytorch_directory,
            shell=False,
            bufsize=-1,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        terminate_process_on_exit(self._process)

        reader = io.TextIOWrapper(self._process.stdout)
        while self._process.poll() is None and self._running:
            if not self._running:
                break
            line = reader.readline().strip()
            if not self._has_finished:
                try:
                    matches = TrainingRun.ITERATION_PATTERN.search(line)
                    if matches:
                        self._current_epoch = int(matches.group(1))
                        self._current_step_in_epoch = int(matches.group(2))
                        self._num_steps_in_epoch = int(matches.group(3))

                        # There appears to be a pytorch lightning bug where it displays
                        # negative speed when running from checkpoint. So we work around this
                        # by computing our own speed.
                        curr_step = self._current_epoch * self._num_steps_in_epoch + self._current_step_in_epoch
                        if curr_step == self._last_step:
                            continue

                        curr_time = time.perf_counter_ns()
                        if self._last_time is None:
                            self._last_time = curr_time
                            self._last_step = curr_step
                            continue

                        #self._momentary_iterations_per_second = float(matches.group(4))
                        self._momentary_iterations_per_second = (curr_step-self._last_step)/((curr_time-self._last_time)/1e9)
                        self._smooth_iterations_per_second.update(self._momentary_iterations_per_second)
                        self._last_time = curr_time
                        self._last_step = curr_step

                        self._current_loss = float(matches.group(5))
                        self._has_started = True
                        if self._current_epoch == self._num_epochs - 1 and self._current_step_in_epoch >= self._num_steps_in_epoch:
                            self._has_finished = True
                        if self._current_step_in_epoch % 100 == 0:
                            LOGGER.info(line)
                except:
                    pass
                if 'CUDA_ERROR_OUT_OF_MEMORY' in line or 'CUDA out of memory' in line:
                    self._process.terminate()
                    self._error = 'Cuda out of memory error.'
                    break

        if self._has_finished:
            self._running = False

        self._has_finished = True
        if self._running:
            LOGGER.warning(f'Training run {self._run_id} exited unexpectedly.')
            if self._error:
                LOGGER.error(f'Error: {self._error}')
            self._has_exited_unexpectedly = True
        else:
            LOGGER.info(f'Training run {self._run_id} finished.')

        self._running = False

    def stop(self):
        self._running = False
        self.join()
        self._process.terminate()
        self._process.wait()

    @property
    def gpu_id(self):
        return self._gpu_id

    @property
    def run_id(self):
        return self._run_id

    @property
    def current_step_in_epoch(self):
        return self._current_step_in_epoch

    @property
    def current_epoch(self):
        return self._current_epoch

    @property
    def num_steps_in_epoch(self):
        return self._num_steps_in_epoch

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def current_loss(self):
        return self._current_loss

    @property
    def momentary_iterations_per_second(self):
        return self._momentary_iterations_per_second

    @property
    def smooth_iterations_per_second(self):
        return self._smooth_iterations_per_second.value

    @property
    def has_finished(self):
        return self._has_finished

    @property
    def has_started(self):
        return self._has_started

    @property
    def networks(self):
        return self._networks

    @property
    def has_exited_unexpectedly(self):
        return self._has_exited_unexpectedly

    @property
    def error(self):
        return self._error

def requests_get_content(url, *args, **kwargs):
    try:
        result = requests.get(url, *args, **kwargs)
        result.raise_for_status()
        return result.content
    except Exception as e:
        raise Exception(f'GET request to {url} failed')

def get_zipfile_members_strip_common_prefix(zipfile):
    parts = []
    for name in zipfile.namelist():
        if not name.endswith('/'):
            parts.append(name.split('/')[:-1])
    offset = len('/'.join(os.path.commonprefix(parts)) + '/')
    for zipinfo in zipfile.infolist():
        name = zipinfo.filename
        if len(name) > offset:
            zipinfo.filename = name[offset:]
            yield zipinfo

def git_download_branch_or_commit(directory, repo, branch_or_commit):
    url = f'http://github.com/{repo}/zipball/{branch_or_commit}'
    zipped_content = requests_get_content(url, timeout=60.0)
    zipped_input = zipfile.ZipFile(io.BytesIO(zipped_content), mode='r')
    zipped_input.extractall(directory, get_zipfile_members_strip_common_prefix(zipped_input))

def make_ordo_executable_path(directory):
    path = os.path.join(directory, 'ordo')
    if sys.platform == "win32":
        path += '.exe'
    return path

def is_ordo_setup(directory):
    try:
        ordo_path = make_ordo_executable_path(directory)
        with subprocess.Popen([ordo_path, '--help'], stdout=subprocess.DEVNULL) as process:
            if process.wait(timeout=10.0):
                return False
            return True
    except:
        return False

def setup_ordo(directory):
    if is_ordo_setup(directory):
        LOGGER.info(f'Ordo already setup in {directory}')
        return

    LOGGER.info(f'Setting up ordo in {directory}.')
    git_download_branch_or_commit(directory, *ORDO_GIT)
    if sys.platform == "win32":
        # need to append -DMINGW
        # ugly hack for a dumb makefile
        with open(os.path.join(directory, 'Makefile'), 'r') as makefile:
            lines = makefile.readlines()
            for i, line in enumerate(lines):
                if line.startswith('CFLAGS'):
                    lines.insert(i+1, 'CFLAGS += -DMINGW')
                    break

        with open(os.path.join(directory, 'Makefile'), 'w') as makefile:
            makefile.write('\n'.join(lines))

    with subprocess.Popen(['make'], cwd=directory) as process:
        if process.wait():
            raise Exception('Ordo compilation failed.')

    if not is_ordo_setup(directory):
        raise Exception('Ordo does not work.')

def make_c_ches_cli_executable_path(directory):
    path = os.path.join(directory, 'c-chess-cli')
    if sys.platform == "win32":
        path += '.exe'
    return path

def is_c_chess_cli_setup(directory):
    try:
        path = make_c_ches_cli_executable_path(directory)
        with subprocess.Popen([path, '-version'], stdout=subprocess.DEVNULL) as process:
            if process.wait(timeout=10.0):
                return False
            return True
    except:
        return False

def setup_c_chess_cli(directory):
    if is_c_chess_cli_setup(directory):
        LOGGER.info(f'c-chess-cli already setup in {directory}')
        return

    LOGGER.info(f'Setting up c-chess-cli in {directory}.')
    git_download_branch_or_commit(directory, *C_CHESS_CLI_GIT)
    with subprocess.Popen([sys.executable, 'make.py'], cwd=directory) as process:
        if process.wait():
            raise Exception('c-chess-cli compilation failed.')

    if not is_c_chess_cli_setup(directory):
        raise Exception('c-chess-cli does not work')

def make_stockfish_executable_path(directory):
    path = os.path.join(directory, 'src/stockfish')
    if sys.platform == "win32":
        path += '.exe'
    return path

def is_stockfish_setup(directory):
    try:
        path = make_stockfish_executable_path(directory)
        with subprocess.Popen([path, 'compiler'], stdout=subprocess.DEVNULL) as process:
            if process.wait(timeout=10.0):
                return False
            return True
    except:
        return False

def setup_stockfish(directory, repo, branch_or_commit, arch, threads=1):
    if is_stockfish_setup(directory):
        LOGGER.info(f'Stockfish already setup in {directory}.')
        return

    LOGGER.info(f'Setting up stockfish in {directory}.')
    git_download_branch_or_commit(directory, repo, branch_or_commit)

    srcdir = os.path.join(directory, 'src')
    env = os.environ.copy()
    if sys.platform == 'win32':
        env['MSYSTEM'] = 'MINGW64'

    with subprocess.Popen(
            ['make', 'build', f'ARCH={arch}', f'-j{threads}'],
            cwd=srcdir,
            env=env
        ) as process:
        if process.wait():
            raise Exception(f'stockfish {repo}/{branch_or_commit} compilation failed')

    if not is_stockfish_setup(directory):
        raise Exception(f'stockfish {repo}/{branch_or_commit} does not work')

def is_nnue_pytorch_setup(directory):
    try:
        with subprocess.Popen([sys.executable, 'nnue_dataset.py'], cwd=directory) as process:
            if process.wait(timeout=30.0):
                return False
            return True
    except:
        return False

def setup_nnue_pytorch(directory, repo, branch_or_commit):
    if is_nnue_pytorch_setup(directory):
        LOGGER.info(f'nnue-pytorch already setup in {directory}')
        return

    LOGGER.info(f'Setting up nnue-pytorch in {directory}')
    git_download_branch_or_commit(directory, repo, branch_or_commit)

    command = []
    if sys.platform == "linux":
        command += ['sh']
    command += [os.path.join(directory, 'compile_data_loader.bat')]
    with subprocess.Popen(command, cwd=directory) as process:
        if process.wait():
            raise Exception(f'nnue-pytorch {repo}/{branch_or_commit} data loader compilation failed')

    if not is_nnue_pytorch_setup(directory):
        raise Exception(f'Incorrect nnue-pytorch setup.')

class OrdoEntry:
    # nets are named experiment_path/run_{}/nn-epoch{}.nnue
    NET_PATTERN = re.compile(r'.*?run_(\d+).*?nn-epoch(\d+)\.nnue')
    def __init__(self, line=None, network_path=None, elo=None, elo_error=None, run_id=None, epoch=None):
        if line:
            fields = line.split()
            self._network_path = fields[1]
            self._elo = float(fields[3])
            self._elo_error = float(fields[4])
            net_parts = OrdoEntry.NET_PATTERN.search(self._network_path)
            self._run_id = int(net_parts[1])
            self._epoch = int(net_parts[2])
        else:
            self._network_path = network_path
            self._elo = elo
            self._elo_error = elo_error
            self._run_id = run_id
            self._epoch = epoch

    @property
    def network_path(self):
        return self._network_path

    @property
    def run_id(self):
        return self._run_id

    @property
    def epoch(self):
        return self._epoch

    @property
    def elo(self):
        return self._elo

    @property
    def elo_error(self):
        return self._elo_error

class CChessCliRunningTestEntry:
    LINE_PATTERN = re.compile(r'Score.*?run_(\d+).*?nn-epoch(\d+)\.nnue:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*')
    def __init__(self, line=None):
        fields = CChessCliRunningTestEntry.LINE_PATTERN.search(line)
        self._run_id = int(fields[1])
        self._epoch = int(fields[2])
        self._losses = int(fields[3]) # from base perspective so reversed
        self._draws = int(fields[4])
        self._wins = int(fields[5])

    @property
    def run_id(self):
        return self._run_id

    @property
    def epoch(self):
        return self._epoch

    @property
    def wins(self):
        return self._wins

    @property
    def draws(self):
        return self._draws

    @property
    def losses(self):
        return self._losses

    @property
    def total_games(self):
        return self._wins + self._draws + self._losses

    @property
    def performance(self):
        return (self._wins + self._draws * 0.5) / self.total_games

    def _elo(self, x):
        epsilon = 1e-3
        x = max(x, epsilon)
        x = min(x, 1 - epsilon)
        return -400 * math.log10(1 / x - 1)

    @property
    def elo(self):
        return self._elo(self.performance)

    @property
    def elo_error_95(self):
        return 400 / math.sqrt(self.total_games)


class NetworkTesting(Thread):
    def __init__(
        self,
        nnue_pytorch_directory,
        root_dir,
        num_parallel_games=4,
        explore_factor=1.5,
        book_file_path='',
        time_per_move=None,
        time_increment_per_move=None,
        nodes_per_move=1000,
        hash=8,
        games_per_round=200,
        ordo_exe=None,
        c_chess_cli_exe=None,
        stockfish_base_exe=None,
        stockfish_test_exe=None,
        features=None,
        active=True,
        additional_args=[]
    ):
        super(NetworkTesting, self).__init__()

        self._nnue_pytorch_directory = os.path.abspath(nnue_pytorch_directory)
        self._root_dir = os.path.abspath(root_dir)
        self._num_parallel_games = num_parallel_games
        self._explore_factor = explore_factor
        self._book_file_path = os.path.abspath(book_file_path)
        self._time_per_move = time_per_move
        self._time_increment_per_move = time_increment_per_move
        self._nodes_per_move = nodes_per_move
        self._hash = hash
        self._games_per_round = games_per_round
        self._ordo_exe = os.path.abspath(ordo_exe)
        self._c_chess_cli_exe = os.path.abspath(c_chess_cli_exe)
        self._stockfish_base_exe = os.path.abspath(stockfish_base_exe)
        self._stockfish_test_exe = os.path.abspath(stockfish_test_exe)
        self._features = features
        self._active = active
        self._additional_args = additional_args

        self._results = []
        self._running = False
        self._process = None
        self._current_test = None
        self._current_convert = None
        self._has_exited_unexpectedly = False

    def _get_stringified_args(self):
        args = [
            self._root_dir,
            f'--concurrency={self._num_parallel_games}',
            f'--explore_factor={self._explore_factor}',
            f'--ordo_exe={self._ordo_exe}',
            f'--c_chess_exe={self._c_chess_cli_exe}',
            f'--stockfish_base={self._stockfish_base_exe}',
            f'--stockfish_test={self._stockfish_test_exe}',
            f'--book_file_name={self._book_file_path}',
            f'--hash={self._hash}',
            f'--games_per_round={self._games_per_round}',
            f'--features={self._features}',
        ]

        if self._time_per_move:
            args.append(f'--time_per_move={self._time_per_move}')

        if self._time_increment_per_move:
            args.append(f'--time_increment_per_move={self._time_increment_per_move}')

        if self._nodes_per_move:
            args.append(f'--nodes_per_move={self._nodes_per_move}')

        for arg in self._additional_args:
            args.append(arg)

        return args

    def get_status_string(self):
        global RESOURCE_MONITOR
        cpu_usage = RESOURCE_MONITOR.resources.cpu_usage
        if not self._active:
            return 'Network testing inactive.'
        elif self._has_exited_unexpectedly:
            return 'Network testing has exited unexpectedly.'
        elif self._current_convert is not None:
            lines = [
                f'Converting network...',
                f'Run  : {self._current_convert[0]}',
                f'Epoch: {self._current_convert[1]}'
            ]
            return '\n'.join(lines)
        elif self._current_test is not None:
            perf_pct = int(round(self._current_test.performance * 100))
            lines = [
                f'CPU load: {cpu_usage * 100:0.1f}%',
                f'Testing run {self._current_test.run_id} epoch {self._current_test.epoch}',
                f'+{self._current_test.wins}={self._current_test.draws}-{self._current_test.losses} [{perf_pct:0.1f}%] ({self._current_test.total_games}/{self._games_per_round})',
                f'{self._current_test.elo:0.1f}±{self._current_test.elo_error_95:0.1f} Elo'
            ]
            return '\n'.join(lines)
        else:
            return 'Waiting for work...'

    def run(self):
        if not self._active:
            self._running = False
            return

        self._running = True

        cmd = [sys.executable, 'run_games.py'] + self._get_stringified_args()
        LOGGER.info(f'Running network testing with command: {cmd}')
        self._process = subprocess.Popen(
            cmd,
            cwd=self._nnue_pytorch_directory,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        terminate_process_on_exit(self._process)

        try:
            reader = io.TextIOWrapper(self._process.stdout)
            while self._process.poll() is None and self._running:
                if not self._running:
                    break
                line = reader.readline().strip()
                if line.startswith('Finished running ordo.'):
                    self._update_results_from_ordo_file(self._get_ordo_file_path())
                elif line.startswith('Score of'):
                    try:
                        self._current_test = CChessCliRunningTestEntry(line=line)
                        if self._current_test.total_games % 100 == 0:
                            LOGGER.info(self.get_status_string())
                        self._current_convert = None
                    except:
                        self._current_test = None
                elif line.startswith('Converting'):
                    fields = OrdoEntry.NET_PATTERN.search(line)
                    try:
                        self._current_convert = (fields[1], fields[2])
                        self._current_test = None
                        LOGGER.info(self.get_status_string())
                    except:
                        self._current_convert = None
                elif line.startswith('Error running match!'):
                    LOGGER.error('Error running matches. Exiting.')
                    break
                else:
                    self._current_test = None
        except:
            self._process.terminate()
            self._process.wait()

        if self._running:
            LOGGER.warning('Network testing exited unexpectedly.')
            self._has_exited_unexpectedly = True
        else:
            LOGGER.info('Network testing finished.')

        self._running = False

    def stop(self):
        self._running = False
        if self._process is not None:
            self._process.terminate()
            self._process.wait()

    def _get_ordo_file_path(self):
        return os.path.join(self._root_dir, 'ordo.out')

    def _update_results_from_ordo_file(self, ordo_file_path):
        new_results = []

        try:
            with open(ordo_file_path, 'r') as ordo_file:
                lines = ordo_file.readlines()
                LOGGER.info(lines[:5])
                for line in lines:
                    try:
                        entry = OrdoEntry(line=line)
                        new_results.append(entry)
                    except:
                        pass
            self._results = new_results
        except:
            pass

    def get_ordered_results(self):
        return list(sorted(self._results, key=lambda x: -x.elo))

    @property
    def has_exited_unexpectedly(self):
        return self._has_exited_unexpectedly

    @property
    def is_active(self):
        return self._active

def duration_string_from_seconds(seconds):
    second = int(seconds) % 60
    minute = int(seconds) // 60 % 60
    hour = int(seconds) // 3600
    return f'{hour}:{minute:02}:{second:02}'

def duration_string_from_seconds_compact(seconds):
    second = int(seconds) % 60
    minute = int(seconds) // 60 % 60
    hour = int(seconds) // 3600
    if hour > 0:
        return f'~{hour}h'
    elif minute > 0:
        return f'~{minute}m'
    else:
        return f'~{second}s'

class TrainerRunsWidget(Widget):
    def __init__(self, runs, name=None):
        super(TrainerRunsWidget, self).__init__(name)

        self._runs = list(sorted(runs, key=lambda x: (x.gpu_id, x.run_id)))

        self._selected_index = 0

    def add_run(self, run):
        self._runs.append(run)
        self._runs = list(sorted(runs, key=lambda x: (x.gpu_id, x.run_id)))

    def required_height(self, offset, w):
        return -135792468

    def reset(self):
        pass

    def _clear_area(self):
        colour, attr, background = self._frame.palette['field']

        height = self._h
        width = self._w - self._offset

        for i in range(height):
            self._frame.canvas.print_at(
                ' ' * width,
                self._x + self._offset,
                self._y + i,
                colour, attr, background
            )

    def _get_gpu_usage(self, gpu_ids):
        global RESOURCE_MONITOR
        gpus = RESOURCE_MONITOR.resources.gpus
        by_gpu_id = dict()
        for gpu_id in gpu_ids:
            if gpu_id in gpus:
                gpu = gpus[gpu_id]
                by_gpu_id[gpu_id] = {
                    'compute_pct' : int(gpu.load * 100),
                    'memory_mb' : int(gpu.memoryUsed),
                    'max_memory_mb' : int(gpu.memoryTotal)
                }
        return by_gpu_id

    def _get_unique_gpu_ids(self):
        ids = set()
        for run in self._runs:
            ids.add(run.gpu_id)
        return list(ids)

    def _make_run_text(self, run):
        if run.has_finished:
            if run.has_exited_unexpectedly:
                lines = [f'  Run {run.run_id} - Exited unexpectedly.']
                if run.error:
                    lines += [f'    Error: {run.error}']
                return lines
            else:
                loss = run.current_loss
                return [f'  Run {run.run_id} - Completed; Loss: {loss}']
        elif not run.has_started:
            return f'  Run {run.run_id} - Starting...',
        else:
            try:
                width = self._w - self._offset
                loss = run.current_loss
                epoch = run.current_epoch
                max_epoch = run.num_epochs - 1
                step_in_epoch = run.current_step_in_epoch
                max_step = run.num_steps_in_epoch - 1
                speed = run.smooth_iterations_per_second

                total_steps = run.num_epochs * run.num_steps_in_epoch
                step = epoch * run.num_steps_in_epoch + step_in_epoch
                complete_pct = step / total_steps * 100
                eta_seconds = (total_steps - step) / speed
                eta_str = duration_string_from_seconds_compact(eta_seconds)

                return [
                    f'  Run {run.run_id} - {complete_pct:0.2f}% ({speed:0.1f}it/s) [ETA {eta_str}]',
                    f'    Epoch: {epoch}/{max_epoch}; Step: {step_in_epoch}/{max_step}',
                    f'    Loss: {loss}',
                ]
            except:
                return [
                    f'  Run {run.run_id} - Waiting for enough data to display...'
                ]

    def _make_gpu_text(self, gpu_id, gpu_usage):
        if gpu_id in gpu_usage:
            gpu_compute_pct = gpu_usage[gpu_id]['compute_pct']
            gpu_memory_mb = gpu_usage[gpu_id]['memory_mb']
            gpu_max_memory_mb = gpu_usage[gpu_id]['max_memory_mb']
            return f'GPU {gpu_id} - Usage: {gpu_compute_pct}% {gpu_memory_mb}MB/{gpu_max_memory_mb}MB '
        else:
            return f'GPU {gpu_id}'

    def update(self, frame_no):
        self._clear_area()

        if self._has_focus:
            if self._selected_index is None:
                self._selected_index = 0
        else:
            self._selected_index = None

        if len(self._runs) <= 0:
            return

        height = self._h
        width = self._w - self._offset
        curr_line = 0
        prev_gpu_id = None

        gpu_usage = self._get_gpu_usage(self._get_unique_gpu_ids())
        for i, run in enumerate(self._runs):
            if curr_line >= height:
                break

            curr_gpu_id = run.gpu_id
            if prev_gpu_id != curr_gpu_id:
                if curr_line >= height:
                    break

                colour, attr, background = self._frame.palette['label']
                text = self._make_gpu_text(curr_gpu_id, gpu_usage)
                if len(text) < width:
                    text += '-' * (len(text) - width)
                self._frame.canvas.paint(
                    text,
                    self._x + self._offset,
                    self._y + curr_line,
                    colour, attr, background
                )
                curr_line += 1

                prev_gpu_id = curr_gpu_id

            colour, attr, background = self._pick_colours('field', i == self._selected_index)
            for line in self._make_run_text(run):
                if curr_line >= height:
                    break

                self._frame.canvas.paint(
                    line[:width-1],
                    self._x + self._offset,
                    self._y + curr_line,
                    colour, attr, background
                )
                curr_line += 1

    def value(self):
        if self._selected_index:
            return self._runs[self._selected_index]
        else:
            return None

    def process_event(self, event):
        if isinstance(event, KeyboardEvent):
            if len(self._runs) > 0 and event.key_code == Screen.KEY_UP:
                # Move up one line in text - use value to trigger on_select.
                self._selected_index = max(0, self._selected_index - 1)
            elif len(self._runs) > 0 and event.key_code == Screen.KEY_DOWN:
                # Move down one line in text - use value to trigger on_select.
                self._selected_index = min(len(self._runs) - 1, self._selected_index + 1)
            elif len(self._runs) > 0 and event.key_code == Screen.KEY_PAGE_UP:
                # Move up one page.
                self._selected_index = max(0, self._selected_index - self._h)
            elif len(self._runs) > 0 and event.key_code == Screen.KEY_PAGE_DOWN:
                # Move down one page.
                self._selected_index = min(len(self._runs) - 1, self._selected_index + self._h)
            else:
                return event
        else:
            # Ignore other events
            return event

        # If we got here, we processed the event - swallow it.
        return None

class MainView(Frame):
    def __init__(self, screen, training_runs, network_testing):
        super(MainView, self).__init__(
            screen,
            screen.height,
            screen.width,
            hover_focus=False,
            can_scroll=False,
            title="Dashboard",
            reduce_cpu=True,
        )

        self._training_runs = training_runs
        self._network_testing = network_testing

        layout = Layout([300, 10, 200], fill_frame=True)
        self.add_layout(layout)

        layout.add_widget(TrainerRunsWidget(self._training_runs, 'TrainerRuns'), 0)
        layout.add_widget(VerticalDivider(), 1)
        layout.add_widget(Label("Testing status:", 1), 2)
        self._network_testing_status = layout.add_widget(TextBox(4, line_wrap=True, readonly=True, as_string=True), 2)
        self._network_testing_status.disabled = True
        layout.add_widget(Divider(), 2)
        self._networks_view = layout.add_widget(
            MultiColumnListBox(
                Widget.FILL_FRAME,
                ['<4', '>4', '<6', '0', '>7', '<6'],
                [],
                add_scroll_bar=True,
                titles=['#', 'Run', 'Epoch', '', 'Elo', 'Err']
            ),
        2)

        layouta = Layout([1])
        self.add_layout(layouta)
        layouta.add_widget(Divider())

        layout2 = Layout([1, 1, 1, 1])
        self.add_layout(layout2)

        # TODO: Revise controls
        layout2.add_widget(Button("Quit", self._quit), 3)

        self.fix()

    def reset(self):
        # Do standard reset to clear out form, then populate with new data.
        super(MainView, self).reset()

    def _update_network_list(self):
        self._networks_view.options.clear()
        for i, entry in enumerate(self._network_testing.get_ordered_results()):
            self._networks_view.options.append(([
                str(i+1),
                str(entry.run_id),
                str(entry.epoch),
                '',
                f'{entry.elo:0.1f}',
                f'±{entry.elo_error:0.1f}'
            ], i))

    def _update_network_testing_status(self):
        self._network_testing_status.value = self._network_testing.get_status_string()

    def update(self, frame_no):
        super(MainView, self).update(frame_no)

        self._update_network_list()
        self._update_network_testing_status()

    def _quit(self):
        self._scene.add_effect(
            PopUpDialog(
                self._screen,
                "Are you sure?",
                ["Yes", "No"],
                has_shadow=True,
                on_close=self._quit_on_yes
            )
        )

    @staticmethod
    def _quit_on_yes(selected):
        # Yes is the first button
        if selected == 0:
            raise StopApplication("User requested exit.")

    @property
    def frame_update_count(self):
        return 1

def app(screen, scene, training_runs, network_testing):
    scenes = [
        Scene([MainView(screen, training_runs, network_testing)], -1, name="Main")
    ]

    screen.play(scenes, stop_on_resize=True, start_scene=scene, allow_int=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Trains the network.")
    parser.add_argument("--workspace-path", type=str, dest='workspace_path')
    parser.add_argument("--experiment-name", type=str, dest='experiment_name')
    parser.add_argument("--training-dataset", type=str, dest='training_dataset', help="Training data (.bin or .binpack)")
    parser.add_argument("--validation-dataset", type=str, dest='validation_dataset', default=None, help="Validation data (.bin or .binpack)")
    parser.add_argument("--lambda", default=1.0, type=float, dest='lambda_', help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).")
    parser.add_argument("--gamma", default=0.992, type=float, dest='gamma', help="Multiplicative factor applied to the learning rate after every epoch.")
    parser.add_argument("--lr", default=8.75e-4, type=float, dest='lr', help="Initial learning rate.")
    parser.add_argument("--num-workers", default=1, type=int, dest='num_workers', help="Number of worker threads to use for data loading. Currently only works well for binpack.")
    parser.add_argument("--batch-size", default=-1, type=int, dest='batch_size', help="Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128.")
    parser.add_argument("--threads", default=-1, type=int, dest='threads', help="Number of torch threads to use. Default automatic (cores) .")
    parser.add_argument("--seed", default=42, type=int, dest='seed', help="torch seed to use.")
    parser.add_argument("--smart-fen-skipping", default=True, type=str2bool, dest='smart_fen_skipping', help="If used then no smart fen skipping will be done. By default smart fen skipping is done.")
    parser.add_argument("--wld-fen-skipping", default=True, type=str2bool, dest='wld_fen_skipping', help="If used then no wld fen skipping will be done. By default wld fen skipping is done.")
    parser.add_argument("--random-fen-skipping", default=3, type=int, dest='random_fen_skipping', help="skip fens randomly on average random_fen_skipping before using one.")
    parser.add_argument("--resume-from-model", default=None, type=str, dest='resume_from_model', help="Initializes training using the weights from the given .pt model")
    parser.add_argument("--gpus", type=str, dest='gpus', default='0')
    parser.add_argument("--runs-per-gpu", type=int, dest='runs_per_gpu', default=1)
    parser.add_argument("--features", type=str, help="The feature set to use")
    parser.add_argument("--max_epoch", type=int, default=400, help="Number of epochs to train for.")
    parser.add_argument("--network-save-period", default=20, dest='network_save_period', help="Number of epochs between network snapshots. None to disable.")
    parser.add_argument("--save-last-network", default=True, dest='save_last_network', help="Whether to always save the last produced network.")
    parser.add_argument("--additional-training-arg", type=str, nargs='*', dest='additional_training_args', help="Additional training args passed verbatim.")
    parser.add_argument("--additional-testing-arg", type=str, nargs='*', dest='additional_testing_args', help="Additional network testing args passed verbatim.")
    parser.add_argument("--engine-base", type=str, dest='engine_base', help="Path to the commit/branch to use for the engine baseline. For example 'official-stockfish/Stockfish/master'")
    parser.add_argument("--engine-test", type=str, dest='engine_test', help="Path to the commit/branch to use for the engine being tested. For example 'official-stockfish/Stockfish/master'")
    parser.add_argument("--nnue-pytorch", type=str, dest='nnue_pytorch', help="Path to the commit/branch to use for the trainer being tested. For example 'glinscott/nnue-pytorch/master'")
    parser.add_argument("--build-engine-arch", type=str, default='x86-64-modern', dest='build_engine_arch', help="ARCH to use for engine compilation")
    parser.add_argument("--build-threads", type=int, default=1, dest='build_threads', help="Number of threads to use for compilation")
    parser.add_argument("--fail-on-experiment-exists", type=str2bool, default=True, dest='fail_on_experiment_exists', help="By default an experiment must be created in an empty directory. Should only be used for debugging.")
    parser.add_argument("--epoch-size", type=int, default=100000000, dest='epoch_size', help="Number of positions per epoch.")
    parser.add_argument("--validation-size", type=int, default=1000000, dest='validation_size', help="Number of positions per validation step.")
    parser.add_argument("--tui", type=str2bool, default=True, dest='tui', help="Whether to show a nice TUI.")
    parser.add_argument("--do-network-testing", type=str2bool, default=True, dest='do_network_testing', help="Whether to test networks as they are generated.")
    parser.add_argument("--do-network-training", type=str2bool, default=True, dest='do_network_training', help="Whether to train networks.")
    parser.add_argument("--network-testing-threads", type=int, default=1, dest='network_testing_threads', help="Number of threads to use for network testing.")
    parser.add_argument("--network-testing-explore-factor", type=float, default=1.5, dest='network_testing_explore_factor', help="Elo error estimates are multiplied by this amount to determine testing candidates.")
    parser.add_argument("--network-testing-book", type=str, default='./noob_3moves.epd', dest='network_testing_book', help="Path to a suitable book, see https://github.com/official-stockfish/books")
    parser.add_argument("--network-testing-time-per-move", type=float, default=None, dest='network_testing_time_per_move', help="Number of seconds per game")
    parser.add_argument("--network-testing-time-increment-per-move", type=float, default=None, dest='network_testing_time_increment_per_move', help="Number of seconds added to clock per move")
    parser.add_argument("--network-testing-nodes-per-move", type=int, default=None, dest='network_testing_nodes_per_move', help="Number of nodes per move to use for testing. Overrides time control. Should be used ove time control for better consistency.")
    parser.add_argument("--network-testing-hash-mb", type=int, default=8, dest='network_testing_hash_mb', help="Number of MB of memory to use for hash allocation for each engine being tested.")
    parser.add_argument("--network-testing-games-per-round", type=int, default=200, dest='network_testing_games_per_round', help="Number of games per round to use. Essentially a testing batch size.")
    parser.add_argument("--resume-training", type=str2bool, default=False, dest='resume_training', help="Attempts to resume each run from its latest checkpoint.")
    args = parser.parse_args()

    args.validation_dataset = args.validation_dataset or args.training_dataset

    # these are not required because testing is optional
    if args.engine_base and args.engine_base.count('/') != 2:
        raise Exception('Invalid base engine repo path')

    if args.engine_test and args.engine_test.count('/') != 2:
        raise Exception('Invalid test engine repo path')

    # this one is required because it has other important scripts
    if not args.nnue_pytorch or args.nnue_pytorch.count('/') != 2:
        raise Exception('Invalid test trainer repo path')

    if not args.network_testing_time_per_move and not args.network_testing_nodes_per_move:
        raise Exception('No time control specified.')

    return args

def do_bookkeeping(directory, args):
    os.makedirs(directory, exist_ok=True)

    args_dump_file_path = os.path.join(directory, 'args_dump.txt')
    with open(args_dump_file_path, 'w') as file:
        file.write(repr(args))

    logs_file_path = os.path.join(directory, 'easy_train.log')

    LOGGER.addHandler(logging.FileHandler(logs_file_path, encoding='utf-8'))

def is_url(path):
    return path.startswith('http://') or path.startswith('https://') or path.startswith('ftp://') or path.startswith('sftp://')

class TqdmDownloadProgressBar(tqdm):
    def update_to(self, blocks_transferred=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        return self.update(blocks_transferred * block_size - self.n)  # also sets self.n = b * bsize

def setup_book(directory, args):
    if not is_url(args.network_testing_book):
        return

    os.makedirs(directory, exist_ok=True)

    url = args.network_testing_book
    temp_filename = urllib.parse.unquote(url.split('/')[-1])
    if temp_filename.endswith('.zip'):
        filename = temp_filename[:-4]
    elif temp_filename.endswith('.epd'):
        filename = temp_filename

    if not filename.endswith('.epd'):
        LOGGER.error('Cannot handle the book. Currently only .epd books are supported. If compressed with .zip the name must be a.epd.zip. No other compression format is supported right now.')
        raise Exception('Cannot handle opening book')

    destination_temp_file_path = os.path.abspath(os.path.join(directory, temp_filename))
    destination_file_path = os.path.abspath(os.path.join(directory, filename))
    args.network_testing_book = destination_file_path

    if not os.path.exists(destination_file_path):
        if temp_filename != filename and not os.path.exists(destination_temp_file_path):
            with TqdmDownloadProgressBar(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=temp_filename) as progress_bar:
                urllib.request.urlretrieve(url, filename=destination_temp_file_path, reporthook=progress_bar.update_to, data=None)
                progress_bar.total = progress_bar.n
        if temp_filename.endswith('.zip'):
            zipped = zipfile.ZipFile(destination_temp_file_path, mode='r')
            names = zipped.namelist()
            if len(names) > 1 or names[0] != filename:
                LOGGER.error(f'Expected only a book with name {filename} in the archive but did not find it or found more')
                raise Exception('Unexpected opening book archive content.')
            LOGGER.info(f'Extracting {temp_filename} to {filename}')
            zipped.extract(filename, directory)

    LOGGER.info('Book setup completed.')

def main():
    LOGGER.info('Initializing...')

    args = parse_cli_args()

    if not args.tui:
        LOGGER.addHandler(logging.StreamHandler(stream=sys.stdout))

    absolute_workspace_path = os.path.abspath(args.workspace_path)
    experiment_directory = os.path.join(absolute_workspace_path, f'experiments/experiment_{args.experiment_name}')
    try:
        os.makedirs(experiment_directory, exist_ok=False)
    except FileExistsError as e:
        if args.fail_on_experiment_exists and os.listdir(experiment_directory):
            LOGGER.error(f'Directory {experiment_directory} already exists. An experiment must use a new directory.')
            return

    ordo_directory = os.path.join(absolute_workspace_path, 'ordo')
    c_chess_cli_directory = os.path.join(absolute_workspace_path, 'c-chess-cli')
    books_directory = os.path.join(absolute_workspace_path, 'books')

    stockfish_base_directory = os.path.join(experiment_directory, 'stockfish_base')
    stockfish_test_directory = os.path.join(experiment_directory, 'stockfish_test')
    nnue_pytorch_directory = os.path.join(experiment_directory, 'nnue-pytorch')
    bookkeeping_directory = os.path.join(experiment_directory, 'bookkeeping')

    do_bookkeeping(bookkeeping_directory, args)

    setup_ordo(ordo_directory)

    setup_c_chess_cli(c_chess_cli_directory)

    do_network_testing = args.engine_base and args.engine_test and args.do_network_testing
    do_network_training = args.do_network_training and args.training_dataset

    if do_network_testing:
        LOGGER.info('Engines provided. Enabling network testing.')

        setup_book(books_directory, args)

        stockfish_base_repo = '/'.join(args.engine_base.split('/')[:2])
        stockfish_test_repo = '/'.join(args.engine_test.split('/')[:2])
        stockfish_base_branch_or_commit = args.engine_base.split('/')[2]
        stockfish_test_branch_or_commit = args.engine_test.split('/')[2]
        setup_stockfish(stockfish_base_directory, stockfish_base_repo, stockfish_base_branch_or_commit, args.build_engine_arch, args.build_threads)
        setup_stockfish(stockfish_test_directory, stockfish_test_repo, stockfish_test_branch_or_commit, args.build_engine_arch, args.build_threads)
    else:
        LOGGER.info('Not doing network testing. Either engines no provided or explicitely disabled.')

    nnue_pytorch_repo = '/'.join(args.nnue_pytorch.split('/')[:2])
    nnue_pytorch_branch_or_commit = args.nnue_pytorch.split('/')[2]
    setup_nnue_pytorch(nnue_pytorch_directory, nnue_pytorch_repo, nnue_pytorch_branch_or_commit)

    LOGGER.info('Initialization completed.')

    # Directory layout:
    #     tmp/experiments/experiment_{name}/training/run_{i}
    #     tmp/experiments/experiment_{name}/stockfish_base
    #     tmp/experiments/experiment_{name}/stockfish_test
    #     tmp/experiments/experiment_{name}/nnue-pytorch
    #     tmp/experiments/experiment_{name}/bookkeeping
    #     tmp/c-chess-cli
    #     tmp/ordo
    training_runs = []
    if do_network_training:
        gpu_ids = [int(v) for v in args.gpus.split(',') if v]
        for gpu_id in gpu_ids:
            for j in range(args.runs_per_gpu):
                run_id = gpu_id*args.runs_per_gpu+j
                training_runs.append(TrainingRun(
                    gpu_id=gpu_id,
                    run_id=run_id,
                    nnue_pytorch_directory=nnue_pytorch_directory,
                    training_dataset=args.training_dataset,
                    validation_dataset=args.validation_dataset,
                    num_data_loader_threads=args.num_workers,
                    num_pytorch_threads=args.threads,
                    num_epochs=args.max_epoch,
                    batch_size=args.batch_size,
                    random_fen_skipping=args.random_fen_skipping,
                    smart_fen_skipping=args.smart_fen_skipping,
                    wld_fen_skipping=args.wld_fen_skipping,
                    features=args.features,
                    lr=args.lr,
                    gamma=args.gamma,
                    network_save_period=args.network_save_period,
                    save_last_network=args.save_last_network,
                    seed=args.seed,
                    resume_from_model=args.resume_from_model,
                    root_dir=os.path.join(experiment_directory, 'training', f'run_{run_id}'),
                    epoch_size=args.epoch_size,
                    validation_size=args.validation_size,
                    resume_training=args.resume_training,
                    additional_args=[arg for arg in args.additional_training_args or []]
                ))
        LOGGER.info(f'Doing network training on gpus {gpu_ids}. {len(training_runs)} runs in total.')
    else:
        LOGGER.info('Not training networks.')

    network_testing = NetworkTesting(
        nnue_pytorch_directory=nnue_pytorch_directory,
        root_dir=os.path.join(experiment_directory, 'training'),
        ordo_exe=make_ordo_executable_path(ordo_directory),
        c_chess_cli_exe=make_c_ches_cli_executable_path(c_chess_cli_directory),
        stockfish_base_exe=make_stockfish_executable_path(stockfish_base_directory),
        stockfish_test_exe=make_stockfish_executable_path(stockfish_test_directory),
        features=args.features,
        num_parallel_games=args.network_testing_threads,
        explore_factor=args.network_testing_explore_factor,
        book_file_path=args.network_testing_book,
        time_per_move=args.network_testing_time_per_move,
        time_increment_per_move=args.network_testing_time_increment_per_move,
        nodes_per_move=args.network_testing_nodes_per_move,
        hash=args.network_testing_hash_mb,
        games_per_round=args.network_testing_games_per_round,
        active=do_network_testing,
        additional_args=[arg for arg in args.additional_testing_args or []]
    )

    for tr in training_runs:
        tr.start()

    network_testing.start()

    if args.tui:
        last_scene = None
        while True:
            try:
                Screen.wrapper(app, catch_interrupt=True, arguments=[last_scene, training_runs, network_testing])
                break
            except ResizeScreenError as e:
                last_scene = e.scene
    else:
        while True:
            v = input()
            if v == 'quit':
                break
            else:
                print('Type `quit` to stop.')

    for tr in training_runs:
        tr.stop()
    network_testing.stop()

if __name__ == '__main__':
    try:
        main()
    finally:
        RESOURCE_MONITOR.stop()
