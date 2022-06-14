import os, json, sys
import logging
import torch


def _load(string):
    '''
    Helper function to decode environment variables.
    '''
    if isinstance(string, str):
        try:
            return json.loads(string)
        except json.decoder.JSONDecodeError:
            return string
    else:
        return string


class TrainingCompilerConfig:
    '''
    Configures the SM Training Compiler
    '''
    SM_OUTPUT_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '')
    DEBUG_PATH = os.path.join(SM_OUTPUT_DIR, "compiler")
    PARENT_ENV_VARIABLE = 'SM_FRAMEWORK_PARAMS'
    SM_TRAINING_COMPILER_PREFIX = "sagemaker_training_compiler_"
    
    HP_ENABLE_COMPILER = "sagemaker_training_compiler_enabled"
    HP_ENABLE_DEBUG = "sagemaker_training_compiler_debug_mode"

    def __init__(self):
        '''
        Check if a configuration is present for SM Training Compiler
        '''
        self.logger = self._get_logger()
        if self.is_already_configured():
            self.logger.debug("SM Training Compiler has already been configured.")
        else:
            sm_framework_params_string = os.environ.get(self.PARENT_ENV_VARIABLE, '')
            if sm_framework_params_string:
                all_params = json.loads(sm_framework_params_string)
                self.params = { key: _load(all_params[key]) for key in all_params \
                                     if key.startswith(self.SM_TRAINING_COMPILER_PREFIX) }
                if self.params:
                    self.logger.debug(f"Received hyper-parameters {json.dumps(self.params)}")
                    self.map_hp_to_config()
                    self.configure_compiler()
                else:
                    self.logger.debug(f"No configuration found for SM Training Compiler")
            else:
                self.logger.debug("Sagemaker context unavailable. Cannot configure SM Training Compiler")


    def _get_logger(self, name=''):
        '''
        Returns a logger which respects the Log Level set by the CreateTrainingJob API.
        '''
        logger = logging.getLogger(__name__).getChild(name or self.__class__.__name__)
        logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', '20')))
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        formatter = logging.Formatter(
                            fmt="[%(asctime)s.%(msecs)03d %(name)s %(levelname)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S",
                            )
        handler.setFormatter(formatter)
        return logger


    def map_hp_to_config(self):
        '''
        Convert hyper-parameters from SM Training ToolKit to 
        environment flags for the compiler.
        '''
        self.enabled = self.params.get(self.HP_ENABLE_COMPILER, False)
        if self.enabled:
            self.debug = self.params.get(self.HP_ENABLE_DEBUG, False)

        self.logger.debug(f"enabled={self.enabled}")
        self.logger.debug(f"debug={self.debug}")


    def _warnings_and_disclaimers(self):
        '''
        Displays warnings about Debug mode
        '''
        self.logger.warning(  "Training Compiler set to debug mode. "
                                "This may impact performance. "
                                f"Debug artifacts will be saved in {self.DEBUG_PATH}. "
                        )


    def _welcome_banner(self):
        '''
        Displays loglines to indicate the compiler has been configured.
        '''
        self.logger.info(  "Configuring SM Training Compiler... "
                        )


    def _set_xla_flags(self, key, value=None):
        '''
        Helper function to set XLA flags
        '''
        parent='XLA_FLAGS'
        pre_existing_value = os.environ.get(parent, '')
        if pre_existing_value:
            for _key in pre_existing_value.split(' '):
                if _key.startswith('--'+key):
                    self.logger.warning(f"Found pre-existing configuration {_key}.")
                    return
            if value:
                os.environ[parent]+=f" --{key}={value}"
                self.logger.debug(f"Set flag {key}={value}")
            else:
                os.environ[parent]+=f" --{key}"
                self.logger.debug(f"Set flag {key}")
        else:
            if value:
                os.environ[parent]=f"--{key}={value}"
                self.logger.debug(f"Set flag {key}={value}")
            else:
                os.environ[parent]=f"--{key}"
                self.logger.debug(f"Set flag {key}")


    def _set_flags(self, key, value):
        '''
        Helper function to set environment variables
        '''
        pre_existing_value = os.environ.get(key, False)
        if pre_existing_value:
            self.logger.warning(f"Found pre-existing configuration {key}={pre_existing_value}.")
        else:
            os.environ[key]=str(value)
            self.logger.debug(f"Set flag {key}={os.environ[key]}")


    def set_debug_flags(self):
        '''
        Configure the compiler for debugging.
        '''
        self._warnings_and_disclaimers()
        
        # Auto-Metrics Analysis
        self._set_flags('PT_XLA_DEBUG', 1)
        
        # Python stack trace for IR generation
        self._set_flags('XLA_IR_DEBUG', 1)
        # Propagate Python stack trace to TF-XLA HLO metadata for end-to-end debugging. Must be used with XLA_IR_DEBUG
        self._set_flags('XLA_HLO_DEBUG', 1)

        # Dump IR graphs to file. File is appended to if it already exists
        self._set_flags('XLA_SAVE_TENSORS_FILE', os.path.join(self.DEBUG_PATH, 'XLA_SAVE_TENSORS_FILE.hlo'))
        # Format of dumped IR graphs. Must be used with XLA_SAVE_TENSORS_FILE
        self._set_flags('XLA_SAVE_TENSORS_FMT', 'hlo')

        # Dump metrics log to file. File is appended to if it already exists
        self._set_flags('XLA_METRICS_FILE', os.path.join(self.DEBUG_PATH, 'XLA_METRICS_FILE.txt'))

        # Dump HLO graph to file if execution fails
        self._set_flags('XLA_SAVE_HLO_FILE', os.path.join(self.DEBUG_PATH, 'XLA_SAVE_HLO_FILE.hlo'))

        self._set_flags('XLA_DUMP_FATAL_STACK', 1)
        self._set_flags('XLA_DUMP_HLO_GRAPH', 1)


    def configure_compiler(self):
        '''
        Configure the compiler as requested by the hyper-parameters.
        Indicate successful configuration.
        '''
        if self.enabled:
            self._welcome_banner()
            # Set GPU_NUM_DEVICES=1 by default
            self._set_flags('GPU_NUM_DEVICES', 1)

            # Set flag to fix the div by fp64 bug for Tesla T4 GPU
            # We'd like to keep this bug for Tesla V100
            # https://github.com/pytorch/xla/pull/3229
            # TODO: Remove it once the performance issue on Tesla V100 is fixed
            if torch.cuda.get_device_name().startswith("Tesla T4"):
                self._set_flags('XLA_FIX_DIV_FP64', 1)

            if self.debug:
                os.makedirs(self.DEBUG_PATH, exist_ok=True)
                self.set_debug_flags()
            self._set_flags('SM_TRAINING_COMPILER_CONFIGURED', 1)


    def is_already_configured(self):
        '''
        Checks if SM Training Compiler has already been configured from a previous import.
        '''
        preconfigured = bool(int(os.environ.get('SM_TRAINING_COMPILER_CONFIGURED', '0')))
        return preconfigured


TrainingCompilerConfig()
