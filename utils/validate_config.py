from configobj import ConfigObj
from validate import Validator


def validate_config(config_file, config_spec):
    config_obj = ConfigObj(config_file, configspec=config_spec.split('\n'))
    validator = Validator()
    result = config_obj.validate(validator, copy=True, preserve_errors=True)
    if result is not True:
        msg = 'Config file validation failed: %s' % result
        raise Exception(msg)
    return config_obj
