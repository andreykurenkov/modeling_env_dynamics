import os 
import random
import logging
import coloredlogs
from hydra.core.hydra_config import HydraConfig

def sample_from_dict(dict_with_weights):
    d = dict_with_weights
    return random.choices(list(d.keys()), weights=d.values(), k=1)[0]

def float_to_hex(number, base = 16):
    if number < 0:                          # Check if the number is negative to manage the sign
        sign = "-"                          # Set the negative sign, it will be used later to generate the first element of the result list
        number = -number                    # Change the number sign to positive
    else:
        sign = ""                           # Set the positive sign, it will be used later to generate the first element of the result list

    s = [sign + str(int(number)) + '.']     # Generate the list, the first element will be the integer part of the input number
    number -= int(number)                   # Remove the integer part from the number

    for i in range(base):                   # Iterate N time where N is the required base
        y = int(number * 16)                # Multiply the number by 16 and take the integer part
        s.append(hex(y)[2:])                # Append to the list the hex value of y, the result is in format 0x00 so we take the value from postion 2 to the end
        number = number * 16 - y            # Calculate the next number required for the conversion

    return ''.join(s).rstrip('0')

def char_code_at(testS):
    l = list(bytes(testS, 'utf-16'))[2:]
    for i, c in enumerate([(b<<8)|a for a,b in list(zip(l,l[1:]))[::2]]):
        return c

def string_to_color(text):
    hash = 0
    for x in text:
        hash = char_code_at(x) + ((hash << 5) - hash)

    colour = '#';
    for i in range(3-1):
        value = (hash >> (i * 8)) & 0xFF;
        colour += ('00' + float_to_hex(value,16)[-2])
    return colour

FORMAT_STR = '%(asctime)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def configure_logging(dir_path=None, format_strs=[None], name='log', log_suffix=''):
    if dir_path is None:
        dir_path = os.path.join(HydraConfig.get().runtime.output_dir)
    logger = logging.getLogger()  # root logger
    formatter = logging.Formatter(FORMAT_STR, DATE_FORMAT)
    file_to_delete = open("info.txt",'w')
    file_to_delete.close()
    file_path = "{0}/{1}.log".format(dir_path, name)
    #file_handler = logging.FileHandler(filename="{0}/{1}.log".format(dir_path, name), mode='w')
    #file_handler.setFormatter(formatter)
    #logger.addHandler(file_handler)
    if os.isatty(2):
        coloredlogs.install(fmt=FORMAT_STR, level='INFO')
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger
