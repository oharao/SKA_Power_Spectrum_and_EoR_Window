"""
Contains functions to initialise the associated logging parameters. This specifies the
logging format and removes "spam" entries diluting key information.
@author:
    Oscar Sage David O'Hara
@email:
    osdo2@cam.ac.uk
"""

import logging


def init_logger(filepath):
    """
    Once called this initialises logging config. By default, logging levels have been altered for
    matplotlib, requests & numexpr.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(filepath + '.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y/%m/%d %H:%M:%S')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    logging.getLogger('requests').setLevel(logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)
    return logger
