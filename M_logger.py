import os
import logging


def init_logger(dunder_name, show_debug=False) -> object:
    logging.getLogger('tensorflow').disabled = True
    logger = logging.getLogger(dunder_name)
    fh = logging.FileHandler(r'log\\core.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(name)-15s: %(funcName)-30s:: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if show_debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Note: these file outputs are left in place as examples
    # Feel free to uncomment and use the outputs as you like

    # Output full log
    # fh = logging.FileHandler(os.path.join('data', log', 'trading.log')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(log_format)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    # # Output warning log
    # fh = logging.FileHandler(os.path.join('data', log', 'trading.warning.log')
    # fh.setLevel(logging.WARNING)
    # formatter = logging.Formatter(log_format)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    # # Output error log
    # fh = logging.FileHandler(os.path.join('data', log', 'trading.error.log')
    # fh.setLevel(logging.ERROR)
    # formatter = logging.Formatter(log_format)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    return logger
