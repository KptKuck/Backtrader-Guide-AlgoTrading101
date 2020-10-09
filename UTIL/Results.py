import os
from datetime import datetime
import errno

unique_str = datetime.now().isoformat().replace(':', '_').replace('.', '_').replace('-', '_').replace('T', '__')
LOG_FILENAME = 'tmp' + os.sep + 'logs' + os.sep + 'masterPVTrainer_'+unique_str+'.log'

def create_dir_string(log):
    string_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    log.info("New output dir: %s", string_time)
    return string_time


def check_output_dir(dirname, log):
    dir_check = os.getcwd() + r"\\" + dirname
    if os.path.isdir(dir_check):
        return True
    else:
        log.warning("Result dir not exits .. create")
        return False


def create_newdir(newOutputDir, log):
    mydir = os.path.join(os.getcwd(), newOutputDir)

    try:
        os.makedirs(mydir)
    except OSError as e:
        log.error("OSError")
        log.error(e.strerror)
        log.error(e.filename)
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        return False
    return True


def check(dirname, log):
    path_now = os.getcwd()
    if check_output_dir(dirname, log):
        newOutputDir = create_dir_string(log)
        os.chdir(dirname)
        if create_newdir(newOutputDir, log):
            log.info("Output dir created: %s", newOutputDir)
            os.chdir(newOutputDir)
            log.info("New output dir created")
            os.chdir(path_now)
            result_dir = path_now + "\\" + dirname + "\\" + newOutputDir
            return True, newOutputDir, result_dir
        else:
            log.error("Create new output dir")
            os.chdir(path_now)
            return False

    else:
        if create_newdir(dirname, log):
            os.chdir(dirname)
            newOutputDir = create_dir_string(log)
            if create_newdir(newOutputDir, log):
                os.chdir(newOutputDir)
                log.info("New output dir created")
                os.chdir(path_now)
                result_dir = path_now + "\\" + dirname + "\\" + newOutputDir
                return True, newOutputDir, result_dir
            else:
                log.error("Create new output dir")
                os.chdir(path_now)
                return False
        else:
            return False
