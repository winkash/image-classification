import os
import cv2
from tempfile import mkstemp
from celery.utils.log import get_task_logger
from affine.model.classifier_models import ClassifierModel
from affine.detection.model_worker.celery_app import celery_app
from affine.detection.model_worker.config import DEFAULT_QUEUE
from affine import librato_tools

logger = get_task_logger(__name__)

# Timeout in seconds
TIMEOUT = 900 # 15 mins

NUM_ASYNC_CALLS = 10

CELERY_QUEUE_KWARG = 'celery_queue'


def convert_files_to_bin(file_names, resize=None):
    """
    This function converts a list of files to binary data and returns that as
    a list

    Args:
        file_names: list of file names

    Kwargs:
        resize: tuple of (w, h) of the new images required size

    Returns:
        List of items with each one corresponding to the image passed in.
    """
    bin_data = []
    for file_name in file_names:
        temp_img_file = file_name
        if resize is not None:
            assert len(resize) == 2, "Resize parameter should be of length 2"
            img = cv2.imread(file_name)
            h, temp_img_file = mkstemp()
            os.close(h)
            temp_img_file += '.{}'.format(file_name.split('.')[-1])
            img = cv2.resize(img, resize)
            cv2.imwrite(temp_img_file, img)
        with open(temp_img_file, 'rb') as handle:
            bin_data.append(handle.read())
    return bin_data


class Pool(object):
    """
    This class provides functionality similar to that provided my pool map
    in multiprocessing.
    """

    def __init__(self, dp_client, num_calls=NUM_ASYNC_CALLS):
        """
        Args:
            dp_client: The data processor client instance to use for making
            function calls.
            num_calls: The number of asynchronous calls to the server that
            can be made by the pool instance
        """
        self.dp_client = dp_client
        self.num_calls = num_calls

    def map_async(self, func_name, arg_list, timeout=TIMEOUT):
        """
        This function makes async calls to the celery broker (limited by
        num_calls)

        Args:
            func_name: String specifying the name of the function provided
                       by the dp_client
            arg_list: List of lists. Each list has a first element which is
                      a tuple of args and an optional second element which is a dict
                      of kwargs

                      Ex:
                          arg_list = [[([1,2], 3,), {'b':1}], [(2,),]]
                          This results in two function calls of the following
                          types,
                          f([1,2], 3, b=1) and f(2)

            timeout: optional timeout to wait for any call to complete.
        """
        out_list = []
        func = getattr(self.dp_client, func_name)
        for ind in range(0, len(arg_list), self.num_calls):
            in_list = []
            for i in range(ind, min(ind + self.num_calls, len(arg_list))):
                if len(arg_list[i]) == 1:
                    arg_list[i].append({})
                arg_list[i][-1].update(async=True)
                in_list.append(func(*arg_list[i][0], **arg_list[i][1]))
            for val in in_list:
                out_list.append(val.wait(timeout=timeout))
        return out_list


class DataProcessorClient(object):

    def __init__(self, clf_model_name):
        self.dp_task = DataProcessorTask()
        self.clf_model_name = clf_model_name
        clf_model = ClassifierModel.query.\
            filter_by(name=clf_model_name).\
            one()
        self.dp_task.queue = clf_model.celery_queue

    def __getattr__(self, name):
        return self.dp_task.get_func(self.clf_model_name, name)

    def set_eager(self, is_local=True):
        self.dp_task.app.conf.update(CELERY_ALWAYS_EAGER=is_local)


class DataProcessorTask(celery_app.Task):

    def __init__(self):
        self.clf_model_map = {}
        self._queue = DEFAULT_QUEUE

    @property
    def queue(self):
        return self._queue

    @queue.setter
    def queue(self, queue):
        self._queue = queue

    @librato_tools.timeit(metric_prefix_func=(lambda *args,**kwargs: "%s.%s" % (args[1], args[2])),
                          report_count=True)
    def run(self, clf_model_name, func_name, *f_args, **f_kwargs):
        f_kwargs.pop(CELERY_QUEUE_KWARG)
        if clf_model_name not in self.clf_model_map:
            clf_model = ClassifierModel.query.\
                filter_by(name=clf_model_name).\
                scalar()
            self.clf_model_map[clf_model_name] = clf_model.get_data_processor()
        func = getattr(self.clf_model_map[clf_model_name], func_name)
        return func(*f_args, **f_kwargs)

    def get_func(self, clf_model_name, name):
        def wrap(*args, **kwargs):
            if 'async' in kwargs:
                async = kwargs.pop('async')
            else:
                async = False
            timeout = kwargs.pop('timeout', TIMEOUT)
            assert CELERY_QUEUE_KWARG not in kwargs, \
                "Cannot use '{}' as a kwarg".format(CELERY_QUEUE_KWARG)
            kwargs[CELERY_QUEUE_KWARG] = self._queue
            if async:
                result = self.apply_async(args=(clf_model_name, name) + args,
                                          kwargs=kwargs, expires=timeout,
                                          queue=self._queue)
                return result
            else:
                return self.apply_async(args=(clf_model_name, name) + args,
                                        kwargs=kwargs, expires=timeout,
                                        queue=self._queue).\
                    wait(timeout=timeout)
        return wrap
