from multiprocessing import Lock

PENDING = 'Pending'
FINISHED = 'Finished'

__all__ = ['Future', 'FutureLambda', 'FutureFlowInput']


class ResultNotSetError(Exception):

    """Raised when future objects's result() method is invoked without
    setting result in the first place
    """


class Future(object):

    """The result of an execution of a step.
    The result on the future is obtained by calling the .result() method
    The result can also re-raise the error that was raised
    during the execution of the step
    """

    def __init__(self):
        self._result = None
        self._status = PENDING
        self._exception = None
        self._traceback = None
        self._lock = Lock()

    def __repr__(self):
        with self._lock:
            if self._status == FINISHED:
                if self._exception:
                    return '<Future at %s status=%s raised %s>' % (
                        hex(id(self)),
                        self._status,
                        self._exception)
                else:
                    return '<Future at %s status=%s returned %s>' % (
                        hex(id(self)),
                        self._status,
                        self._result.__class__.__name__)
            return '<Future at %s status=%s>' % (
                hex(id(self)),
                self._status)

    def _get_result(self):
        if self._exception:
            raise self._exception, None, self._traceback
        return self._result

    def result(self):
        """Invoked to get result of the future object
        If the future result not set but the set_exception method was invoked,
        this method call will re-raise the same exception
        Raises:
            ResultNotSetError : Raised when the method is invoked
                before invoking the set_result method.
        """
        with self._lock:
            if self._status == FINISHED:
                return self._get_result()
            else:
                raise ResultNotSetError("Result is not Set yet !")

    def set_result(self, result):
        """Used to set the result of the futures object
           sets the status to FINISHED
        """
        assert self._exception is None
        with self._lock:
            self._result = result
            self._status = FINISHED

    def set_exception(self, exception, traceback):
        """Sets the future result to Exception that was raised
           while execution of the step
           Args :
                exception : The exception that was raised
                traceback : the traceback for the exception
        """
        with self._lock:
            self._exception = exception
            self._traceback = traceback
            self._status = FINISHED

    def reset(self):
        """Resets the future object
           Sets result/exception to None
           and status to PENDING
        """
        with self._lock:
            self._result = None
            self._status = PENDING
            self._exception = None
            self._traceback = None

    def done(self):
        """Returns True if future is in FINISHED state"""
        return self._status == FINISHED

    def pending(self):
        """Returns True if future is in PENDING state"""
        return self._status == PENDING


class FutureLambda(object):

    """Used to apply a lambda on the result of a Future Object"""

    def __init__(self, future, function):
        assert callable(function)
        assert isinstance(future, (Future, FutureFlowInput))
        self.future = future
        self.function = function

    def result(self):
        return self.function(self.future.result())


class FutureFlowInput(object):

    """Used to inform a flow of future input"""

    def __init__(self, flow, key):
        from affine.detection.model.mlflow.flow import Flow
        assert isinstance(flow, Flow)
        assert isinstance(key, str)
        self.flow = flow
        self.key = key

    def result(self):
        return self.flow.data_store.get(self.key)
