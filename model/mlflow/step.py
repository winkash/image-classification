import sys

from logging import getLogger
from multiprocessing import Lock

# Step possible States
RUNNING = 'Running'
DONE = 'Done'
FAILED = 'Failed'
NOT_STARTED = 'Not Started'

logger = getLogger(__name__)


__all__ = ['Step']


class Step(object):

    """A step is the encapsulation of an actors specific activity on the actor"""

    def __init__(self, name, actor, activity=None):
        """
        Args:
            name : Name for the Step,
                   The name acts as a unique identifier for the step
            actor : The actor can be either any class instance or a method.
                    If it is an instance it needs to be also given an activity
                    that the actor will perform
            activity : The activity that the actor is supposed to perform
                       The step's execution is achieved by calling
                       actor.activity(*args, **kwargs)
        """
        from affine.detection.model.mlflow.future import Future
        self._name = name
        self.actor = actor
        if not callable(actor):
            assert activity is not None, \
                "activity cannot be None if actor is not callable"
            assert hasattr(actor, activity)
            assert callable(getattr(actor, activity))

        self.activity = activity
        self._done_callbacks = []
        # status cannot be chnaged by multiple threads at the same time
        self._status = NOT_STARTED
        self._status_lock = Lock()
        # Only one thread can exceute the Step at one time
        self._execute_lock = Lock()
        self._output = Future()

    def __repr__(self):
        return u'<Step name=%s state=%s>' % (self.name, self.status)

    @property
    def name(self):
        """ The name of the Step class is its hashable identity
            Cannot be modified once set in the constructor
        """
        return self._name

    def reset(self):
        assert self.status != RUNNING
        self._output.reset()
        self.status = NOT_STARTED

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, val):
        with self._status_lock:
            assert val in [DONE, FAILED, NOT_STARTED, RUNNING], val
            self._status = val

    @property
    def is_callable(self):
        return callable(self.actor)

    def add_done_callback(self, fn):
        """Add ability to call a function which accepts the step argument
           post execution of the Step Activity

           Args:
                fn : A callable which accepts only one argument
                     which is the Step itself
        """
        assert self.status == NOT_STARTED, \
            "Done callbacks can only be added if step has not started exectution"
        assert callable(fn), "Done callbacks should be functions"
        self._done_callbacks.append(fn)

    def _get_func_to_exec(self):
        if self.is_callable:
            func = self.actor
        else:
            func = getattr(self.actor, self.activity)
        return func

    def _run(self, *args, **kwargs):
        func = self._get_func_to_exec()
        return func(*args, **kwargs)

    def _execute_done_callbacks(self):
        for fn in self._done_callbacks:
            try:
                fn(self)
            except Exception:
                logger.exception("Failed to execute Done callback, %s", fn)
                raise

    def execute(self, *args, **kwargs):
        logger.debug("Executing Step : %s", self.name)
        with self._execute_lock:
            if self.status == NOT_STARTED:
                logger.debug("%s Got args : %s", self.name, args)
                logger.debug("%s Got kwargs : %s", self.name, kwargs)
                try:
                    self.status = RUNNING
                    result = self._run(*args, **kwargs)
                    self._execute_done_callbacks()
                except Exception:
                    logger.exception(
                        "Failed to execute run for %s" % self.name)
                    self.status = FAILED
                    self.output = sys.exc_info()[1:]
                else:
                    self.output = result
                    self.status = DONE
                    logger.debug("Finished Step : %s", self.name)
            else:
                # this can only hapend when 2 threads try
                # to start executing the thread at the same time
                # In that case, the second thread does not re-execute
                # and just bails out
                logger.debug(
                    "Will not run Step. Status is not %s", NOT_STARTED)

    @property
    def output(self):
        if self._output.done():
            return self._output.result()
        else:
            return self._output

    @output.setter
    def output(self, val):
        if isinstance(val, tuple) and isinstance(val[0], Exception):
            self._output.set_exception(*val)
        else:
            self._output.set_result(val)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return self.name != other.name
