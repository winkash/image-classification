import logging
from mock import Mock, create_autospec
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock
from logging import getLogger

from affine.detection.model.mlflow.future import Future, FutureFlowInput, FutureLambda
from affine.detection.model.mlflow.step import Step, FAILED


logger = getLogger(name=__name__)

__all__ = ['Flow', 'FlowTester']


class Flow(object):

    """A connection of various steps"""

    def __init__(self, name=''):
        self.name = name
        # other attrs
        self.steps = {}
        self.splits = defaultdict(set)
        self.merges = defaultdict(set)
        # map holding ip_args to be presented to each step at execution time
        self._ip_args = {}
        # the map holding future flow inputs
        self.data_store = {}
        self.step_run_result = {}
        self._output = Future()
        self.starting_step_name = None
        self.testing_enabled = False
        # lock to ensure only one thread
        # can make updates to pending_parents_count map
        self._child_lock = Lock()
        # map to record number of parents steps yet to finsh execution
        self._pending_parents_count = {}

    def enable_test_mode(self):
        self._reset()
        self.ip_args_for_step = {}
        self.testing_enabled = True
        self._ip_args_lock = Lock()

    def disable_test_mode(self):
        self.testing_enabled = False

    @staticmethod
    def enable_log():
        logger.setLevel(logging.DEBUG)

    @staticmethod
    def disable_log():
        logger.setLevel(logging.INFO)

    @property
    def output(self):
        if self._output.done():
            return self._output.result()
        else:
            self._output

    @output.setter
    def output(self, future):
        assert isinstance(future, Future)
        self._output = future

    def add_step(self, step):
        assert step.name not in self.steps, "Step %s is not added to the flow" % step.name
        self.steps[step.name] = step

    def _connect_steps(self, from_steps, to_step):
        assert to_step.name in self.steps
        if not isinstance(from_steps, list):
            assert isinstance(from_steps, Step)
            from_steps = [from_steps]
        for step in from_steps:
            assert isinstance(step, Step)
            assert step.name in self.steps, step.name
            self.splits[step.name].add(to_step.name)
            self.merges[to_step.name].add(step.name)

    def connect(self, from_steps, to_step, *args, **kwargs):
        assert isinstance(to_step, Step)
        self._connect_steps(from_steps, to_step)
        self._ip_args[to_step.name] = (args, kwargs)

    def start_with(self, step, *args, **kwargs):
        assert step.name in self.steps
        assert isinstance(step, Step)
        self.starting_step_name = step.name
        self._ip_args[step.name] = (args, kwargs)

    def _gather_data(self, step):
        logger.debug("Gathering data for %s", step.name)
        key = step.name
        assert key in self._ip_args
        args, kwargs = self._ip_args[key]
        args = [a if not isinstance(
            a, (Future, FutureLambda, FutureFlowInput)) else a.result()
            for a in args]
        kwargs = {k: v if not isinstance(
            v, (Future, FutureLambda, FutureFlowInput)) else v.result()
            for k, v in kwargs.iteritems()}
        return args, kwargs

    def trigger_children(self, step):
        with self._child_lock:
            for name in self.splits[step.name]:
                self._pending_parents_count[name] -= 1
                if self._pending_parents_count[name] == 0:
                    logger.debug("Step %s ready, count : %s", name, self._pending_parents_count[name])
                    e = ThreadPoolExecutor(max_workers=1)
                    self.step_run_result[name] = e.submit(self._run_step, self.steps[name])
                else:
                    logger.debug("Step %s not ready, count : %s", name, self._pending_parents_count[name])

    def _run_step(self, step):
        logger.debug("Starting to Run : %s", step.name)
        args, kwargs = self._gather_data(step)
        logger.debug("Args and kwargs : %s, %s", args, kwargs)
        # recording the args and kwargs for debugging purposes
        if self.testing_enabled:
            self.ip_args_for_step[step.name] = (args, kwargs)
        step.execute(*args, **kwargs)
        if step.status == FAILED:
            logger.error("Failed to run Step: %s" % step.name)
            raise step.output[0], None, step.output[1]
        logger.debug("Done Step execution : %s", step.name)
        self.trigger_children(step)

    def run_flow(self, **kwargs):
        self._reset()
        self.data_store.update(kwargs)

        starting_step = self.steps[self.starting_step_name]
        with ThreadPoolExecutor(max_workers=2) as e:
            self.step_run_result[starting_step.name] = e.submit(
                self._run_step, starting_step)
            e.submit(self.print_step_status())
        logger.debug("Finished Running Flow")
        return self.output

    def print_step_status(self):
        done_count = 0
        while done_count < len(self.steps):
            status = "\n"
            done_count = 0
            for name, future_res in self.step_run_result.items():
                status += "%-20s : %s\n" % (name, self.steps[name].status)
                if future_res.done():
                    done_count += 1
                    if future_res.exception() is not None:
                        logger.debug(status)
                        return future_res.result(timeout=10)
        logger.debug(status)

    def _reset(self):
        self.data_store = {}
        self.step_run_result = {}
        for step in self.steps.values():
            step.reset()
        # re-setting the pending_parents_count map
        for name in self.steps.keys():
            if name != self.starting_step_name:
                self._pending_parents_count[name] = len(self.merges[name])

    def _log_visit(self, step_name, visits):
        visits[step_name] += 1
        if visits[step_name] > len(self.merges[step_name]):
            raise ValueError("Cycle detected in the flow")
        for step_name in self.splits[step_name]:
            self._log_visit(step_name, visits.copy())

    def validate(self):
        """ Validate the flow for basic checks
        this method runs a series of checks against the flow object
        to verify that it conforms to all the necessary basic requirements
        """
        self._reset()
        # assert these is a starting step
        if self.starting_step_name is None:
            raise ValueError("Flow does not have a starting step")

        # assert startingstep has no parents
        if self.merges[self.starting_step_name] != set([]):
            raise ValueError("Starting step cannot have parent steps")

        # assert no cycles in the flow
        visits = defaultdict(int)
        for step_name in self.splits[self.starting_step_name]:
            self._log_visit(step_name, visits)

        # all steps in the flow have some connections
        for name, step in self.steps.items():
            if name != self.starting_step_name:
                if not self.merges[name]:
                    raise ValueError("Step %s is missing parent" % name)

        # the op of the flow is one of the step's output
        output = False
        for name, step in self.steps.items():
            if self._output == step.output:
                output = True
                break
        if not output:
            raise ValueError(
                "Flow's output is not set to be output of any of its steps")


class FlowTester(object):

    def __init__(self, flow_factory, *args, **kwargs):
        """ A class to perform simplify testing a flow-factory

        flow_factory : the flow-factory method that is supposed to be tested
        *args, **kwargs : the ip args that are supposed to be provided
                          to the flow factory to get the flow object

        """
        self.flow = flow_factory(*args, **kwargs)
        self.flow.enable_test_mode()

    def mock_step(self, step_name, return_value=None):
        """Mock a step in the Flow

        Args:
            step_name : name of the step that should be mocked
            as part of the flow exectuion

            return_value : The value that should be set
            as the output of the step.

        The mock created is an autospec of the original method and
        hence will require the right number of args during execution

        """
        assert step_name in self.flow.steps, "Step %s not part of flow " % step_name
        step = self.flow.steps[step_name]
        func_to_exec = step._get_func_to_exec()
        mock_func = create_autospec(func_to_exec, return_value=return_value)
        step._get_func_to_exec = Mock(return_value=mock_func)

    def run_in_test_mode(self, **kwargs):
        """Run the flow in a test mode, so that we can record
        all inputs and outputs of each step in the flow

        Args :
            **kwargs : all kwargs as reqd by the run_flow method of the flow

        Returns:
            step_ip_args : dict mapping from step_name to ip_args for the step
            execution. The values are a tuple (arg, kwargs)

            step_op_results : dict mapping from step_name to the result of the
            step's execution.

        The method validates the flow before running it.
        """
        self.flow.validate()
        self.flow.run_flow(**kwargs)
        results_for_step = {
            name: step.output for name, step in self.flow.steps.items()
        }
        return self.flow.ip_args_for_step, results_for_step
