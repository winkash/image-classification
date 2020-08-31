import multiprocessing as mp
from logging import getLogger

from affine.parallel import StartedProcess

logger = getLogger(name=__name__)

__all__ = ['IterFlow', 'ParallelFlow', 'QueueItem', 'Done']


class Done(object):

    """ The done object notifies the worker-processes to stop and exit"""


class QueueItem(object):

    """An item for the Parallel Flow queue"""

    def __init__(self, idx, data):
        self.idx = idx
        self.data = data


class ParallelFlow(object):

    """ Parallel version of the iterative flow """

    def __init__(self, flow_factory, max_workers=None, enable_log=False):
        self.factory = flow_factory
        self.max_workers = max_workers or mp.cpu_count()
        self.factor = 5
        self.enable_log = enable_log

    # TODO : Add profiling tools
    def operate(self, data, *args, **kwargs):
        """
        Args:
            data : an iterable
                   the iterative_flow's operate method will run on each chunk
                   of data as a separate process
            Returns :
                    batch output
        """
        # initializing queues
        self._input_queue = mp.Queue(maxsize=self.factor * self.max_workers)
        self._output_queue = mp.Queue()

        # Start the queue_writer
        writer = StartedProcess(self.queue_writer, data)
        # start the workers
        workers = [StartedProcess(self.worker, *args, **kwargs)
                   for _ in xrange(self.max_workers)]
        try:
            # start the results_reader
            results = self.results_reader()
        except Exception, e:
            # if we failed to read results
            # either the worker returned an Exception since it must have failed
            # or smething else went wrong in the reader itself
            # In either case we need to kill all the workers
            # as well as the queue_writer
            self._kill_all_workers(workers)
            # Note: Killing the writer may corrupt the queue
            # however, since we are in cleanup mode, ignoring for now
            writer.terminate()
            logger.exception(
                "Failed to read all results, a child process must have failed.")
            raise e
        else:
            # wait for all to finish
            writer.join()
            for w in workers:
                w.join()
                assert w.exitcode == 0, w.exitcode

            return results

    def _kill_all_workers(self, worker_processes):
        for proc in worker_processes:
            proc.terminate()

    def worker(self, *args, **kwargs):
        flow = self.factory(*args, **kwargs)
        if self.enable_log:
            flow.enable_log()
        # process items from the queue until we get a Done
        while True:
            item = self._input_queue.get()
            if isinstance(item, Done):
                self._output_queue.put(Done())
                break

            try:
                flow.run_flow(ip_data=item.data)
            except Exception, e:
                self._output_queue.put(e)
                break
            else:
                op_item = QueueItem(item.idx, flow.output)
                self._output_queue.put(op_item)

    def queue_writer(self, data):
        for idx, d in enumerate(data):
            # a blocking put will keep trying indefinitely
            # if the size of queue is equal to its max-size
            item = QueueItem(idx, d)
            self._input_queue.put(item)
        # Once we have put all messages into the queue
        # We should put the `Done` messages so that the
        # child processes finish processing and finish
        for _ in xrange(self.max_workers):
            self._input_queue.put(Done())

    def results_reader(self):
        results = []
        done_count = 0
        # wait for all child processes to finish
        # once all are finished, gather all results
        while done_count < self.max_workers:
            res = self._output_queue.get()
            if isinstance(res, Done):
                done_count += 1
                continue
            elif isinstance(res, Exception):
                raise res
            results.append(res)

        results = [
            row.data for row in sorted(results, key=lambda x: x.idx)]
        return results


class IterFlow(object):

    """ A wrapper to perform iterative operations using a flow """

    def __init__(self, factory):
        """The Iter-Flow constructor
        Args:
            factory : A method that builds anf returns a flow object
                      on being invoked.
        """
        self.factory = factory

    # TODO : Add ability to add other kwargs which can be passed to
    #        flow's run_flow for each iteration when involing operate
    # TODO : Fix too much alerting for iter_flows
    def operate(self, data):
        """
        Args:
            data : any iterable, the flow.run_flow method
                   will operate on each data point of this iterable
                   [ can also be a genearator ]
        Returns:
            batch output
        """
        flow = self.factory()
        output = []
        for d in data:
            flow.run_flow(ip_data=d)
            output.append(flow.output)
        return output
