import json
import os
import socket
import SocketServer
import subprocess
import traceback
from logging import getLogger

from jsonschema import validate

from affine import config
from affine.model.detection import LdaModel
from affine.retries import retry_operation, memoize

SERVER_CHECK_INTERVAL = 5
SERVER_CHECK_ATTEMPTS = 5
CLIENT_SCHEMA = {
    "type": "object",
    "properties": {
        "tm_id": {"type": "integer"},
        "input_str": {"type": "string"},
        "mallet_args": {"type": "string"}},
    "required": ["tm_id", "input_str", "mallet_args"],
    "additionalProperties": False}

logger = getLogger(__name__)


class LdaRequestHandler(SocketServer.StreamRequestHandler):

    def handle(self):
        response = ''
        try:
            json_string = self.rfile.readline().strip()
            json_dict = json.loads(json_string)
            if 'cmd' in json_dict:
                assert json_dict['cmd'] == 'POLL'
                assert MalletServerManager._poll_server()
                response = json.dumps({"status": "ok", "response": 'True'})
            else:
                response = MalletServerManager.query_lda_server(json_dict)
        except Exception:
            logger.exception('Error handling request')
            # sending traceback to client is not really a good thing to do
            response = json.dumps({"status": "error", "response":
                                   traceback.format_exc()})
        finally:
            self.wfile.write(response)


class LdaServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass


class MalletServerManager(object):

    # mallet server params
    QUERY_TIMEOUT = 120.0
    MALLET_SERVER_PORT = 7070
    KILL_CMD = {"cmd": "kill"}
    POLL_CMD = {"cmd": "poll"}

    @classmethod
    def start_server(cls):
        # check if server already up
        if cls._poll_server():
            logger.info("Server already up")
            return

        topic_model_dir = os.path.join(config.bin_dir(), 'topic_model')
        cmd = ['java', '-Xmx4096m', '-jar', 'LDAServer.jar', '--port',
               str(cls.MALLET_SERVER_PORT)]
        server_log = os.path.join(config.log_dir(), 'lda_server.log')
        log_handle = open(server_log, 'a')
        subprocess.Popen(cmd, cwd=topic_model_dir, stdout=log_handle,
                         stderr=subprocess.STDOUT)
        cls._wait_for_server()

    @classmethod
    def _wait_for_server(cls):
        try:
            retry_operation(cls._poll_server, raise_exception=True, error_class=socket.error,
                            num_tries=SERVER_CHECK_ATTEMPTS, sleep_time=SERVER_CHECK_INTERVAL,
                            error_message='Mallet server not ready yet', with_traceback=False)
        except socket.error:
            # stop server subprocess and raise exception
            cls.stop_server()
            logger.exception('Timed out waiting for Mallet server to come up')
            raise Exception('Timed out waiting for Mallet server to come up')

    @classmethod
    def _poll_server(cls, raise_exception=False):
        ''' Poll the Mallet server. Return True if server is up, False otherwise '''
        try:
            cls._query_server(cls.POLL_CMD, timeout=10.0)
            return True
        except socket.error:
            if raise_exception:
                raise
            return False

    @classmethod
    def _inverse_poll_server(cls):
        if cls._poll_server():
            raise Exception("Server still up")

    @classmethod
    def stop_server(cls):
        #check if server has gone away already
        if not cls._poll_server():
            logger.info("Server either stopped or not reachable")
            return
        logger.info("stopping Mallet server process")
        cls._query_server(cls.KILL_CMD)
        retry_operation(cls._inverse_poll_server, error_class=Exception,
                    num_tries=SERVER_CHECK_ATTEMPTS, sleep_time=SERVER_CHECK_INTERVAL,
                    error_message='Server still up', with_traceback=False)
        logger.info("done stopping Mallet server process")

    @classmethod
    def query_lda_server(cls, json_dict):
        validate(json_dict, CLIENT_SCHEMA)
        tm_id = json_dict['tm_id']
        lda_model = memoized_lda_get(tm_id)
        assert lda_model, 'Lda model %d does not exist!' % (tm_id)
        lda_model.grab_files()
        json_dict.update({'inferencer_file' : lda_model.local_path('inferencer_file'),
            'pipe_file' : lda_model.local_path('pipe_file')})
        return cls._query_server(json_dict, timeout=cls.QUERY_TIMEOUT)

    @classmethod
    def _query_server(cls, json_dict, timeout=120.0):
        json_string = json.dumps(json_dict)
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.settimeout(timeout)
        clientsocket.connect(('localhost', cls.MALLET_SERVER_PORT))
        clientsocket.sendall(json_string + '\n')
        return clientsocket.recv(1024)


@memoize
def memoized_lda_get(tm_id):
    lda_model = LdaModel.get(tm_id)
    assert lda_model is not None
    return lda_model
