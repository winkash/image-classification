import socket
import json

from affine import config


class LdaClient(object):

    INFER_TIMEOUT = 180.0
    ERROR_KEY = 'error'
    RESPONSE_KEY = 'response'

    @classmethod
    def poll(cls):
        args_dict = {'cmd': 'POLL'}
        response_json = cls._query_server(args_dict, timeout=10.0)
        response_dict = json.loads(response_json)
        assert response_dict['status'] == 'ok'
        assert response_dict['response'] == 'True'
        return True

    @classmethod
    def infer_topics(cls, input_str, tm_id, mallet_args=' '):
        assert isinstance(input_str, basestring)
        if isinstance(input_str, unicode):
            input_str = input_str.encode('utf-8')
        args_dict = {'input_str': input_str, 'tm_id': tm_id,
                     'mallet_args': mallet_args}
        return cls.query_and_parse(args_dict)

    @classmethod
    def query_and_parse(cls, args_dict):
        response_json = cls._query_server(args_dict, timeout=cls.INFER_TIMEOUT)
        return cls._parse_json(response_json)

    @classmethod
    def _parse_json(cls, response_json):
        response_dict = json.loads(response_json)
        if response_dict['status'] == cls.ERROR_KEY:
            raise MalletServerException(response_dict[cls.RESPONSE_KEY])
        else:
            response = response_dict[cls.RESPONSE_KEY]
            assert isinstance(response, dict)
            # convert unicode topic ids to integer
            topic_dist = {int(i): response[i] for i in response}
            return topic_dist

    @classmethod
    def _query_server(cls, args_dict, timeout=180.0):
        mgr_hostname = config.get('lda_server.host')
        mgr_port = config.get('lda_server.port')
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.settimeout(timeout)
        clientsocket.connect((mgr_hostname, mgr_port))
        try:
            clientsocket.sendall(json.dumps(args_dict)+'\n')
            return clientsocket.recv(1024)
        finally:
            clientsocket.close()


class MalletServerException(Exception):
    pass
