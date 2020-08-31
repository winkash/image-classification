from logging import getLogger
import struct

from boto.dynamodb2.fields import HashKey
from boto.dynamodb.types import Binary
from boto.dynamodb2.types import NUMBER

from affine import config
from affine.model import Box
from affine.aws.dynamodb import DynamoTableManager

__all__ = ['DynamoFaceSignatureClient', 'FaceSignatureClientError']

logger = getLogger(__name__)


class FaceSignatureClientError(Exception):
    """Error in DynamoFaceSignatureClient"""


class DynamoFaceSignatureClient(DynamoTableManager):
    """DynamoDB wrapper for converting data_dictionary to be DynamoDB
    compliant"""
    SIGNATURE_SIZE = 1937
    STRUCT_FORMAT = 'd' * SIGNATURE_SIZE

    def __init__(self):
        table_name = config.get('dynamo.face_signature_table_name')
        super(DynamoFaceSignatureClient, self).__init__(table_name)

    def create_table(self):
        super(DynamoFaceSignatureClient, self).create_table(
            schema=[HashKey('box_id', data_type=NUMBER)])

    def from_dynamo(self, item):
        sig = str(item['signature'])
        return list(struct.unpack(self.STRUCT_FORMAT, sig))

    def to_dynamo(self, data):
        signature = data['signature']
        try:
            signature = Binary(struct.pack(self.STRUCT_FORMAT, *signature))
        except Exception:
            raise FaceSignatureClientError("Malformed signature {}".format(signature))
        return {
            'box_id': data['box_id'],
            'signature': signature,
        }

    def put(self, box_id, signature):
        data_dictionary = {'box_id': box_id,
                           'signature': signature}
        super(DynamoFaceSignatureClient, self).put(data_dictionary)
        Box.query.filter_by(id=box_id).update(
                {"signature": True}, synchronize_session=False)

    def delete(self, box_id):
        super(DynamoFaceSignatureClient, self).delete(key=box_id)
        Box.query.filter_by(id=box_id).update(
                {"signature": False}, synchronize_session=False)

    def get_signatures(self, box_ids, raise_exp=True):
        result_dict = self.batch_get(box_ids, raise_exp=raise_exp)
        return [result_dict[box_id] for box_id in box_ids]

    def has_signature(self, box_id):
        """Check if we've a signature for the box_id. Returns bool """
        return self.has_item(box_id=box_id, consistent=True)

    def put_signatures(self, box_ids, face_signatures):
        data_dictionaries = []
        for box_id, sig in zip(box_ids, face_signatures):
            data_dictionaries.append({'box_id': box_id,
                                      'signature': sig})
        self.batch_put(data_dictionaries)
        Box.query.filter(Box.id.in_(box_ids)).update(
                {"signature": True}, synchronize_session=False)
