from affine.model.base import Base, session
from affine.model._sqla_imports import Column, Integer, VARCHAR, ForeignKey, DateTime, backref, relationship
from datetime import datetime

__all__ = [
    'Stream', 'StreamState'
]


class Stream(Base):
    __tablename__ = "streams"
    id = Column(Integer, nullable=False, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, default=False)
    shard_width = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime)
    deleted_at = Column(DateTime)

    states = relationship('StreamState', backref="stream")


class StreamState(Base):
    __tablename__ = "stream_states"
    id = Column(Integer, nullable=False, primary_key=True)
    stream_id = Column(Integer, ForeignKey('streams.id'), nullable=False)
    processor = Column(VARCHAR(128), nullable=False, default=False)
    shard_id = Column(VARCHAR(128), nullable=False, default=False)
    sequence_number = Column(VARCHAR(256))
    updated_at = Column(DateTime)

    @classmethod
    def get_or_create(cls, processor, stream_name, shard_id):
        astream = Stream.by_name(stream_name)
        if (astream is None):
            return None
        stream_state = cls.query.filter_by(stream_id=astream.id, shard_id=shard_id, processor=processor).first()
        if (not stream_state):
            kwargs = dict(processor=processor, stream_id=astream.id, shard_id=shard_id, updated_at=datetime.utcnow())
            stream_state = cls.create(**kwargs)
        return stream_state








