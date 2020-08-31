from sqlalchemy import Column, Integer
from affine.model._sqla_imports import Unicode
from affine.model.base import Base


__all__ = ['Placement']


class Placement(Base):
    __tablename__ = 'placements'
    id = Column(Integer, nullable=False, primary_key=True)
    name = Column(Unicode(255), nullable=False)
