from datetime import datetime
from affine.model.base import *
from affine.model._sqla_imports import *
from affine.model.labels import *
from affine.model.videos import *
import PIL.Image, PIL.ImageDraw

__all__ = ['Box', 'FaceInfo']


class Box(Base):
    __tablename__ = 'boxes'

    SIGNATURE_SIZE = 1937

    id = Column(Integer, primary_key=True)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=False)
    timestamp = Column(Integer, nullable=False)
    signature = Column(Boolean, nullable=False, default=False)
    box_type = Column(Enum('Face', 'Text', 'Logo'), nullable=False)

    video = relation('Video', backref = backref('boxes', passive_deletes=True))

    def __unicode__(self):
        return u'<%s Box for (video_id=%s,timestamp=%s)>'\
            % (self.box_type, self.video_id, self.timestamp)

    @classmethod
    def get_or_create(cls, x, y, width, height, video, timestamp, box_type):
        try:
            result = session.execute('''
                INSERT INTO `%s`
                    (x, y, width, height, video_id, timestamp, box_type)
                    VALUES (%d, %d, %d, %d, %d, %d, "%s")
            ''' % (cls.__tablename__, x, y, width, height, video.id, timestamp, box_type))
            box_id = result.lastrowid
            result.close()
        except IntegrityError:
            query = session.query(cls.id).filter_by(x=x,
                                                    y=y,
                                                    width=width,
                                                    height=height,
                                                    video_id=video.id,
                                                    timestamp=timestamp,
                                                    box_type=box_type)
            box_id = query.scalar()
            assert box_id is not None
        session.expire(video, ['boxes'])
        return box_id

    def show(self):
        img = Image(self.video_id, self.timestamp).pil_image()
        draw = PIL.ImageDraw.Draw(img)
        draw.rectangle([(self.x, self.y), (self.x+self.width, self.y+self.height)])
        if self.face_info:
            parts = self.face_info.parts
            for i in range(0,len(parts),2):
                box = [(parts[i]-1, parts[i+1]-1),(parts[i]+1, parts[i+1]+1)]
                draw.ellipse(box)
            for i, j in [(0,1), (2,3), (1,2), (7,8)]:
                x = (parts[2*i] + parts[2*j])/2
                y = (parts[2*i+1] + parts[2*j+1])/2
                box = [x-1, y-1, x+1, y+1]
                draw.ellipse(box)

        img.show()
        del draw

    def download_image(self, path, extension='.jpg'):
        if extension not in path:
            path += extension
        self.video.download_image(self.timestamp, path)
        return path


class FaceInfo(Base):
    __tablename__ = 'face_info'
    box_id = Column(Integer, ForeignKey('boxes.id'), primary_key=True)
    confidence = Column(Float, nullable=False)
    _parts = Column('parts', String(255), nullable=False)

    box = relation('Box', backref=backref('face_info', cascade='all,delete-orphan', uselist=False))

    def __unicode__(self):
        return u'<FaceInfo for Box(%s) with confidence: %s>' %(self.box_id, self.confidence)

    @property
    def parts(self):
        return map(float, self._parts.split(','))

    @parts.setter
    def parts(self, parts_list):
        assert len(parts_list) == 18, 'Need (x,y) for 9 points [18 values], got %s values' %len(parts_list)
        self._parts = ','.join(map(str, parts_list))
