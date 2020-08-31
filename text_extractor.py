from logging import getLogger
import math
import os
import shutil
import tempfile

from affine.aws import s3client
from affine import config
from affine.model import session
from affine.video_processing import run_cmd, extract_audio, video_duration, VideoProcessingError
from affine.video_processing.extract import RAW_AUDIO_FORMAT

TEXT_EXTRACT_CMD  = '%(bin_dir)s/raw2text %(raw)s %(hmm)s %(lm)s %(dic)s %(text)s'
FFMPEG_EXTRACT_AUDIO_CLIP = '%(bin_dir)s/ffmpeg' + RAW_AUDIO_FORMAT + '-i %(infile)s -vn' + RAW_AUDIO_FORMAT + '-ss %(secs)d -t %(clip_length)d %(outfile)s' 

logger = getLogger(__name__)


class TextExtractor(object):
    def __init__(self, video, video_path):
        self.video = video
        self.video_path = video_path

        self.base_dir = tempfile.mkdtemp()
        self.transcript_path = os.path.join(self.base_dir, 'transcript.txt')

        self.model_dir = os.path.join(config.scratch_detector_path(), 'asr') 
        self.hmm = os.path.join(self.model_dir, 'hmm', 'hub4wsj_sc_8k')
        self.lm = os.path.join(self.model_dir, 'lm', 'hub4.5000.DMP')
        self.dic = os.path.join(self.model_dir, 'lm', 'cmu07a.dic')
        
    def transcribe_video(self):
        """Main function that performs speech to text and uploads transcript to s3"""
        try:
            self.generate_transcript()
            self.upload_transcript()
        finally:
            self.clean()

    def generate_transcript(self):
        """Process our video file and write its transcript to self.transcript_path"""
        self.grab_s3_files()

        clip_length = 10
        duration = video_duration(self.video_path)
        chunks = int(math.ceil(duration / float(clip_length)))

        audio_path = os.path.join(self.base_dir, 'audio.raw')
        audio_clip_path = os.path.join(self.base_dir, 'clip.raw')
        text_clip_path = os.path.join(self.base_dir, 'clip.txt')

        logger.info("generating transcript")

        try:
            extract_audio(self.video_path, audio_path)
        except VideoProcessingError as e:
            if 'does not contain any stream' not in str(e):
                raise
            logger.error("Video %s has no audio stream", self.video)
            # Empty transcript because we haven no audio
            open(self.transcript_path, 'w').close()
            return

        for chunk in xrange(chunks):
            start = chunk * clip_length
            params = {  'infile': audio_path,
                        'outfile': audio_clip_path,
                        'secs': start, 
                        'clip_length' : clip_length,
                        'bin_dir': config.bin_dir(),
                     }
            run_cmd(FFMPEG_EXTRACT_AUDIO_CLIP, params)

            self.extract_text(audio_clip_path, text_clip_path)
            self.append_to_transcript(text_clip_path)

            os.unlink(audio_clip_path)
            os.unlink(text_clip_path)
        logger.info("done generating transcript")

    def extract_text(self, audio_path, text_path):
        """Run speech model on a audio to generate transcript"""
        params = {
            'bin_dir' : config.bin_dir(),
            'raw': audio_path,
            'text' : text_path,
            'hmm' : self.hmm, 
            'lm' : self.lm, 
            'dic' : self.dic, 
        }

        try:
            run_cmd(TEXT_EXTRACT_CMD, params)
        except VideoProcessingError:
            # ASR occasionally fails with an error finding the start node or similar
            # In that case, save an empty file as transcript for the chunk
            # run_cmd already logged the stdout/ stderr from the failed proc
            logger.error("Failed running ASR on chunk for video %s", self.video)
            open(text_path, 'w').close()

    def append_to_transcript(self, text_file):
        """Append contents of a text file to our transcript"""
        with open(self.transcript_path, 'a') as destination:
            with open(text_file, 'rb') as infile:
                shutil.copyfileobj(infile, destination)

    def clean(self):
        """Remove temporary files"""
        shutil.rmtree(self.base_dir)

    def upload_transcript(self):
        """Upload transcript to s3"""
        self.video.upload_transcript(self.transcript_path)
        self.video.mark_transcript_uploaded()
        session.flush()

    def grab_s3_files(self):
        """Download model files from s3 and untar them to destination dir."""
        bucket = config.s3_detector_bucket()
        logger.info("downloading files from S3")
        s3client.download_tarball(bucket, 'asr_model', self.model_dir)
        logger.info("done downloading files from S3")
