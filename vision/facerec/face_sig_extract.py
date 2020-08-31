import os
import shutil
import numpy as np
import cv2.cv as cv
from tempfile import NamedTemporaryFile, mkdtemp

from affine import config
from affine.video_processing import run_cmd

FACE_EXTRACT_CMD  = '%s/aff_face_features_extract -t -gray %s %s'


class FaceSigExtractor(object):
    MAX_PATCHES_PER_ITER = 100

    def _run(self, patches):
        infofile = self._make_infofile(patches)
        outdir, extract_filename, parts_filename = self._make_output()
        self._run_extraction(infofile, extract_filename)
        infofile.close()
        return outdir, extract_filename, parts_filename

    def extract_signatures(self, patches):
        outdir, extract_filename, _ = self._run(patches)
        signatures = self._capture_signatures(extract_filename)
        shutil.rmtree(outdir)
        return signatures

    def extract(self, patches):
        outdir, extract_filename, parts_filename = self._run(patches)
        signatures = self._capture_signatures(extract_filename)
        face_infos = self._capture_face_infos(parts_filename)
        shutil.rmtree(outdir)
        return signatures, face_infos

    def batch_extract(self, patches):
        all_signatures = all_face_infos = []
        for i in xrange(len(patches), self.MAX_PATCHES_PER_ITER):
            signatures, face_infos = self.extract(patches[i:i+self.MAX_PATCHES_PER_ITER])
            all_signatures += signatures
            all_face_infos += face_infos
        return all_signatures, all_face_infos

    def _capture_signatures(self, extract_filename):
        # the first row is unused, rest are actual descriptors
        signatures_arr = np.asarray(cv.Load(extract_filename))[1:, :]
        return signatures_arr.transpose().tolist()

    def _capture_face_infos(self, parts_filename):
        face_infos = []
        with open(parts_filename) as f:
            for line in f:
                line = line.strip()
                if line != 'Skipped':
                    float_list = map(float, line.split(','))
                    confidence = float_list[-1]
                    parts = float_list[:-1]
                else:
                    confidence = parts = None
                face_infos.append((confidence, parts))
        return face_infos

    def _make_output(self):
        outdir = mkdtemp()
        extract_filename = os.path.join(outdir, 'out.xml')
        fname, _ = os.path.splitext(extract_filename)
        parts_filename = fname + '_parts.txt'
        return outdir, extract_filename, parts_filename

    def _make_infofile(self, patches):
        infofile = NamedTemporaryFile()
        for image_path, (x, y, w, h) in patches:
            infofile.write('%s %d %d %d %d -1 0\n' % (image_path, x, y, w, h))
        infofile.flush()
        return infofile

    def _run_extraction(self, infofile, extract_filename):
        cmd = FACE_EXTRACT_CMD % (config.bin_dir(), infofile.name, extract_filename)
        run_cmd(cmd)
