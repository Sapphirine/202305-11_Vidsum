from django.core.files.storage import default_storage
import tempfile
import os
import uploader.config as config

def read_uploaded_file(uploaded_file):
    uploaded_vid_dir = config.UPLOADED_VID_DIR
    
    with tempfile.NamedTemporaryFile(delete=False) as tmpFile:
        for chunk in uploaded_file.chunks():
            tmpFile.write(chunk)
        
        if not os.path.exists(uploaded_vid_dir):
            os.makedirs(uploaded_vid_dir)
        vid_save_path = os.path.join(uploaded_vid_dir, uploaded_file.name)
        if not os.path.exists(vid_save_path):
            default_storage.save(vid_save_path, tmpFile)
    
    return tmpFile, vid_save_path