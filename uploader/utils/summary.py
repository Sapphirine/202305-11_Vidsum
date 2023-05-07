from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector
import cv2
from uploader.networks.CNN import ResNet
from uploader.networks.models import DSN
import numpy as np
from tqdm import tqdm
import os
import uploader.config as config
from uploader.utils.vsum_tools import generate_summary
import h5py
import torch
import shutil

def get_scenes(video):
    video_manager = VideoManager([video])
    video_manager.set_downscale_factor()
    
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    
    return scenes

def extract_feature(model, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res_pool = model(frame)
    frame_features = res_pool.cpu().data.numpy().flatten()
    
    return frame_features
    
def get_change_points(scenes):
    change_points = []
    frames_per_seg = []
    for scene in scenes:
        start_frame, end_frame = scene[0].frame_num, scene[1].frame_num-1
        n_frames_per_seg = end_frame - start_frame + 1

        change_points.append([start_frame, end_frame])
        frames_per_seg.append(n_frames_per_seg)
    
    return change_points, frames_per_seg
    
def generate_data(model, video_path, frames_save_dir):
    
    video_basename = os.path.basename(video_path).split('.')[0]

    if not os.path.exists(os.path.join(frames_save_dir, video_basename)):
        os.makedirs(os.path.join(frames_save_dir, video_basename))

    video_capture = cv2.VideoCapture(video_path)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    picks = []
    video_feat = None
    video_feat_for_train = None
    for frame_idx in tqdm(range(n_frames-1)):
        success, frame = video_capture.read()
        if success:
            frame_feat = extract_feature(model, frame)
            if frame_idx % 15 == 0:
                picks.append(frame_idx)

                if video_feat_for_train is None:
                    video_feat_for_train = frame_feat
                else:
                    video_feat_for_train = np.vstack((video_feat_for_train, frame_feat))


            if video_feat is None:
                video_feat = frame_feat
            else:
                video_feat = np.vstack((video_feat, frame_feat))

            img_filename = "{}.jpg".format(str(frame_idx).zfill(5))
            cv2.imwrite(os.path.join(frames_save_dir, video_basename, img_filename), frame)
        else:
            break

    video_capture.release()
    
    return list(video_feat_for_train), np.array(list(picks)), n_frames, fps

def vid_to_h5(feat, picks, n_frames, fps, change_pts, frames_per_seg, h5_save_path):
    if os.path.exists(h5_save_path):
        os.remove(h5_save_path)
    h5f = h5py.File(h5_save_path, "w")
    h5f.create_group('video_1')
    
    h5f['video_1']['features'] = list(feat)
    h5f['video_1']['picks'] = np.array(list(picks))
    h5f['video_1']['n_frames'] = n_frames
    h5f['video_1']['fps'] = fps
    h5f['video_1']['change_points'] = np.array(change_pts)
    h5f['video_1']['n_frame_per_seg'] = np.array(frames_per_seg)
    
    h5f.close()
    
def read_h5(h5_path):
    data = h5py.File(h5_path, "r")
    data_keys = [key for idx, key in enumerate(data.keys())]
    
    return data, data_keys
    
def get_sum_frames(frm_dir, summary, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx, val in enumerate(summary):
        if val == 1:
            frm_name = str(idx+1).zfill(5) + '.jpg'
            frm_path = os.path.join(frm_dir, frm_name)
            sum_path = os.path.join(save_dir, frm_name)
            shutil.copyfile(frm_path, sum_path)

def summarize(vid_path):
    model = ResNet()
    
    scenes = get_scenes(vid_path)
    change_points, frames_per_seg = get_change_points(scenes)
    
    h5_save_path = os.path.join(config.H5_DATA_DIR, os.path.basename(vid_path).replace(".mp4", ".h5").replace(".MP4", ".h5"))
    if os.path.exists(h5_save_path) == False:
        feat, picks, n_frames, fps = generate_data(model, vid_path, config.FRAME_ROOT_DIR)
        vid_to_h5(feat, picks, n_frames, fps, change_points, frames_per_seg, h5_save_path)  
    data, data_keys = read_h5(h5_save_path)
    
    summary_h5_path = h5_save_path.replace(".h5", "_result.h5")
    
    # load model
    model = DSN(in_dim=1024, hid_dim=256, num_layers=1, cell='lstm')
    checkpoint = torch.load(config.MODEL_CKPT)
    model.load_state_dict(checkpoint)
    
    if os.path.exists(summary_h5_path):
        return summary_h5_path
    
    # run inference
    with torch.no_grad():
        model.eval()
        
        h5_res = h5py.File(summary_h5_path, "w")
        for key_idx, key in enumerate(data_keys):
            seq = data[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()
            
            cps = data[key]['change_points'][...]
            num_frames = data[key]['n_frames'][()]
            nfps = data[key]['n_frame_per_seg'][...].tolist()
            positions = data[key]['picks'][...]
            
            machine_summary = generate_summary(probs, cps, num_frames, nfps, positions, method='rank')
            
            h5_res.create_dataset(key + '/score', data=probs)
            h5_res.create_dataset(key + '/machine_summary', data=machine_summary)    
            h5_res.close()
            
    return summary_h5_path

def visualize(summary_h5_path):
    vid_name = os.path.basename(summary_h5_path).replace("_result.h5", "")
    
    summary = h5py.File(summary_h5_path, "r")['video_1']['machine_summary'][...]
    frames_dir = os.path.join(config.FRAME_ROOT_DIR, vid_name)
    summary_frames_dir = os.path.join(config.FRAME_ROOT_DIR, vid_name + "_summary")
    
    selected_frms = []
    if not os.path.exists(summary_frames_dir):
        os.makedirs(summary_frames_dir)
        
        for idx, val in enumerate(summary):
            if val == 1:
                frame_name = str(idx+1).zfill(5) + ".jpg"
                frame_path = os.path.join(frames_dir, frame_name)
                sum_path = os.path.join(summary_frames_dir, frame_name)
                try:
                    shutil.copyfile(frame_path, sum_path)
                    selected_frms.append(idx+1)
                except FileNotFoundError:
                    pass
        
    vid_path = write_video(vid_name, summary_frames_dir)
             
    return vid_path

def write_video(vid_name, frm_dir):
    h5_data_path = os.path.join(config.H5_DATA_DIR, vid_name + ".h5")
    fps = h5py.File(h5_data_path, "r")['video_1']['fps'][()]

    frm_sorted = sorted([f for f in os.listdir(frm_dir) if f.endswith(".jpg")])
    
    img = cv2.imread(os.path.join(frm_dir, frm_sorted[0]))
    height, width, channels = img.shape

    vid_save_file = os.path.join(config.SUMMARY_PATH, vid_name + ".mp4")
    vid = cv2.VideoWriter(vid_save_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frm in frm_sorted:
        img = cv2.imread(os.path.join(frm_dir, frm))
        vid.write(img)

    vid.release()
    cv2.destroyAllWindows()
    
    return vid_save_file