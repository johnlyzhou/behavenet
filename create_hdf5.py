from behavenet.data.preprocess import build_hdf5

save_file = '/Users/johnzhou/code/behavenet/results/free.hdf5'
video_file = '/Users/johnzhou/code/VAME/results/videos/aligned_video.mp4'
label_file = '/Users/johnzhou/code/VAME/results/shifted_dlc.csv'

build_hdf5(save_file, video_file, label_file=label_file, pose_algo='dlc')
