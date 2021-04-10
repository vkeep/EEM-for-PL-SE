import os

# front-end parameter settings
fs = 16000
win_size = int(0.020*fs)   # default: 20ms window length
fft_num = 320   # FFT point, default: True
overlap_ratio = 0.5   # default: 50% overlap
win_shift = int(win_size * (1 - overlap_ratio))
chunk_length = int(0.5*fs)   # the max utterance length
is_scale = True   # whether to norm the waveform variance into 1, default: True


# network configurations
stride = (1, 2)   # the stride in time and frequency axis, default: (1, 2)
k = (2, 3)     # the kernel size in time and freqency axis, default: (2, 3)
ci = 4
c = 64   # the number of channels, default: 64
lstm_num = 256   # default: 256
is_causal = True  # whether to use causal setup, default: True
is_gate = True  # whether to use GLU for both encoder and decoder, default: True
stage_num = 3    # default: 5


# project parameter settings
# json_dir = '/media/liandong/generalized_timit/Json'
# file_path = '/media/liandong/generalized_timit'
json_dir = './Json'
file_path = 'F:/一些语音数据集/generalized_timit'
batch_size = 2
epochs = 50
lr = 1e-3
loss_dir = './LOSS/eem_se_stage_{}_loss.mat'.format(stage_num)
model_best_path = './BEST_MODEL/eem_se_causal_stage_{}_gate_{}_best_model.pth.tar'.format(stage_num, is_causal, is_gate)
is_cp = True
check_point_path = './CP_dir'
is_conti = False
conti_path = './CP_dir/checkpoint_epoch_9_eem_se_model.pth.tar'


multi_gpus = False
if multi_gpus:
    gpu_id = [0, 1]
else:
    gpu_id = 0

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)
