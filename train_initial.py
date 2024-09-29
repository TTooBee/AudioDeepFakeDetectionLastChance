import torch, torchaudio
import torchaudio.transforms as T

import numpy as np
import matplotlib.pyplot as plt

import os
import glob

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

import torch.nn as nn

import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/experiment1')

def load_audios(wav_files, frame_size, sample_rate):
    
    mfcc_frames_all = []
    
    # MFCC 변환 설정
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=40,               # 13개의 MFCC 계수 추출
        melkwargs={"n_fft": 1024,  # FFT 크기
                   "hop_length": 160,  # 프레임 간 이동 크기
                   "n_mels": 128,  # 멜 필터의 개수
                   "center": True}       
    )
    
    for audio_path in wav_files:
        waveform, _ = torchaudio.load(audio_path)
        mfcc = mfcc_transform(waveform)
        # 10개 프레임을 하나의 데이터셋으로 보자. 
        num_frames = 10
        start_frame = 0
        end_frame = start_frame + num_frames
        max_frame_num = (waveform.shape[1]/frame_size)
        while(end_frame < max_frame_num):
            
            mfcc_frames = mfcc[:, :, start_frame:end_frame]
            mfcc_frames_all.append(mfcc_frames)
            
            start_frame = start_frame + 1
            end_frame = end_frame + 1

    print(f'length of list mfcc_frames_all : {len(mfcc_frames_all)}')
    mfcc_frames_all_tensor = torch.cat(mfcc_frames_all, dim=0)
    
    return mfcc_frames_all_tensor


def preprocess(real_auido_path, fake_audio_path):
    print('preprocessing data..')
    real_wav_files = glob.glob(os.path.join(real_auido_path, "*.wav"))
    fake_wav_files = glob.glob(os.path.join(fake_audio_path, "*.wav"))
    
    real_wav_files.sort()
    fake_wav_files.sort()
        
    frame_size = 320
    sample_rate = 16000
    
    mfcc_frames_all_tensor_real = load_audios(real_wav_files, frame_size, sample_rate)
    mfcc_frames_all_tensor_fake = load_audios(fake_wav_files, frame_size, sample_rate)
    
    if mfcc_frames_all_tensor_real.shape[0] > mfcc_frames_all_tensor_fake.shape[0]:
        shorter_len = mfcc_frames_all_tensor_fake.shape[0]
        mfcc_frames_all_tensor_real = mfcc_frames_all_tensor_real[:shorter_len, :, :]
    elif mfcc_frames_all_tensor_real.shape[0] > mfcc_frames_all_tensor_fake.shape[0]:
        shorter_len = mfcc_frames_all_tensor_real.shape[0]
        mfcc_frames_all_tensor_fake = mfcc_frames_all_tensor_fake[:shorter_len, :, :]
        
    
    return mfcc_frames_all_tensor_real, mfcc_frames_all_tensor_fake
    


# 데이터 준비 완료
#######################################################################


class AudioDataset(Dataset):
    def __init__(self, mfcc_frames_all_tensor_real, mfcc_frames_all_tensor_fake): # 여기서 데이터는 mfcc_frames_all_tensor_real, mfcc_frames_all_tensor_fake 가 될 것임
        self.mfcc_frames_all_tensor_real = mfcc_frames_all_tensor_real
        self.mfcc_frames_all_tensor_fake = mfcc_frames_all_tensor_fake
        
        self.label_real = torch.ones(mfcc_frames_all_tensor_real.shape[0], 1)
        self.label_fake = torch.zeros(mfcc_frames_all_tensor_real.shape[0], 1)
        
    
        # 생성자에서 데이터를 합쳐서 하나의 데이터, 하나의 레이블로 만들어야 한다
        self.mfcc_frames_all_tensor = torch.cat((self.mfcc_frames_all_tensor_real, self.mfcc_frames_all_tensor_fake), dim=0)
        self.label = torch.cat((self.label_real, self.label_fake), dim=0)
        self.label = self.label.reshape(-1)
        
    def __len__(self):
        return len(self.mfcc_frames_all_tensor)
    
    def __getitem__(self, idx):
        return self.mfcc_frames_all_tensor[idx, :, :], self.label[idx]
        
# 데이터셋 준비 완료
##############################################################
# SimpleRNN = nn.RNN(input_size=13, hidden_size=3, batch_first=True, num_layers=2)
class SimpleRNN(nn.Module):
    def __init__(self, input_size=40, hidden_size=3, batch_first=True, num_layers=2):
        super(SimpleRNN, self).__init__()
        # 이후 여기서 모든 레이어를 정의해야 한다. 그리고 forward에서 이것들을 나열할 것이다. 
        # rnn 층 정의
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first, num_layers=num_layers)
        # 마지막에 적용할 fc 레잉 정의
        self.fc = nn.Linear(in_features=hidden_size, out_features=1) 
        # rnn 레이어의 마지막 출력만을 사용할 것이기 때문에, hidden size만큼만 입력으로 받음
        
    def forward(self, x):
        # print(f'shape of initial data : {x.shape}')
        # x가 처음 들어오면 (batch_size, 13, 10)의 형태일 것이다. 우선 (batch_size, 10, 13)으로 바꿔준다
        x = x.transpose(2, 1)
        # print(f'shape of transpoced data : {x.shape}')
        # rnn에 집어넣는다
        output, hidden = self.rnn(x) # 여기서 output만 쓸 것이다 output은 (1, 10, hidden_size)의 모양일 것이다.. 라고 생각했는데 여기서 batch 사이즈만큼 나오네?
        # print(f'shape of rnn output : {output.shape}') 
        output = output[:, -1, :] # -1은 마지막 요소라는 뜻 이러면 (1, hidden_size) 로 나옴. 이걸 fc층에 집어넣을 것임. 여기는 그럼 (batch_size, hidden_size)
        # rint(f'shape of last rnn output : {output.shape}')
        final_output = self.fc(output)
        # print(f'shape of final output : {final_output.shape}')
        final_output = final_output.reshape(-1)
        
        return final_output


# rnn 모델 정의 끝. 파이토치에서 rnn은 굉장히 간단하다. 


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=3, batch_first=True, num_layers=2): # 여기서 만들어야 할 모든 구조 정의하기
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=3, num_layers=num_layers, batch_first=batch_first)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

        
    def forward(self, x): # 여기서 나열하기
        x = x.transpose(2, 1)
        output, hidden = self.lstm(x)
        output = output[:, -1, :]
        final_output = self.fc(output)
        final_output = final_output.reshape(-1)
        
        return final_output

##############################################################
# 학습 함수 만들기

device = torch.device("cuda" if torch.cuda.is_available else "cpu") # gpu 또는 cpu 사용
# model = SimpleRNN().to(device)
model = SimpleLSTM().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# ReduceLROnPlateau 스케줄러 정의: 성능이 개선되지 않으면 학습률을 0.1배로 감소
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

def train(model, data, target, optimizer, criterion, device): # 여기서 target은 정답값
    
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    accuracy = 0
    
    
    data = data.to(device)
    target = target.to(device)
    
    model.train() # 학습모드. 테스트 모드와 구분됨
    optimizer.zero_grad() # 옵티마이저 초기화
    output = model(data) # 모델에 데이터 집어넣음. fc 레이어의 출력은 숫자 하나. 그리고 output은 배치사이즈 만큼의 길이의 리스트로 출력된다. 
    sigmoid_output = torch.sigmoid(output)
    predicted = (sigmoid_output > 0.5).float()
    correct += (predicted == target).sum().item()
    total += target.size(0)
    
    
    loss = criterion(output, target) # 손실 구하기. 이후 과정은 역전파?
    loss.backward() # 오류 역전파
    optimizer.step() # 가중치 업데이트
    
    total_loss += loss.item() # 여기서 loss는 하나의 텐서이다
    avg_loss = total_loss
    
    accuracy = correct/total

    
    return avg_loss, accuracy # train 함수는 학습의 결과로 loss를 반환한다.

def validate(model, data, target, criterion, device):
    
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    accuracy = 0
    
    data = data.to(device)
    target = target.to(device)
    
    output = model(data)
    sigmoid_output = torch.sigmoid(output)
    predircted = (sigmoid_output > 0.5).float()
    correct += (predircted == target).sum().item()
    total += target.size(0)
    
    loss = criterion(output, target)
    total_loss += loss.item()
    avg_loss = total_loss
    accuracy = correct/total
    
    return avg_loss, accuracy # 한 배치사이즈에 대해서 출력

def train_and_validate(model, data, criterion, optimizer, device): # data : Dataset 객체
    
    total_train_loss = 0
    total_train_accuracy = 0
    total_val_loss = 0
    total_val_accuracy = 0    
    
    train_size = int(0.8*len(data))
    val_size = len(data) - train_size
    train_dataset, validation_dataset = random_split(data, [train_size, val_size])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=validation_dataset, batch_size=32, shuffle=False)
    
    # 여기서 train, validate으로 return 되는 loss, acc 값은 한 배치사이즈에 대한 값(스칼라)이다. 따라서 이후에 누적해주고 더해줘야 된다?
    for train_data, train_label in train_loader:
        train_loss, train_accuracy = train(model=model, data=train_data, target=train_label, criterion=criterion, optimizer=optimizer, device=device)
        total_train_loss += train_loss
        total_train_accuracy += train_accuracy
    avg_train_loss = total_train_loss/len(train_loader)
    avg_train_accuracy = total_train_accuracy/len(train_loader)
        
    for val_data, val_label in val_loader:
        val_loss, val_accuracy = validate(model=model, data=val_data, target=val_label, criterion=criterion, device=device)
        total_val_loss += val_loss
        total_val_accuracy += val_accuracy
    avg_val_loss = total_val_loss/len(val_loader)
    avg_val_accuracy = total_val_accuracy/len(val_loader)    
            
    return avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy
        
    
    
    
    
    





# 여기서부터 학습    
real_audio_path = "../AudioDeepFakeDetection/real_temp/wav"
fake_audio_path = "../AudioDeepFakeDetection/fake_temp/wav"
    
mfcc_frames_all_tensor_real, mfcc_frames_all_tensor_fake = preprocess(real_audio_path , fake_audio_path)

print(f'shape of real data : {mfcc_frames_all_tensor_real.shape}')
print(f'shape of fake data : {mfcc_frames_all_tensor_fake.shape}')

audio_dataset = AudioDataset(mfcc_frames_all_tensor_real, mfcc_frames_all_tensor_fake)
# 이렇게 생성된 audio_dataset은 AudioDataset 객체. 이 객체 안에는 여러 변수와 메소드가 있다. 

print(f'length of data : {audio_dataset.label.shape}')

epochs = 10

for epoch in range(epochs):
    print(f'training.. epech : {epoch}') 
    
    total_train_loss, total_train_accuracy, total_val_loss, total_val_accuracy = train_and_validate(model=model, data=audio_dataset, criterion=criterion, optimizer=optimizer, device=device)
    
    avg_train_loss = total_train_loss
    avg_train_accuracy = total_train_accuracy
    avg_val_loss = total_val_loss
    avg_val_accuracy = total_val_accuracy
    
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Training Acc: {avg_train_loss:.4f}, Validation Acc: {avg_val_accuracy:.4f}")
    
