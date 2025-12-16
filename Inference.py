from preprocessing import bandpass_filter, hilbert_signal, normalize, windowing
from feature_extraction import frequency_domain_features, time_frequency_domain_features, time_domain_features
import pandas as pd
import numpy as np
import pickle


# vibration_data= pd.read_csv("Vibration data/0Nm_Normal.csv") 
# data = vibration_data.iloc[:, 0].values # Lấy cột đầu tiên (index = 0)


fs= 25600  #tần số lấy mẫu
lowcut = 10  #tần số cắt thấp
highcut = 4000 #tần số cắt cao
window_size = 2**15  #số mẫu mỗi đoạn
step_size = int((2**15)/2)  #bước nhảy
files = {"Vibration_Data/0Nm_BPFI_03.csv": "BPFI",
        "Vibration_Data/2Nm_BPFI_03.csv": "BPFI",
        "Vibration_Data/4Nm_BPFI_03.csv": "BPFI",
        "Vibration_Data/0Nm_BPFI_10.csv": "BPFI",
        "Vibration_Data/2Nm_BPFI_10.csv": "BPFI",
        "Vibration_Data/4Nm_BPFI_10.csv": "BPFI",
        "Vibration_Data/0Nm_BPFO_03.csv": "BPFO",
        "Vibration_Data/2Nm_BPFO_03.csv": "BPFO",
        "Vibration_Data/4Nm_BPFO_03.csv": "BPFO",
        "Vibration_Data/0Nm_BPFO_10.csv": "BPFO",
        "Vibration_Data/2Nm_BPFO_10.csv": "BPFO",
        "Vibration_Data/4Nm_BPFO_10.csv": "BPFO",
        "Vibration_Data/0Nm_Misalign_01.csv": "Misalign",
        "Vibration_Data/2Nm_Misalign_01.csv": "Misalign",
        "Vibration_Data/4Nm_Misalign_01.csv": "Misalign",
        "Vibration_Data/0Nm_Misalign_03.csv": "Misalign",
        "Vibration_Data/2Nm_Misalign_03.csv": "Misalign",
        "Vibration_Data/4Nm_Misalign_03.csv": "Misalign",
        "Vibration_Data/0Nm_Unbalance_0583mg.csv": "Unbalance",
        "Vibration_Data/2Nm_Unbalance_0583mg.csv": "Unbalance",
        "Vibration_Data/4Nm_Unbalance_0583mg.csv": "Unbalance",
        "Vibration_Data/0Nm_Unbalance_1169mg.csv": "Unbalance",
        "Vibration_Data/2Nm_Unbalance_1169mg.csv": "Unbalance",
        "Vibration_Data/4Nm_Unbalance_1169mg.csv": "Unbalance",
        "Vibration_Data/0Nm_Normal.csv": "Normal",
        "Vibration_Data/2Nm_Normal.csv": "Normal",
        "Vibration_Data/4Nm_Normal.csv": "Normal"
        }
LABEL_MAP = {
    0:"Normal",
    1:"BPFI",
    2:"BPFO",
    3:"Misalign",
    4:"Unbalance"
}

data=pd.read_csv("Vibration_Data/0Nm_BPFI_03.csv").iloc[:, 0].values

#Tiền xử lý
filtered_data = bandpass_filter(data, lowcut, highcut, fs)
envelope_data = hilbert_signal(filtered_data)['envelope']
normalized_data = normalize(envelope_data)
normalized_data = normalized_data - np.mean(normalized_data)
windows = windowing(normalized_data, window_size, step_size)

#Tính đặc trưng 
# time_feats = [time_domain_features(window) for window in windows]
freq_feats = [frequency_domain_features(window,fs) for window in windows]
time_freq_feats = [time_frequency_domain_features(window,fs,nperseg=256, noverlap=128) for window in windows]
    
# time_cols = ["Mean_time", "RMS_time", "MAX", "MIN", "PTP", "CREST_FACTOR"]
freq_cols = ["Mean_freq",
             "RMS_freq", 
             "Var", 
             "Skewness",
             "Kurtosis",
             "Central", 
             "STD", 
             "Spectral_centroid",
             "Spectral_spread",
             "Spectral_entropy"]

time_freq_cols = ["Mean_energy",
                  "RMS_freq_TF",
                  "Var_energy",
                  "Skewness_energy",
                  "Kurtosis_energy",
                  "TF_centroid", 
                  "TF_spread", 
                  "Mean_time_energy",
                  "Spectral_flux", 
                  "TF_entropy"]

#Tạo DataFrame có tên cột
# df_time = pd.DataFrame(time_feats, columns=time_cols)
df_freq = pd.DataFrame(freq_feats, columns=freq_cols)
df_time_freq = pd.DataFrame(time_freq_feats, columns=time_freq_cols)

# Gộp lại
features_test = pd.concat([df_freq,df_time_freq], axis=1)  


with open("softmax_logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# ===== PREDICT =====
X_test = features_test.values              # (n_windows, n_features)
y_pred_window = model.predict(X_test)             # (n_windows,)
y_proba_window = model.predict_proba(X_test)       # (n_windows, n_classes)

# ===== MAP INDEX -> LABEL NAME =====
LABEL_MAP = {
    0: "Normal",
    1: "BPFI",
    2: "BPFO",
    3: "Misalign",
    4: "Unbalance"
}

pred_labels_window = [LABEL_MAP[i] for i in y_pred_window]
mean_proba = y_proba_window.mean(axis=0)   # (n_classes,)

pred_class_idx = np.argmax(mean_proba)
final_label = LABEL_MAP[pred_class_idx]
print("Mean probability over all windows:")
for i, p in enumerate(mean_proba):
    print(f"{LABEL_MAP[i]:10s}: {p:.4f}")

print("\nFinal predicted label (mean softmax):", final_label)

