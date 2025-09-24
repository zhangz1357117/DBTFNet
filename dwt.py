import numpy as np
import pywt
import torch
from scipy.stats import entropy


def compute_wavelet_features(coeffs):
    """
    计算 8 个统计特征：
    1. 均值
    2. 中位数
    3. 标准差
    4. 方差
    5. 均方根值（RMS）
    6. 过零率（ZCR）
    7. 平均过零率（AZCR）
    8. 熵（Entropy）
    """
    mean_val = np.mean(coeffs)
    median_val = np.median(coeffs)
    std_val = np.std(coeffs)
    var_val = np.var(coeffs)
    rms_val = np.sqrt(np.mean(coeffs ** 2))  # 均方根

    # 过零率 (ZCR)
    zero_crossings = np.sum((coeffs[:-1] * coeffs[1:]) < 0) / len(coeffs)

    # 平均过零率 (AZCR)（设定一个小阈值，计算信号变化率）
    threshold = np.std(coeffs) * 0.1
    avg_zero_crossing = np.sum(np.abs(coeffs[:-1] - coeffs[1:]) > threshold) / len(coeffs)

    # 熵 (Entropy)
    coeffs_pdf = np.abs(coeffs) / np.sum(np.abs(coeffs))  # 归一化为概率分布
    coeffs_pdf = coeffs_pdf[coeffs_pdf > 0]  # 避免 log(0)
    entropy_val = entropy(coeffs_pdf)

    return np.array([mean_val, median_val, std_val, var_val, rms_val, zero_crossings, avg_zero_crossing, entropy_val])


class WaveletFeatureExtractor:
    def __init__(self, wavelet='db4', level=5):
        self.wavelet = wavelet
        self.level = level

    def wavelet_transform(self, ecg_signal):
        """
        计算 ECG 信号的小波变换统计特征。
        :param ecg_signal: 输入形状 (batch, channels, seq_len)
        :return: 统计特征张量 (batch, channels, level+1, 8)
        """
        batch, channels, seq_len = ecg_signal.shape
        transformed = []

        for i in range(batch):
            channel_coeffs = []
            for ch in range(channels):
                coeffs = pywt.wavedec(ecg_signal[i, ch].cpu().numpy(), self.wavelet, level=self.level)
                coeffs_stats = np.array([compute_wavelet_features(c) for c in coeffs]).reshape(-1)  # (48,)
                channel_coeffs.append(coeffs_stats)  # list of (level+1, 8)

            channel_coeffs = np.stack(channel_coeffs, axis=0)  # (channels, level+1, 8)
            transformed.append(channel_coeffs)

        transformed = np.stack(transformed, axis=0)  # (batch, channels, level+1, 8)
        return torch.tensor(transformed, dtype=torch.float).to(ecg_signal.device)