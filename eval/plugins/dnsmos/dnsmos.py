import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort

class DNSMOSModel:
    def __init__(self, model_path, device="cpu"):
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        self.sr = 16000
        self.segment_sec = 9
        self.stride_sec = 4.5   # 50% overlap（工程推荐）

    # ----------------------
    # 音频加载
    # ----------------------
    def load_audio(self, path):
        audio, sr = sf.read(path)

        # 转单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 重采样
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        # 归一化
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        return audio

    # ----------------------
    # 分段（核心工程策略）
    # ----------------------
    def segment_audio(self, audio):
        len_samples = int(9.01 * self.sr)
        
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
            
        segments = []
        hop_len_samples = self.sr  # 1s hop
        num_hops = int(np.floor(len(audio)/self.sr) - 9.01)+1
        if num_hops < 1:
            num_hops = 1
            
        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+9.01)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue
            segments.append(audio_seg)
            
        if not segments:
            segments.append(audio[:len_samples])
            
        return segments

    # ----------------------
    # 特征提取
    # ----------------------
    def extract_feature(self, audio):
        pass

    # ----------------------
    # 单段推理
    # ----------------------
    def infer_segment(self, segment):
        feat = np.array(segment).astype('float32')[np.newaxis, :]
        out = self.session.run(None, {self.input_name: feat})[0][0]
        return out  # [sig, bak, ovr]

    # ----------------------
    # 主推理接口
    # ----------------------
    def infer(self, audio_path):
        audio = self.load_audio(audio_path)
        segments = self.segment_audio(audio)

        scores = []
        for seg in segments:
            s = self.infer_segment(seg)
            scores.append(s)

        scores = np.array(scores)

        result = {
            "dnsmos_sig": float(np.mean(scores[:, 0])),
            "dnsmos_bak": float(np.mean(scores[:, 1])),
            "dnsmos_ovr": float(np.mean(scores[:, 2])),
            "dnsmos_std": float(np.std(scores[:, 2])),  # 稳定性
            "num_segments": len(scores)
        }

        return result