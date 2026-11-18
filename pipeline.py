"""
pipeline.py  –  Multimodal Deepfake Detection
Audio (RawNet2), image + video (EfficientNet-B4 on FACE CROPS), score fusion.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Tunables (override with environment variables if needed)
IMG_SIZE    = int(os.environ.get('IMG_SIZE', 224))
FACE_MARGIN = float(os.environ.get('FACE_MARGIN', 0.2))
USE_FACE_CROP = os.environ.get('USE_FACE_CROP', '1') == '1'


# ─────────────────────────────────────────────────────────────
# ARCHITECTURE DEFINITIONS
# ─────────────────────────────────────────────────────────────

class SincConv(nn.Module):
    """Mel-scale sinc filterbank for RawNet2."""

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size, in_channels=1,
                 sample_rate=16000, stride=1, padding=0, dilation=1):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        filbands = np.linspace(np.min(fmel), np.max(fmel), self.out_channels + 1)
        self.mel = self.to_hz(filbands)
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                   (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp.numpy() / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * self.hsupp.numpy() / self.sample_rate)
            self.band_pass[i, :] = (Tensor(np.hamming(self.kernel_size)) *
                                    Tensor(hHigh - hLow))
        filters = self.band_pass.to(self.device).view(
            self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm1d(nb_filts[0])
        self.lrelu = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], 3, padding=1)
        self.bn2 = nn.BatchNorm1d(nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], 3, padding=1)
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv1d(nb_filts[0], nb_filts[1], 1)
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        out = x if self.first else self.lrelu(self.bn1(x))
        out = self.lrelu(self.bn2(self.conv1(out)))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        return self.mp(out + identity)


class RawNet2(nn.Module):
    """End-to-end audio anti-spoofing (Tak et al., ICASSP 2021)."""

    def __init__(self, d_args, device):
        super().__init__()
        self.device = device
        self.Sinc_conv = SincConv(device, d_args['filts'][0],
                                  d_args['first_conv'], d_args['in_channels'])
        self.first_bn = nn.BatchNorm1d(d_args['filts'][0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(Residual_block(d_args['filts'][1], first=True))
        self.block1 = nn.Sequential(Residual_block(d_args['filts'][1]))
        self.block2 = nn.Sequential(Residual_block(d_args['filts'][2]))
        d_args['filts'][2][0] = d_args['filts'][2][1]
        self.block3 = nn.Sequential(Residual_block(d_args['filts'][2]))
        self.block4 = nn.Sequential(Residual_block(d_args['filts'][2]))
        self.block5 = nn.Sequential(Residual_block(d_args['filts'][2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        def att(f):
            return nn.Sequential(nn.Linear(f, f))

        self.fc_attention0 = att(d_args['filts'][1][-1])
        self.fc_attention1 = att(d_args['filts'][1][-1])
        self.fc_attention2 = att(d_args['filts'][2][-1])
        self.fc_attention3 = att(d_args['filts'][2][-1])
        self.fc_attention4 = att(d_args['filts'][2][-1])
        self.fc_attention5 = att(d_args['filts'][2][-1])
        self.bn_before_gru = nn.BatchNorm1d(d_args['filts'][2][-1])
        self.gru = nn.GRU(d_args['filts'][2][-1], d_args['gru_node'],
                          d_args['nb_gru_layer'], batch_first=True)
        self.fc1_gru = nn.Linear(d_args['gru_node'], d_args['nb_fc_node'])
        self.fc2_gru = nn.Linear(d_args['nb_fc_node'], d_args['nb_classes'],
                                 bias=True)
        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def _att(self, block, fc_att, x):
        xb = block(x)
        y = self.sig(fc_att(self.avgpool(xb).view(xb.size(0), -1)))
        y = y.view(y.size(0), y.size(1), -1)
        return xb * y + y

    def forward(self, x):
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = self.Sinc_conv(x.view(nb_samp, 1, len_seq))
        x = self.selu(self.first_bn(F.max_pool1d(torch.abs(x), 3)))
        x = self._att(self.block0, self.fc_attention0, x)
        x = self._att(self.block1, self.fc_attention1, x)
        x = self._att(self.block2, self.fc_attention2, x)
        x = self._att(self.block3, self.fc_attention3, x)
        x = self._att(self.block4, self.fc_attention4, x)
        x = self._att(self.block5, self.fc_attention5, x)
        x = self.selu(self.bn_before_gru(x))
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        return self.logsoftmax(self.fc2_gru(self.fc1_gru(x[:, -1, :])))


class DeepfakeDetector(nn.Module):
    """EfficientNet-B4 backbone with two-stage dropout head."""

    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_b4',
                                          pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


class ScoreFusionMLP(nn.Module):
    """Learned score-level fusion: 4-dim posterior vector -> joint P(fake)."""

    def __init__(self, input_dim=4, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# MODEL SINGLETONS (lazy load on first use)
# ─────────────────────────────────────────────────────────────

_audio_model  = None
_image_model  = None
_video_model  = None
_fusion_model = None

# (real_idx, fake_idx) of the image / video classifiers' output
_image_idx = (1, 0)
_video_idx = (1, 0)


def _label_idx(ck, env_name):
    """Read (real_idx, fake_idx) from checkpoint class_to_idx, env override, or default."""
    override = os.environ.get(env_name)
    if override in ('0', '1'):
        r = int(override)
        return r, 1 - r
    c2i = ck.get('class_to_idx') if isinstance(ck, dict) else None
    if c2i and 'real' in c2i and 'fake' in c2i:
        return int(c2i['real']), int(c2i['fake'])
    return 1, 0


def _load_audio_model():
    global _audio_model
    if _audio_model is not None:
        return _audio_model
    path = os.path.join(MODEL_DIR, 'rawnet2_best.pth')
    if not os.path.exists(path):
        print(f'[Audio] Model not found: {path}')
        return None
    try:
        d_args = {
            'nb_classes': 2, 'nb_fc_node': 1024, 'gru_node': 1024,
            'nb_gru_layer': 3, 'nb_samp': 64600, 'first_conv': 1024,
            'in_channels': 1, 'filts': [20, [20, 20], [20, 128], [128, 128]],
        }
        m = RawNet2(d_args, device).to(device)
        ck = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ck['model_state_dict'])
        m.eval()
        _audio_model = m
        print('[Audio] Model loaded OK')
        return m
    except Exception as e:
        print(f'[Audio] Load error: {e}')
        return None


def _load_image_model():
    global _image_model, _image_idx
    if _image_model is not None:
        return _image_model
    path = os.path.join(MODEL_DIR, 'efficientnet_b4_best.pth')
    if not os.path.exists(path):
        print(f'[Image] Model not found: {path}')
        return None
    try:
        m = DeepfakeDetector().to(device)
        ck = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ck['model_state_dict'])
        m.eval()
        _image_idx = _label_idx(ck, 'IMAGE_REAL_IDX')
        _image_model = m
        print(f'[Image] Model loaded OK | ckpt keys={list(ck.keys())} '
              f'| real_idx={_image_idx[0]} fake_idx={_image_idx[1]}')
        return m
    except Exception as e:
        print(f'[Image] Load error: {e}')
        return None


def _load_video_model():
    global _video_model, _video_idx
    if _video_model is not None:
        return _video_model
    path = os.path.join(MODEL_DIR, 'video_detector_best.pth')
    if not os.path.exists(path):
        print(f'[Video] Model not found: {path}')
        return None
    try:
        m = DeepfakeDetector().to(device)
        ck = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ck['model_state_dict'])
        m.eval()
        _video_idx = _label_idx(ck, 'VIDEO_REAL_IDX')
        _video_model = m
        print(f'[Video] Model loaded OK | ckpt keys={list(ck.keys())} '
              f'| real_idx={_video_idx[0]} fake_idx={_video_idx[1]}')
        return m
    except Exception as e:
        print(f'[Video] Load error: {e}')
        return None


def _load_fusion_model():
    global _fusion_model
    if _fusion_model is not None:
        return _fusion_model
    path = os.path.join(MODEL_DIR, 'fusion_mlp.pth')
    if not os.path.exists(path):
        print('[Fusion] fusion_mlp.pth not found – using equal-weight fallback.')
        return None
    try:
        m = ScoreFusionMLP(input_dim=4, hidden_dim=16).to(device)
        ck = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ck['model_state_dict'])
        m.eval()
        _fusion_model = m
        print('[Fusion] Model loaded OK')
        return m
    except Exception as e:
        print(f'[Fusion] Load error: {e}')
        return None


# ─────────────────────────────────────────────────────────────
# SHARED PREPROCESSING
# ─────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
MAX_LEN     = 64600

import torchvision.transforms as T
_img_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_face_cascade = None


def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _face_cascade


def _crop_face(rgb):
    """rgb: HxWx3 uint8 numpy array. Returns the largest face crop (with margin) or None."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _get_cascade().detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
    if len(faces) == 0:
        # try again, more permissive
        faces = _get_cascade().detectMultiScale(gray, 1.05, 3, minSize=(32, 32))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(FACE_MARGIN * max(w, h))
    H, W = rgb.shape[:2]
    x0, y0 = max(x - m, 0), max(y - m, 0)
    x1, y1 = min(x + w + m, W), min(y + h + m, H)
    return rgb[y0:y1, x0:x1]


def _prep(rgb):
    """numpy RGB -> (tensor, face_found). Crops the face when enabled and found."""
    from PIL import Image as PILImage
    face_found = False
    if USE_FACE_CROP:
        crop = _crop_face(rgb)
        if crop is not None and crop.size > 0:
            rgb = crop
            face_found = True
    return _img_transform(PILImage.fromarray(rgb).convert('RGB')), face_found


# ─────────────────────────────────────────────────────────────
# INTERNAL SCORING HELPERS
# ─────────────────────────────────────────────────────────────

def _score_image(image):
    """Return (real_prob, fake_prob, face_found) for a PIL/numpy image, or None."""
    model = _load_image_model()
    if model is None or image is None:
        return None
    try:
        from PIL import Image as PILImage
        if not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(image)
        rgb = np.array(image.convert('RGB'))
        tensor, found = _prep(rgb)
        with torch.no_grad():
            probs = torch.softmax(model(tensor.unsqueeze(0).to(device)), dim=1)[0]
        r, f = _image_idx
        return probs[r].item(), probs[f].item(), found
    except Exception as e:
        print(f'[Image] Scoring error: {e}')
        return None


def _score_audio(audio_path):
    """Return (real_prob, fake_prob) for an audio file path, or None."""
    model = _load_audio_model()
    if model is None or not audio_path or not os.path.exists(audio_path):
        return None
    try:
        import soundfile as sf
        import librosa

        ext = os.path.splitext(audio_path)[1].lower()
        if ext in ('.m4a', '.mp3', '.mpeg', '.mpg', '.aac', '.ogg', '.wma'):
            from pydub import AudioSegment
            seg = AudioSegment.from_file(audio_path)
            tmp = audio_path.rsplit('.', 1)[0] + '_converted.wav'
            seg.export(tmp, format='wav')
            audio_path = tmp

        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = wav[:, 0]
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        wav = wav.astype(np.float32)
        if len(wav) >= MAX_LEN:
            wav = wav[:MAX_LEN]
        else:
            wav = np.pad(wav, (0, MAX_LEN - len(wav)))

        tensor = torch.FloatTensor(wav).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.exp(model(tensor))[0]
        return probs[1].item(), probs[0].item()   # real, fake
    except Exception as e:
        print(f'[Audio] Scoring error: {e}')
        return None


def _score_video(video_path, n_samples=30):
    """
    Return (real_prob, fake_prob, n_frames, n_faces) or None.
    Samples n_samples frames, crops faces, and averages posteriors over
    frames where a face was found (falls back to all frames if none found).
    """
    model = _load_video_model()
    if model is None or video_path is None:
        return None
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        wanted = set(np.linspace(0, total - 1, n_samples, dtype=int).tolist())

        face_tensors, all_tensors = [], []
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor, found = _prep(rgb)
                    all_tensors.append(tensor)
                    if found:
                        face_tensors.append(tensor)
            idx += 1
            if idx > max(wanted):
                break
        cap.release()

        use = face_tensors if face_tensors else all_tensors
        if not use:
            return None

        batch = torch.stack(use).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        r, f = _video_idx
        return (float(probs[:, r].mean()), float(probs[:, f].mean()),
                len(all_tensors), len(face_tensors))
    except Exception as e:
        print(f'[Video] Scoring error: {e}')
        return None


# ─────────────────────────────────────────────────────────────
# PUBLIC PREDICTION FUNCTIONS (called by Gradio)
# ─────────────────────────────────────────────────────────────

def predict_image(image):
    if image is None:
        return "No image provided."
    result = _score_image(image)
    if result is None:
        return ("Model unavailable. Place efficientnet_b4_best.pth in the "
                "models/ folder.")
    real_prob, fake_prob, found = result
    verdict = "✅ REAL" if real_prob > 0.5 else "🚨 FAKE (Deepfake)"
    note = "" if found or not USE_FACE_CROP else \
        "\nNote            : no face detected, whole image used (less reliable)"
    return (f"{verdict}\n\n"
            f"Real probability : {real_prob * 100:.1f}%\n"
            f"Fake probability : {fake_prob * 100:.1f}%\n"
            f"Confidence       : {max(real_prob, fake_prob) * 100:.1f}%{note}")


def predict_audio(audio_path):
    if audio_path is None:
        return "No audio provided."
    result = _score_audio(audio_path)
    if result is None:
        return ("Model unavailable. Place rawnet2_best.pth in the "
                "models/ folder.")
    real_prob, fake_prob = result
    verdict = "✅ REAL (Genuine Voice)" if real_prob > 0.5 else "🚨 FAKE (Synthetic Voice)"
    return (f"{verdict}\n\n"
            f"Real probability : {real_prob * 100:.1f}%\n"
            f"Fake probability : {fake_prob * 100:.1f}%\n"
            f"Confidence       : {max(real_prob, fake_prob) * 100:.1f}%")


def predict_video(video_path):
    if video_path is None:
        return "No video provided."
    result = _score_video(video_path)
    if result is None:
        return ("Model unavailable or no frames extracted. Place "
                "video_detector_best.pth in the same folder as pipeline.py.")
    real_prob, fake_prob, n_frames, n_faces = result
    verdict = "✅ REAL" if real_prob > 0.5 else "🚨 FAKE (Deepfake Video)"
    return (f"{verdict}\n\n"
            f"Real probability : {real_prob * 100:.1f}%\n"
            f"Fake probability : {fake_prob * 100:.1f}%\n"
            f"Frames analysed  : {n_frames} ({n_faces} with a detected face)\n"
            f"Confidence       : {max(real_prob, fake_prob) * 100:.1f}%")


def predict_fusion(audio_file, video_path):
    """Joint audio-video classification via the score-fusion MLP."""
    audio_path = audio_file.name if audio_file else None

    if audio_path is None and video_path is None:
        return "Please provide both an audio file and a video clip."

    audio_result = _score_audio(audio_path) if audio_path else None
    video_result = _score_video(video_path) if video_path else None

    warnings = []
    if audio_path and audio_result is None:
        warnings.append("⚠️ Audio model unavailable – using video branch only.")
    if video_path and video_result is None:
        warnings.append("⚠️ Video model unavailable – using audio branch only.")

    if audio_result is None and video_result is None:
        return "Both branch models are unavailable. Check model files."

    if audio_result is None:
        real_prob, fake_prob = video_result[0], video_result[1]
        method = "video branch only (audio unavailable)"
    elif video_result is None:
        real_prob, fake_prob = audio_result
        method = "audio branch only (video unavailable)"
    else:
        a_real, a_fake = audio_result
        v_real, v_fake = video_result[0], video_result[1]

        fusion_model = _load_fusion_model()

        if fusion_model is not None:
            f = torch.tensor([[a_real, a_fake, v_real, v_fake]],
                             dtype=torch.float32).to(device)
            with torch.no_grad():
                joint_fake = fusion_model(f).item()
            fake_prob = joint_fake
            real_prob = 1.0 - joint_fake
            method = (f"learned MLP fusion | audio={a_fake*100:.1f}% fake, "
                      f"video={v_fake*100:.1f}% fake")
        else:
            fake_prob = (a_fake + v_fake) / 2.0
            real_prob = 1.0 - fake_prob
            method = (f"equal-weight average (fusion MLP not found) | "
                      f"audio={a_fake*100:.1f}% fake, video={v_fake*100:.1f}% fake")

    verdict = "✅ REAL" if real_prob > 0.5 else "🚨 FAKE (Deepfake)"
    header  = "\n".join(warnings) + ("\n\n" if warnings else "")
    return (f"{header}{verdict}\n\n"
            f"Joint fake probability : {fake_prob * 100:.1f}%\n"
            f"Joint real probability : {real_prob * 100:.1f}%\n"
            f"Confidence             : {max(real_prob, fake_prob) * 100:.1f}%\n"
            f"Method                 : {method}")