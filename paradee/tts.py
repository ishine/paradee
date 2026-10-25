"""Text -> phonemes (misaki, Kokoro's own G2P) -> token ids -> one ONNX graph -> 24 kHz audio."""
import json, re
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPOS = {"1.2": "sahilmahendrakar/Paradee-8M-v1.2", "1.1": "sahilmahendrakar/Paradee-8M-v1.1", "1.0": "sahilmahendrakar/Paradee-8M-v1.0"}
# the voices of each version, in the order the 1.2 model numbers them. 1.0 and 1.1 have af_heart only.
VOICES = {"1.2": ["af_heart", "bf_isabella", "am_puck", "bm_fable"], "1.1": ["af_heart"], "1.0": ["af_heart"]}
SAMPLE_RATE = 24000
MAX_PHONEMES = 510   # the text side has 512 positions, two of which are the pad tokens at each end


class Paradee:
    """Load once, then call with text to get a float32 waveform at 24 kHz.

    version: "1.2" (default, four voices), "1.1" or "1.0" (af_heart only; 1.0 has the slower decoder).
    quantized: use the 8.5 MB int8 file (default) instead of the 35 MB fp32 one. They sound the same.
    model_path / config_path: local files instead of downloading from the Hugging Face Hub.
    threads: CPU threads for onnxruntime. One thread already runs about 60x faster than real time.
    """

    def __init__(self, version="1.2", quantized=True, model_path=None, config_path=None, threads=1):
        if version not in REPOS:
            raise ValueError(f"unknown version {version!r}; choose one of {', '.join(REPOS)}")
        repo, revision = REPOS[version], f"v{version}"
        self.version, self.voices = version, VOICES[version]
        name = "onnx/paradee_int8.onnx" if quantized else "onnx/paradee.onnx"
        model_path = model_path or hf_hub_download(repo, name, revision=revision)
        config_path = config_path or hf_hub_download(repo, "config.json", revision=revision)
        self.vocab = json.load(open(config_path, encoding="utf-8"))["vocab"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self._g2p = {}

    def g2p(self, british=False):
        """The same G2P setup as kokoro.KPipeline, which made the training data: lang_code "a" (American) for
        the af_/am_ voices, "b" (British) for the bf_/bm_ ones. Each is loaded the first time it is needed."""
        if british not in self._g2p:
            from misaki import en, espeak
            self._g2p[british] = en.G2P(trf=False, british=british, fallback=espeak.EspeakFallback(british=british), unk="")
        return self._g2p[british]

    def phonemize(self, text, voice="af_heart"):
        """British phonemes for the British voices (bf_isabella, bm_fable), American for the rest."""
        phonemes, _ = self.g2p(british=voice.startswith("b"))(text)
        return phonemes

    def check_voice(self, voice):
        if voice not in self.voices:
            raise ValueError(f"Paradee {self.version} has no voice {voice!r}; its voices are {', '.join(self.voices)}")

    def generate_from_phonemes(self, phonemes, speed=1.0, voice="af_heart"):
        self.check_voice(voice)
        ids = [0] + [self.vocab[c] for c in phonemes if c in self.vocab][:MAX_PHONEMES] + [0]
        feed = {"input_ids": np.array([ids], dtype=np.int64), "speed": np.array([speed], dtype=np.float32)}
        if len(self.voices) > 1:
            feed["voice"] = np.array([self.voices.index(voice)], dtype=np.int64)
        return self.session.run(None, feed)[0][0]

    def __call__(self, text, speed=1.0, voice="af_heart"):
        """speed > 1 speaks faster. voice is one of self.voices. Long text is read one sentence at a time."""
        self.check_voice(voice)
        parts = [self.generate_from_phonemes(ps, speed, voice) for ps in self._chunks(text, voice)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _chunks(self, text, voice="af_heart"):
        for sentence in re.split(r"(?<=[.!?…])\s+|\n+", text.strip()):
            if not sentence.strip():
                continue
            ps = self.phonemize(sentence, voice)
            while len(ps) > MAX_PHONEMES:   # a very long sentence: cut at the last space that fits
                cut = ps.rfind(" ", 0, MAX_PHONEMES)
                cut = cut if cut > 0 else MAX_PHONEMES
                yield ps[:cut]
                ps = ps[cut:].lstrip()
            if ps:
                yield ps
