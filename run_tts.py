import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 1. Load model onto Mac M1 (MPS)
print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    "g-group-ai-lab/gwen-tts-0.6B",
    device_map="mps",
    dtype=torch.float32,
)

# 2. Recommended generation parameters
generation_config = dict(
    temperature=0.3,
    top_k=20,
    top_p=0.9,
    max_new_tokens=4096,
    repetition_penalty=2.0,
    subtalker_do_sample=True,
    subtalker_temperature=0.1,
    subtalker_top_k=20,
    subtalker_top_p=1.0,
)

# 3. Define absolute paths to prevent "Format not recognised" errors
# (Update this if your project folder is located somewhere else)
ref_audio = "data/ref_audio/ai_vy.wav"
ref_text = "việt nam đang kiêu hãnh bước vào kỷ nguyên vươn mình rực rỡ với khát vọng mãnh liệt, trí tuệ đổi mới và tinh thần đoàn kết đất nước, tự tin bứt phá, kiến tạo một tương lai thịnh vượng và vươn tầm quốc tế."

text_to_speak = "Thưa quý vị, sáng nay tại thành phố Hồ Chí Minh, Diễn đàn Kinh tế số đã chính thức khai mạc."

# Check if the reference file exists before running
if not os.path.exists(ref_audio):
    raise FileNotFoundError(f"Could not find reference audio at: {ref_audio}")

print("Generating speech...")
wavs, sr = model.generate_voice_clone(
    text=text_to_speak,
    language="Vietnamese",
    ref_audio=ref_audio,
    ref_text=ref_text,
    **generation_config,
)

# 4. Save output
output_file = "output/tts_output.wav"
sf.write(output_file, wavs[0], sr)
print(f"Successfully saved audio to {output_file}")