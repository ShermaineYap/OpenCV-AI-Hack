"""Concatenate brag + slides + live-demo segments, place narration, add music bed, encode final.mp4."""
import json, subprocess, os, numpy as np, wave
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
ORDER = ["brag", "team", "problem", "lag", "idea", "demo1", "live", "demo3",
         "vision", "predict", "aws", "results", "limits", "responsible", "close"]
LEAD = 0.35
def dur(p): return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p]))
files = {s: ("../../brag-output/brag.mp4" if s == "brag" else f"seg_{s}.mp4") for s in ORDER}
starts, t = {}, 0.0
for s in ORDER:
    starts[s] = t; t += dur(files[s])
TOTAL = t
print("total", round(TOTAL, 2), "s")
# video: re-encode every piece to identical params, then concat demuxer
with open("concat.txt", "w") as f:
    for s in ORDER:
        norm = f"n_{s}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", files[s], "-an", "-vf", "fps=30,scale=1920:1080,format=yuv420p,setsar=1",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", norm], check=True)
        f.write(f"file '{norm}'\n")
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt", "-c", "copy", "video_only.mp4"], check=True)
# audio
SR = 48000; N = int(TOTAL * SR) + SR
mix = np.zeros((N, 2))
def read(p):
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", p, "-f", "s16le", "-ac", "2", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, "<i2").reshape(-1, 2).astype(np.float64) / 32768
def place(sig, t0, g=1.0):
    i = int(t0 * SR); j = min(N, i + len(sig)); mix[i:j] += sig[:j - i] * g
place(read(files["brag"]), 0.0, 1.0)                                   # brag keeps its own soundtrack
subprocess.run(["python3", "bed.py", str(TOTAL - 20.0 + 0.5)], check=True)
place(read("bed.wav"), 20.0, 0.16)                                      # quiet bed under narration
for s in ORDER[1:]:
    place(read(f"n_{s}.wav"), starts[s] + LEAD, 0.95)
mix = np.tanh(mix * 1.1) / np.tanh(1.1)
mix *= 0.89 / np.abs(mix).max()
with wave.open("final_audio.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix[:int(TOTAL * SR)] * 32767).astype("<i2").tobytes())
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "video_only.mp4", "-i", "final_audio.wav",
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-shortest", "-movflags", "+faststart", "../BoilGuard_OpenCV26_demo.mp4"], check=True)
json.dump({"starts": starts, "total": TOTAL}, open("timeline.json", "w"), indent=1)
print("written ../BoilGuard_OpenCV26_demo.mp4")
