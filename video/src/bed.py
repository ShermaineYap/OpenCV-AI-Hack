import numpy as np, scipy.signal as ss, wave, sys
SR = 48000; DUR = float(sys.argv[1]); N = int(SR * DUR)
rng = np.random.default_rng(5); L = np.zeros(N); R = np.zeros(N)
midi = lambda m: 440 * 2 ** ((m - 69) / 12)
def lp(x, fc): b, a = ss.butter(2, fc / (SR / 2)); return ss.lfilter(b, a, x)
def saw(f, n): ph = (np.arange(n) * f / SR) % 1; return 2 * ph - 1
chords = [[57, 60, 64], [53, 57, 60], [48, 55, 60, 64], [55, 59, 62]]
BAR = 3.0
for b in range(int(DUR / BAR) + 1):
    ch = chords[b % 4]; n = int((BAR + 1.5) * SR); sig = np.zeros(n)
    for m in ch:
        for det in (-0.07, 0.07): sig += saw(midi(m + det), n)
    k = np.arange(n) / SR
    e = np.minimum(1, k / 0.8) * np.minimum(1, np.maximum(0, (BAR + 1.5 - k) / 1.5))
    sig = lp(sig, 900) / 8 * e
    i = int(b * BAR * SR); j = min(N, i + n)
    L[i:j] += sig[:j - i] * 0.9; R[i:j] += np.roll(sig, 300)[:j - i] * 0.9
    for q in range(6):                        # soft pluck arpeggio, 8ths at 120 bpm feel
        t0 = b * BAR + q * 0.5
        m = (ch + [ch[0] + 12])[q % 4] + 12; nn = int(0.6 * SR); kk = np.arange(nn) / SR
        pl = np.sin(2 * np.pi * midi(m) * kk) * np.exp(-kk * 7) * 0.12
        i = int(t0 * SR); j = min(N, i + nn)
        if i < N: (L if q % 2 else R)[i:j] += pl[:j - i]
fo = int(2.5 * SR); fi = int(1.5 * SR)
for c in (L, R): c[:fi] *= np.linspace(0, 1, fi); c[-fo:] *= np.linspace(1, 0, fo)
st = np.stack([L, R], 1); st *= 0.5 / np.abs(st).max()
with wave.open("bed.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st * 32767).astype("<i2").tobytes())
print("bed ok")
