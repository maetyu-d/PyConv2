import numpy as np
import struct
import json

# =========================
# WAV I/O
# =========================

def _read_wav_bytes(b: bytes):
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file")

    i = 12
    fmt = None
    data = None
    while i + 8 <= len(b):
        cid = b[i:i+4]
        csz = struct.unpack_from("<I", b, i+4)[0]
        cdat = b[i+8:i+8+csz]
        i += 8 + csz + (csz % 2)

        if cid == b"fmt ":
            audio_fmt, num_ch, sr, byte_rate, block_align, bps = struct.unpack_from("<HHIIHH", cdat, 0)
            fmt = (audio_fmt, num_ch, sr, bps)
        elif cid == b"data":
            data = cdat

    if fmt is None or data is None:
        raise ValueError("Missing fmt or data chunk")

    audio_fmt, ch, sr, bps = fmt

    if audio_fmt == 1:  # PCM
        if bps == 16:
            x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        elif bps == 24:
            raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
            vals = (raw[:,0].astype(np.int32) |
                    (raw[:,1].astype(np.int32) << 8) |
                    (raw[:,2].astype(np.int32) << 16))
            vals = (vals << 8) >> 8
            x = vals.astype(np.float32) / (2**23)
        elif bps == 32:
            x = np.frombuffer(data, dtype="<i4").astype(np.float32) / (2**31)
        else:
            raise ValueError(f"Unsupported PCM bit depth: {bps}")
    elif audio_fmt == 3:  # float32
        if bps != 32:
            raise ValueError("Only float32 WAV supported for IEEE float")
        x = np.frombuffer(data, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV format code: {audio_fmt}")

    x = x.reshape(-1, ch) if ch > 1 else x.reshape(-1, 1)
    return sr, x

def _write_wav_bytes(sr: int, x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    n, ch = x.shape
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)

    data_bytes = pcm.tobytes()
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, ch, sr, sr * ch * 2, ch * 2, 16)
    data_chunk = struct.pack("<4sI", b"data", len(data_bytes)) + data_bytes
    file_size = 12 + len(fmt_chunk) + len(data_chunk)
    riff_size = file_size - 8
    header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    return header + fmt_chunk + data_chunk

# =========================
# Helpers
# =========================

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def _partition_ir(ir: np.ndarray, part_size: int):
    L = ir.shape[0]
    n_parts = int(np.ceil(L / part_size))
    parts = []
    for k in range(n_parts):
        s = k * part_size
        e = min((k + 1) * part_size, L)
        p = np.zeros((part_size,), dtype=np.float32)
        p[:e - s] = ir[s:e]
        parts.append(p)
    return parts

def _build_H_parts(ir_parts, fft_size):
    return [np.fft.rfft(p, n=fft_size).astype(np.complex64) for p in ir_parts]

def _norm_01(v, mode: str):
    v = np.asarray(v, dtype=np.float32)
    if v.size == 0:
        return v
    if mode == "none":
        return v
    if mode == "minmax":
        lo = float(np.min(v))
        hi = float(np.max(v))
        if hi <= lo + 1e-12:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)
    if mode == "percentile":
        lo = float(np.percentile(v, 5))
        hi = float(np.percentile(v, 95))
        if hi <= lo + 1e-12:
            return np.zeros_like(v)
        v2 = np.clip(v, lo, hi)
        return (v2 - lo) / (hi - lo)
    raise ValueError("Unknown norm mode: " + mode)

def _one_pole_smooth(x, sr, smooth_ms):
    x = np.asarray(x, dtype=np.float32)
    if smooth_ms <= 0 or x.size == 0:
        return x
    tau = smooth_ms / 1000.0
    a = np.exp(-1.0 / (max(1, sr) * tau))
    y = np.empty_like(x)
    acc = float(x[0])
    for i in range(x.size):
        acc = a * acc + (1.0 - a) * float(x[i])
        y[i] = acc
    return y

def _envelope_follower(x, sr, attack_ms, release_ms, detector="abs"):
    x = np.asarray(x, dtype=np.float32)
    d = x * x if detector == "rms" else np.abs(x)

    atk = max(attack_ms, 0.001) / 1000.0
    rel = max(release_ms, 0.001) / 1000.0
    a_atk = np.exp(-1.0 / (sr * atk))
    a_rel = np.exp(-1.0 / (sr * rel))

    env = np.empty_like(d)
    e = 0.0
    for i in range(d.size):
        target = float(d[i])
        a = a_atk if target > e else a_rel
        e = a * e + (1.0 - a) * target
        env[i] = e

    if detector == "rms":
        env = np.sqrt(env + 1e-12)
    return env.astype(np.float32)

def _lfo_values(num_blocks, hop, sr, rate_hz, phase01, shape):
    if num_blocks <= 0:
        return np.zeros((0,), dtype=np.float32)
    t = (np.arange(num_blocks, dtype=np.float32) * hop) / float(sr)
    ph = (2.0 * np.pi) * (rate_hz * t + phase01)

    if shape == "sine":
        v = np.sin(ph)
    elif shape == "triangle":
        saw = (ph / (2.0*np.pi)) % 1.0
        v = 4.0 * np.abs(saw - 0.5) - 1.0
    elif shape == "square":
        v = np.sign(np.sin(ph))
    elif shape == "saw":
        saw = (ph / (2.0*np.pi)) % 1.0
        v = 2.0 * saw - 1.0
    else:
        raise ValueError("Unknown LFO shape: " + shape)
    return v.astype(np.float32)

def _automation_values(num_blocks, hop, sr, points, interp="linear"):
    if num_blocks <= 0:
        return np.zeros((0,), dtype=np.float32)
    if not points:
        return np.zeros((num_blocks,), dtype=np.float32)

    pts = sorted([(float(t), float(v)) for t, v in points], key=lambda x: x[0])
    times = np.array([p[0] for p in pts], dtype=np.float32)
    vals  = np.array([p[1] for p in pts], dtype=np.float32)
    bt = (np.arange(num_blocks, dtype=np.float32) * hop) / float(sr)

    if interp == "step":
        out = np.empty((num_blocks,), dtype=np.float32)
        j = 0
        cur = float(vals[0])
        for i in range(num_blocks):
            while j + 1 < len(times) and bt[i] >= times[j+1]:
                j += 1
                cur = float(vals[j])
            out[i] = cur
        return out
    return np.interp(bt, times, vals).astype(np.float32)

# =========================
# Onset detection (block-based)
# =========================

def _smooth_frames(v, n=2):
    v = np.asarray(v, dtype=np.float32)
    if n <= 0 or v.size == 0:
        return v
    k = int(n)
    w = np.ones((k,), dtype=np.float32) / k
    return np.convolve(v, w, mode="same").astype(np.float32)

def _onset_strength_energy(x_mono, hop):
    n = x_mono.shape[0]
    num_blocks = int(np.ceil(n / hop))
    E = np.zeros((num_blocks,), dtype=np.float32)
    for bi in range(num_blocks):
        s = bi * hop
        e = min(s + hop, n)
        blk = x_mono[s:e]
        if blk.size:
            E[bi] = float(np.mean(blk * blk) + 1e-12)
    logE = np.log(E + 1e-12).astype(np.float32)
    d = np.diff(logE, prepend=logE[0])
    return np.maximum(d, 0.0).astype(np.float32)

def _onset_strength_spectral_flux(x_mono, hop):
    n = x_mono.shape[0]
    num_blocks = int(np.ceil(n / hop))
    fft_size = _next_pow2(hop)
    prev = None
    flux = np.zeros((num_blocks,), dtype=np.float32)
    window = np.hanning(hop).astype(np.float32)
    for bi in range(num_blocks):
        s = bi * hop
        e = min(s + hop, n)
        frame = np.zeros((hop,), dtype=np.float32)
        frame[:max(0, e - s)] = x_mono[s:e]
        frame *= window
        mag = np.abs(np.fft.rfft(frame, n=fft_size)).astype(np.float32)
        if prev is None:
            prev = mag
            continue
        diff = np.maximum(mag - prev, 0.0)
        flux[bi] = float(np.sum(diff))
        prev = mag
    return flux

def _detect_onsets_blocks(x_mono, sr, hop, opts):
    n = x_mono.shape[0]
    num_blocks = int(np.ceil(n / hop))

    method = str(opts.get("onset_method", "energy"))
    thresh = float(opts.get("onset_thresh", 2.5))
    smooth_frames = int(opts.get("onset_smooth_frames", 2))
    min_interval_ms = float(opts.get("onset_min_interval_ms", 80.0))

    s = _onset_strength_spectral_flux(x_mono, hop) if method == "spectral_flux" else _onset_strength_energy(x_mono, hop)
    s = _smooth_frames(s, smooth_frames)

    mu = float(np.mean(s)) if s.size else 0.0
    sd = max(float(np.std(s)), 1e-12)
    T = mu + thresh * sd

    onset = np.zeros((num_blocks,), dtype=np.bool_)
    refractory_blocks = int(np.ceil((min_interval_ms / 1000.0) * sr / hop)) if hop > 0 else 0
    last = -10**9

    for i in range(num_blocks):
        if i - last < refractory_blocks:
            continue
        left = s[i - 1] if i - 1 >= 0 else -1e9
        right = s[i + 1] if i + 1 < num_blocks else -1e9
        if s[i] > T and s[i] >= left and s[i] >= right:
            onset[i] = True
            last = i
    return onset

# =========================
# Driver -> a_per_block in [0,1]
# =========================

def _compute_a_per_block(x_mono, sr, hop, num_blocks, opts, ctrl_wav_bytes: bytes):
    driver = str(opts.get("driver", "static"))

    if driver == "static":
        a = np.full((num_blocks,), float(opts.get("morph", 0.5)), dtype=np.float32)
        return np.clip(a, 0.0, 1.0)

    if driver == "ramp":
        aS = float(opts.get("ramp_start", 0.0))
        aE = float(opts.get("ramp_end", 1.0))
        t = np.linspace(0.0, 1.0, num_blocks, dtype=np.float32) if num_blocks > 1 else np.array([0.0], dtype=np.float32)
        a = (1.0 - t) * aS + t * aE
        return np.clip(a, 0.0, 1.0)

    if driver == "control_wav":
        if ctrl_wav_bytes is None or len(ctrl_wav_bytes) == 0:
            raise ValueError("control_wav driver requires CTRL_WAV_BYTES")

        srC, c = _read_wav_bytes(bytes(ctrl_wav_bytes))
        if srC != sr:
            mode = str(opts.get("resample_mode", "off"))
            c_m = _resample(c_m, srC, sr, mode)

        c_m = c.mean(axis=1).astype(np.float32)
        measure = str(opts.get("ctrl_measure", "abs_mean"))
        smooth_ms = float(opts.get("ctrl_smooth_ms", 0.0))
        norm = str(opts.get("ctrl_norm", "minmax"))

        a = np.zeros((num_blocks,), dtype=np.float32)
        for bi in range(num_blocks):
            s = bi * hop
            e = min(s + hop, c_m.shape[0])
            if e <= s:
                v = 0.0
            else:
                blk = c_m[s:e]
                if measure == "rms":
                    v = float(np.sqrt(np.mean(blk * blk) + 1e-12))
                elif measure == "peak":
                    v = float(np.max(np.abs(blk)))
                else:
                    v = float(np.mean(np.abs(blk)))
            a[bi] = v

        blocks_per_sec = max(1, int(sr / hop))
        a = _one_pole_smooth(a, sr=blocks_per_sec, smooth_ms=smooth_ms)
        a = _norm_01(a, norm)
        return np.clip(a, 0.0, 1.0)

    if driver == "envelope":
        attack = float(opts.get("env_attack_ms", 5.0))
        release = float(opts.get("env_release_ms", 60.0))
        detector = str(opts.get("env_detector", "abs"))
        norm = str(opts.get("env_norm", "percentile"))

        env = _envelope_follower(x_mono, sr, attack, release, detector=detector)
        a = np.zeros((num_blocks,), dtype=np.float32)
        for bi in range(num_blocks):
            s = bi * hop
            e = min(s + hop, env.shape[0])
            a[bi] = float(np.mean(env[s:e])) if e > s else 0.0

        a = _norm_01(a, norm)
        return np.clip(a, 0.0, 1.0)

    if driver == "lfo":
        rate = float(opts.get("lfo_rate_hz", 0.25))
        phase01 = float(opts.get("lfo_phase", 0.0))
        depth = float(opts.get("lfo_depth", 1.0))
        offset = float(opts.get("lfo_offset", 0.5))
        shape = str(opts.get("lfo_shape", "sine"))

        v = _lfo_values(num_blocks, hop, sr, rate_hz=rate, phase01=phase01, shape=shape)
        a = offset + depth * 0.5 * v
        return np.clip(a, 0.0, 1.0)

    if driver == "automation":
        auto_json = str(opts.get("auto_json", "[]"))
        interp = str(opts.get("auto_interp", "linear"))
        try:
            pts = json.loads(auto_json)
        except Exception as e:
            raise ValueError("Failed to parse automation JSON: " + str(e))
        a = _automation_values(num_blocks, hop, sr, pts, interp=interp)
        return np.clip(a, 0.0, 1.0)

    raise ValueError("Unknown driver: " + driver)



# =========================
# Resampling
# =========================

def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int):
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out or x.size == 0:
        return x
    n_out = int(np.round(x.size * (float(sr_out) / float(sr_in))))
    if n_out <= 1:
        return np.zeros((max(1, n_out),), dtype=np.float32)
    t = np.linspace(0.0, x.size - 1, n_out, dtype=np.float32)
    i0 = np.floor(t).astype(np.int32)
    i1 = np.minimum(i0 + 1, x.size - 1)
    frac = t - i0.astype(np.float32)
    return ((1.0 - frac) * x[i0] + frac * x[i1]).astype(np.float32)

def _resample_sinc(x: np.ndarray, sr_in: int, sr_out: int, taps: int = 32):
    \"\"\"Windowed-sinc resampler (offline). Reasonable quality without SciPy.
    taps should be even; larger = better quality but slower.
    \"\"\"
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out or x.size == 0:
        return x
    if taps < 8:
        taps = 8
    if taps % 2 == 1:
        taps += 1

    ratio = float(sr_out) / float(sr_in)
    n_out = int(np.round(x.size * ratio))
    if n_out <= 1:
        return np.zeros((max(1, n_out),), dtype=np.float32)

    # Output sample positions in input-sample units
    t = (np.arange(n_out, dtype=np.float32) / ratio)

    half = taps // 2
    # Kernel sample offsets
    k = np.arange(-half + 1, half + 1, dtype=np.float32)  # taps length

    # Hann window
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(taps, dtype=np.float32) / (taps - 1)))

    y = np.zeros((n_out,), dtype=np.float32)
    for i in range(n_out):
        ti = float(t[i])
        ci = int(np.floor(ti))
        frac = ti - ci

        idx = (ci + k).astype(np.int32)
        # Bounds
        idx_clip = np.clip(idx, 0, x.size - 1)
        xg = x[idx_clip]

        # sinc at (n - frac)
        z = (k - frac)
        # Avoid divide by zero
        sinc = np.where(np.abs(z) < 1e-8, 1.0, np.sin(np.pi * z) / (np.pi * z)).astype(np.float32)

        h = (sinc * win).astype(np.float32)
        # Normalize kernel gain
        s = float(np.sum(h))
        if abs(s) > 1e-12:
            h /= s

        y[i] = float(np.sum(xg * h))
    return y

def _resample(x: np.ndarray, sr_in: int, sr_out: int, mode: str):
    mode = (mode or "off").lower()
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    if mode == "off":
        raise ValueError("Sample rate mismatch and resampling is OFF")
    if mode == "sinc":
        return _resample_sinc(x, sr_in, sr_out)
    return _resample_linear(x, sr_in, sr_out)

# =========================
# Output processing
# =========================

def _highpass_1p(x: np.ndarray, sr: int, fc_hz: float):
    \"\"\"Simple 1-pole high-pass (DC blocker style).
    y[n] = a * (y[n-1] + x[n] - x[n-1])
    \"\"\"
    if fc_hz <= 0:
        return x
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    dt = 1.0 / float(sr)
    rc = 1.0 / (2.0 * np.pi * float(fc_hz))
    a = rc / (rc + dt)
    y = np.empty_like(x)
    y_prev = 0.0
    x_prev = float(x[0])
    y[0] = 0.0
    for i in range(1, x.size):
        xi = float(x[i])
        yi = a * (y_prev + xi - x_prev)
        y[i] = yi
        y_prev = yi
        x_prev = xi
    return y

def _peak_normalize(y: np.ndarray, target: float = 0.999):
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-12:
        return y
    return y * (float(target) / peak)

# =========================
# Bank index selection
# =========================

def _bank_indices(num_blocks, onset_blocks, a_per_block, bank_len, opts):
    # Returns idx_per_block in [0, bank_len-1]
    if bank_len <= 0:
        return np.zeros((num_blocks,), dtype=np.int32)

    hold = str(opts.get("bank_hold", "hold"))              # hold | always
    mode = str(opts.get("bank_index_mode", "random"))      # random | from_driver | round_robin
    seed = opts.get("rand_seed", None)
    rng = np.random.default_rng(None if seed is None else int(seed))

    idx = np.zeros((num_blocks,), dtype=np.int32)
    cur = 0

    def pick(i, cur):
        if mode == "from_driver":
            a = float(a_per_block[i]) if a_per_block is not None and a_per_block.size else 0.0
            j = int(np.round(a * (bank_len - 1)))
            return int(np.clip(j, 0, bank_len - 1))
        if mode == "round_robin":
            return (cur + 1) % bank_len
        return int(rng.integers(0, bank_len))

    if hold == "always":
        for i in range(num_blocks):
            cur = pick(i, cur)
            idx[i] = cur
        return idx

    # hold: update on onsets (or at i==0), otherwise keep current
    for i in range(num_blocks):
        trigger = (i == 0) or (bool(onset_blocks[i]) if onset_blocks is not None else False)
        if trigger:
            cur = pick(i, cur)
        idx[i] = cur
    return idx

# =========================
# Bank convolution (partitioned OLA)
# =========================

def _precompute_bank(ir_bank_mono, part_size, fft_size):
    parts_lists = [_partition_ir(ir, part_size) for ir in ir_bank_mono]
    n_parts = max((len(p) for p in parts_lists), default=0)
    bank = []
    for parts in parts_lists:
        while len(parts) < n_parts:
            parts.append(np.zeros((part_size,), dtype=np.float32))
        H_parts = _build_H_parts(parts, fft_size)
        bank.append({"n_parts": n_parts, "H_parts": H_parts})
    return bank, n_parts

def _process_channel_partitioned_bank(x, bank_pre, idx_per_block, block_size, part_size):
    hop = block_size
    fft_size = _next_pow2(block_size + part_size)

    n = x.shape[0]
    y = np.zeros((n + part_size + block_size,), dtype=np.float32)
    ola = np.zeros((fft_size - hop,), dtype=np.float32)

    if not bank_pre:
        return x.copy()

    n_parts = bank_pre[0]["n_parts"]
    X_hist = [np.zeros((fft_size // 2 + 1,), dtype=np.complex64) for _ in range(n_parts)]

    num_blocks = int(np.ceil(n / hop))
    if idx_per_block.shape[0] != num_blocks:
        raise ValueError("idx_per_block length mismatch")

    for bi in range(num_blocks):
        s = bi * hop
        e = min(s + hop, n)
        xb = np.zeros((hop,), dtype=np.float32)
        xb[:e - s] = x[s:e]

        xb_pad = np.zeros((fft_size,), dtype=np.float32)
        xb_pad[:hop] = xb
        X0 = np.fft.rfft(xb_pad).astype(np.complex64)

        X_hist.pop()
        X_hist.insert(0, X0)

        idx = int(idx_per_block[bi])
        idx = max(0, min(idx, len(bank_pre) - 1))
        H_parts = bank_pre[idx]["H_parts"]

        Y = np.zeros_like(X0)
        for k in range(n_parts):
            Y += X_hist[k] * H_parts[k]

        yt = np.fft.irfft(Y, n=fft_size).astype(np.float32)

        out = yt[:hop].copy()
        out[:ola.shape[0]] += ola
        ola = yt[hop:].copy()

        y[s:s+hop] += out

    return y[:n].astype(np.float32)

# =========================
# Public entrypoint
# =========================

def process_bytes(in_wav_bytes, opts, ctrl_wav_bytes=None, ir_bank_list=None):
    sr, x = _read_wav_bytes(bytes(in_wav_bytes))

    # Bank required
    if ir_bank_list is None:
        raise ValueError("IR_BANK_LIST is required")
    ir_bank_mono = []
    for item in list(ir_bank_list):
        bb = bytes(item)
        if len(bb) == 0:
            continue
        srI, irX = _read_wav_bytes(bb)
        if srI != sr:
            mode = str(opts.get("resample_mode", "off"))
            ir_m = irX.mean(axis=1).astype(np.float32)
            ir_m = _resample(ir_m, srI, sr, mode)
            ir_bank_mono.append(ir_m)
            continue
        ir_bank_mono.append(irX.mean(axis=1).astype(np.float32))
    if len(ir_bank_mono) == 0:
        raise ValueError("IR bank is empty")

    block_size = int(opts["block_size"])
    part_size  = int(opts["part_size"])
    hop = block_size
    n = x.shape[0]
    num_blocks = int(np.ceil(n / hop))

    x_mono = x.mean(axis=1).astype(np.float32)

    # Driver control signal a_per_block (for from_driver mode)
    a_per_block = _compute_a_per_block(
        x_mono=x_mono,
        sr=sr,
        hop=hop,
        num_blocks=num_blocks,
        opts=opts,
        ctrl_wav_bytes=bytes(ctrl_wav_bytes or b""),
    )

    onset_blocks = None
    if bool(opts.get("onset_enable", True)):
        onset_blocks = _detect_onsets_blocks(x_mono, sr, hop, opts)

    idx_per_block = _bank_indices(num_blocks, onset_blocks, a_per_block, len(ir_bank_mono), opts)

    # Nested convolution option affects IRs used
    nest_enable = bool(opts.get("nest_enable", False))
    nest_mode = str(opts.get("nest_mode", "ir_within_ir"))

    bank_source = ir_bank_mono
    if nest_enable and nest_mode == "ir_within_ir":
        bank_source = [np.convolve(irX, irX).astype(np.float32) for irX in ir_bank_mono]

    fft_size = _next_pow2(block_size + part_size)
    bank_pre, _ = _precompute_bank(bank_source, part_size=part_size, fft_size=fft_size)

    ch = x.shape[1]
    y = np.zeros_like(x, dtype=np.float32)
    for c in range(ch):
        y[:, c] = _process_channel_partitioned_bank(x[:, c].astype(np.float32), bank_pre, idx_per_block, block_size, part_size)

    if nest_enable and nest_mode == "two_stage":
        y2 = np.zeros_like(y, dtype=np.float32)
        for c in range(ch):
            y2[:, c] = _process_channel_partitioned_bank(y[:, c], bank_pre, idx_per_block, block_size, part_size)
        y = y2

    # Post-processing: high-pass + normalize
    out_hpf_hz = float(opts.get("out_hpf_hz", 0.0) or 0.0)
    out_normalize = bool(opts.get("out_normalize", False))

    if out_hpf_hz > 0.0:
        for c in range(y.shape[1]):
            y[:, c] = _highpass_1p(y[:, c], sr, out_hpf_hz)

    if out_normalize:
        y = _peak_normalize(y, target=0.999)
    else:
        # Safety limiting only (avoid clipping)
        peak = float(np.max(np.abs(y))) if y.size else 1.0
        if peak > 0.999:
            y *= (0.999 / peak)

    return _write_wav_bytes(sr, y)
