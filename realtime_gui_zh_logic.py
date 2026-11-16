"""Pure helpers for the Chinese real-time GUI.

No audio, torch, or Tk imports. The desktop entry point uses these so config
keys, event routing, and the algorithm-delay formula stay testable.
"""

import argparse
import os

# Keys written to configs/inuse/config.json when conversion starts.
# Same set as real-time-gui.py. Derived radios (sr_model / sr_device) are not saved.
SAVED_CONFIG_KEYS = (
    "reference_audio_path",
    "sg_hostapi",
    "sg_wasapi_exclusive",
    "sg_input_device",
    "sg_output_device",
    "sr_type",
    "diffusion_steps",
    "inference_cfg_rate",
    "max_prompt_length",
    "block_time",
    "crossfade_length",
    "extra_time_ce",
    "extra_time",
    "extra_time_right",
)

# Slider/radio defaults used when a key is absent from a loaded file.
# These match the original layout's data.get(...) defaults, which are not the
# same numbers as the fallback file written for a corrupt config.
LAYOUT_DEFAULTS = {
    "reference_audio_path": "",
    "sg_hostapi": "",
    "sg_wasapi_exclusive": False,
    "sg_input_device": "",
    "sg_output_device": "",
    "sr_model": True,
    "sr_device": False,
    "diffusion_steps": 10,
    "inference_cfg_rate": 0.7,
    "max_prompt_length": 3.0,
    "block_time": 1.0,
    "crossfade_length": 0.1,
    "extra_time_ce": 5.0,
    "extra_time": 5.0,
    "extra_time_right": 2.0,
}

# Hot updates: do not stop the stream. Same events as the original GUI.
HOT_EVENTS = frozenset({"diffusion_steps", "inference_cfg_rate", "vc", "im"})

# Appearance-only events added by the Chinese window.
UI_EVENTS = frozenset({"toggle_regular", "toggle_performance"})

# Device-list events. The window stops and closes any stream before reinit.
RELOAD_EVENTS = frozenset({"reload_devices", "sg_hostapi"})


def parse_bool(value):
    """Same acceptance rules as modules.commons.str2bool."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def cuda_target_for_gpu(gpu):
    """Preserve the original quirk: gpu id 0 selects plain 'cuda'."""
    return f"cuda:{gpu}" if gpu else "cuda"


def select_device_kind(cuda_available, mps_available):
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def with_sr_flags(data):
    """Copy config and set the two radios from sr_type, as load() does."""
    out = dict(data)
    sr_type = out.get("sr_type", "sr_model")
    out["sr_type"] = sr_type
    out["sr_model"] = sr_type == "sr_model"
    out["sr_device"] = sr_type == "sr_device"
    return out


def layout_value(data, key):
    return data.get(key, LAYOUT_DEFAULTS[key])


def fallback_settings(hostapi, input_name, output_name):
    """In-memory defaults matching the original corrupt-config branch.

    Numbers are the original fallback (block 0.30, crossfade 0.04, and so on),
    not the layout defaults used when a key is simply missing.
    """
    return {
        "sg_hostapi": hostapi,
        "sg_wasapi_exclusive": False,
        "sg_input_device": input_name,
        "sg_output_device": output_name,
        "sr_type": "sr_model",
        "block_time": 0.3,
        "crossfade_length": 0.04,
        "extra_time_ce": 2.5,
        "extra_time": 0.5,
        "extra_time_right": 0.02,
        "diffusion_steps": 10,
        "inference_cfg_rate": 0.7,
        "max_prompt_length": 3.0,
    }


def align_device_selection(data, hostapis, input_devices, output_devices, fallback):
    """Keep saved device names when they still exist. Otherwise call fallback().

    fallback() must return (hostapi, input_name, output_name). Other keys in
    data are left untouched. This function does not write the config file.
    """
    if (
        hostapis
        and data.get("sg_hostapi") in hostapis
        and data.get("sg_input_device") in input_devices
        and data.get("sg_output_device") in output_devices
    ):
        return data
    if not hostapis:
        return data
    hostapi, input_name, output_name = fallback()
    data["sg_hostapi"] = hostapi
    data["sg_input_device"] = input_name
    data["sg_output_device"] = output_name
    return data


def settings_from_values(values):
    """Build the JSON object saved on a successful start."""
    flags = [bool(values["sr_model"]), bool(values["sr_device"])]
    if True not in flags:
        raise ValueError("sr_type")
    settings = {
        "reference_audio_path": values["reference_audio_path"],
        "sg_hostapi": values["sg_hostapi"],
        "sg_wasapi_exclusive": values["sg_wasapi_exclusive"],
        "sg_input_device": values["sg_input_device"],
        "sg_output_device": values["sg_output_device"],
        "sr_type": ["sr_model", "sr_device"][flags.index(True)],
        "diffusion_steps": values["diffusion_steps"],
        "inference_cfg_rate": values["inference_cfg_rate"],
        "max_prompt_length": values["max_prompt_length"],
        "block_time": values["block_time"],
        "crossfade_length": values["crossfade_length"],
        "extra_time_ce": values["extra_time_ce"],
        "extra_time": values["extra_time"],
        "extra_time_right": values["extra_time_right"],
    }
    if tuple(settings.keys()) != SAVED_CONFIG_KEYS:
        raise RuntimeError("config key order drifted")
    return settings


def validate_reference_path(path):
    """Return 'empty', 'non_ascii', or None.

    Non-ASCII rejection is the original GUI check, kept on purpose.
    """
    if len(str(path).strip()) == 0:
        return "empty"
    if any(ord(ch) > 127 for ch in str(path)):
        return "non_ascii"
    return None


def validate_extra_context(extra_time_ce, extra_time):
    """Engine raises if content-encoder context is shorter than DiT context."""
    if float(extra_time_ce) - float(extra_time) < 0:
        return "ce_shorter_than_dit"
    return None


def decide_event(event, running):
    """Classify one window event.

    'stop' is the original catch-all: anything that is not a hot update and
    not the start button stops the stream. Fold/expand buttons are 'ui' so
    they do not stop audio. 'reload' tells the window to close the stream
    before rebuilding the PortAudio device list. '__TIMEOUT__' only refreshes
    the timing labels; it must not stop the stream.
    """
    if event is None:
        return "close"
    if event == "__TIMEOUT__":
        return "tick"
    if event in UI_EVENTS:
        return "ui"
    if event in RELOAD_EVENTS:
        return "reload"
    if event == "start_vc":
        return "start" if not running else "already_running"
    if event in HOT_EVENTS:
        return "hot"
    if event == "stop_vc" or event != "start_vc":
        return "stop"
    return "ignore"


# Blocks to keep synthesizing after FunASR reports a finished segment.
# Mute replaces the whole next waveform with zeros, and the 40ms crossfade
# then fades the saved tail into that silence. One more block lets a release
# that crossed the chunk boundary play before the fade.
TAIL_HOLD_BLOCKS = 1

# facebook/wav2vec2-xls-r-300m feature CNN. Valid convolutions, no pad.
_WAV2VEC2_CONV = (
    (10, 5),
    (3, 2),
    (3, 2),
    (3, 2),
    (3, 2),
    (2, 2),
    (2, 2),
)


def wav2vec2_frame_count(num_samples):
    """Frames emitted by the wav2vec2 feature CNN for a 16 kHz waveform."""
    length = int(num_samples)
    for kernel, stride in _WAV2VEC2_CONV:
        if length < kernel:
            return 0
        length = (length - kernel) // stride + 1
    return length


def wav2vec2_right_pad(num_samples):
    """Trailing zeros so the CNN covers floor(num_samples / 320) frames.

    A 16 kHz buffer built as an exact multiple of 320 loses its last 20ms
    frame. With the current block math that frame is extra_time_right.
    """
    target = int(num_samples) // 320
    if target <= 0:
        return 0
    pad = 0
    while wav2vec2_frame_count(num_samples + pad) < target and pad < 320:
        pad += 1
    return pad


def _vad_segments(res_value):
    """Normalize one FunASR streaming `value` into [beg_ms, end_ms] pairs.

    Documented streaming shapes are [[beg, -1]] (open), [[-1, end]] (end),
    [[beg, end]] (both in this chunk), and [] (no event). end < 0 means the
    segment is still open. The 23:43 log prints [[550, -1]].
    """
    if not res_value:
        return []
    first = res_value[0]
    if isinstance(first, (list, tuple)):
        return [list(seg) for seg in res_value if isinstance(seg, (list, tuple)) and len(seg) >= 2]
    if all(isinstance(item, (int, float)) for item in res_value):
        if len(res_value) % 2 == 1:
            return [[res_value[0], -1]]
        if len(res_value) >= 2:
            return [[res_value[-2], res_value[-1]]]
    return []


def update_vad_playback(res_value, detected, hold, tail_hold=TAIL_HOLD_BLOCKS):
    """Return (audible, detected_after, hold_after).

    [] does not mean this chunk is silence. A second [[beg, -1]] must not
    close the gate: len(value) is 1 for an open segment, an end, and a
    complete segment, so a parity toggle fades the following tail to silence.
    A finished segment (end >= 0) still plays this chunk, then `tail_hold`
    more chunks, then mutes.
    """
    segments = _vad_segments(res_value)
    if not segments:
        if detected:
            return True, True, hold
        if hold > 0:
            return True, False, hold - 1
        return False, False, 0
    if any(seg[1] < 0 for seg in segments):
        return True, True, 0
    return True, False, tail_hold


def algorithm_delay_seconds(output_latency, block_time, crossfade_length, extra_time_right):
    """Original estimate. Not a measured microphone-to-speaker delay."""
    return (
        float(output_latency)
        + float(block_time)
        + float(crossfade_length)
        + float(extra_time_right)
        + 0.01
    )


def algorithm_delay_ms(output_latency, block_time, crossfade_length, extra_time_right):
    seconds = algorithm_delay_seconds(
        output_latency, block_time, crossfade_length, extra_time_right
    )
    return int(round(seconds * 1000))


def stop_audio_stream(stream):
    """Close a stream object even if the running flag was already cleared.

    Returns (new_stream, closed). abort() errors still attempt close().
    """
    if stream is None:
        return None, False
    try:
        stream.abort()
    except Exception:
        pass
    try:
        stream.close()
    except Exception:
        pass
    return None, True


def reference_browse_folder(reference_path, cwd, examples_reference):
    if reference_path:
        folder = os.path.dirname(reference_path)
        if folder and os.path.isdir(folder):
            return folder
    if examples_reference and os.path.isdir(examples_reference):
        return examples_reference
    return cwd
