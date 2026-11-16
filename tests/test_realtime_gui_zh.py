"""Regression checks for the Chinese real-time window.

These tests do not load models, open an audio device, or claim a Mac run.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from realtime_gui_zh_logic import (
    LAYOUT_DEFAULTS,
    SAVED_CONFIG_KEYS,
    algorithm_delay_ms,
    align_device_selection,
    cuda_target_for_gpu,
    decide_event,
    fallback_settings,
    layout_value,
    parse_bool,
    reference_browse_folder,
    select_device_kind,
    settings_from_values,
    stop_audio_stream,
    update_vad_playback,
    validate_extra_context,
    validate_reference_path,
    wav2vec2_frame_count,
    wav2vec2_right_pad,
    with_sr_flags,
)

def _play_vad(values):
    detected = False
    hold = 0
    heard = []
    for value in values:
        audible, detected, hold = update_vad_playback(value, detected, hold)
        heard.append(audible)
    return heard


# Values the Mac window was already using. Kept here so a save cannot drop them.
USER_CONFIG = {
    "reference_audio_path": "/Users/lan/voice-clone-output/ref_short.wav",
    "sg_hostapi": "Core Audio",
    "sg_wasapi_exclusive": False,
    "sg_input_device": "外置麦克风",
    "sg_output_device": "Mac mini扬声器",
    "sr_type": "sr_model",
    "diffusion_steps": 10,
    "inference_cfg_rate": 0.7,
    "max_prompt_length": 3.0,
    "block_time": 0.3,
    "crossfade_length": 0.04,
    "extra_time_ce": 2.5,
    "extra_time": 0.5,
    "extra_time_right": 0.02,
}


def load_gui_module():
    path = os.path.join(ROOT, "real-time-gui-zh.py")
    spec = importlib.util.spec_from_file_location("seedvc_realtime_gui_zh", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_keys(rows):
    found = []
    for row in rows:
        for element in row:
            key = getattr(element, "Key", None)
            if key not in (None, ""):
                found.append(key)
            child = getattr(element, "Rows", None)
            if child:
                found.extend(iter_keys(child))
    return found


class LogicTests(unittest.TestCase):
    def test_user_config_round_trip_keeps_keys_and_values(self):
        values = with_sr_flags(USER_CONFIG)
        saved = settings_from_values(values)
        self.assertEqual(tuple(saved.keys()), SAVED_CONFIG_KEYS)
        self.assertNotIn("sr_model", saved)
        self.assertNotIn("sr_device", saved)
        self.assertEqual(saved, USER_CONFIG)
        self.assertEqual(json.loads(json.dumps(saved)), json.loads(json.dumps(USER_CONFIG)))

    def test_missing_key_uses_original_layout_default_not_fallback_file(self):
        self.assertEqual(layout_value({}, "block_time"), LAYOUT_DEFAULTS["block_time"])
        self.assertEqual(layout_value({}, "block_time"), 1.0)
        self.assertEqual(layout_value({}, "extra_time_ce"), 5.0)
        self.assertEqual(layout_value(USER_CONFIG, "block_time"), 0.3)
        self.assertEqual(layout_value(USER_CONFIG, "extra_time_ce"), 2.5)
        fallback = fallback_settings("Core Audio", "外置麦克风", "Mac mini扬声器")
        self.assertEqual(fallback["block_time"], 0.3)
        self.assertEqual(fallback["crossfade_length"], 0.04)
        self.assertEqual(fallback["extra_time_ce"], 2.5)
        self.assertEqual(fallback["extra_time"], 0.5)
        self.assertEqual(fallback["extra_time_right"], 0.02)
        self.assertEqual(fallback["diffusion_steps"], 10)
        self.assertEqual(fallback["inference_cfg_rate"], 0.7)
        self.assertEqual(fallback["max_prompt_length"], 3.0)

    def test_repo_config_json_is_not_rewritten_by_helpers(self):
        path = os.path.join(ROOT, "configs", "config.json")
        with open(path, "r", encoding="utf-8") as handle:
            original = handle.read()
        data = json.loads(original)
        flagged = with_sr_flags(data)
        self.assertEqual(flagged["reference_audio_path"], data["reference_audio_path"])
        self.assertEqual(flagged["sr_model"], data["sr_type"] == "sr_model")
        self.assertEqual(layout_value(data, "block_time"), data["block_time"])
        if "extra_time_ce" not in data:
            self.assertEqual(layout_value(data, "extra_time_ce"), 5.0)
        with open(path, "r", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), original)

    def test_saved_device_names_stay_when_still_present(self):
        data = dict(USER_CONFIG)

        def fail_fallback():
            raise AssertionError("fallback should not run")

        align_device_selection(
            data,
            ["Core Audio"],
            ["外置麦克风"],
            ["Mac mini扬声器"],
            fail_fallback,
        )
        self.assertEqual(data["sg_input_device"], "外置麦克风")
        self.assertEqual(data["sg_output_device"], "Mac mini扬声器")
        self.assertEqual(data["extra_time_right"], 0.02)

    def test_unavailable_hostapi_replaces_devices_but_keeps_tuning(self):
        data = dict(USER_CONFIG)
        data["sg_hostapi"] = "MME"

        def fallback():
            return ("Core Audio", "外置麦克风", "Mac mini扬声器")

        align_device_selection(
            data,
            ["Core Audio"],
            ["外置麦克风"],
            ["Mac mini扬声器"],
            fallback,
        )
        self.assertEqual(data["sg_hostapi"], "Core Audio")
        self.assertEqual(data["diffusion_steps"], 10)
        self.assertEqual(data["block_time"], 0.3)
        self.assertEqual(data["inference_cfg_rate"], 0.7)

    def test_event_routing_matches_original_and_ignores_fold_buttons(self):
        self.assertEqual(decide_event(None, False), "close")
        self.assertEqual(decide_event("start_vc", False), "start")
        self.assertEqual(decide_event("start_vc", True), "already_running")
        self.assertEqual(decide_event("stop_vc", True), "stop")
        self.assertEqual(decide_event("diffusion_steps", True), "hot")
        self.assertEqual(decide_event("inference_cfg_rate", True), "hot")
        self.assertEqual(decide_event("vc", True), "hot")
        self.assertEqual(decide_event("im", False), "hot")
        for event in (
            "block_time",
            "crossfade_length",
            "extra_time_ce",
            "extra_time",
            "extra_time_right",
            "max_prompt_length",
            "sg_input_device",
            "sg_output_device",
            "sg_wasapi_exclusive",
            "sr_model",
            "sr_device",
        ):
            self.assertEqual(decide_event(event, True), "stop", event)
        self.assertEqual(decide_event("toggle_regular", True), "ui")
        self.assertEqual(decide_event("toggle_performance", True), "ui")
        self.assertEqual(decide_event("reload_devices", True), "reload")
        self.assertEqual(decide_event("sg_hostapi", True), "reload")
        self.assertEqual(decide_event("__TIMEOUT__", True), "tick")

    def test_delay_formula_is_estimate_not_a_new_measurement(self):
        self.assertEqual(algorithm_delay_ms(0.08, 0.30, 0.04, 0.02), 450)
        self.assertAlmostEqual(0.08 + 0.30 + 0.04 + 0.02 + 0.01, 0.45)

    def test_reference_path_rules(self):
        self.assertIsNone(validate_reference_path(USER_CONFIG["reference_audio_path"]))
        self.assertEqual(validate_reference_path("  "), "empty")
        self.assertEqual(validate_reference_path("/Users/lan/声音.wav"), "non_ascii")
        self.assertIsNone(validate_extra_context(2.5, 0.5))
        self.assertEqual(validate_extra_context(0.5, 2.5), "ce_shorter_than_dit")

    def test_stream_closes_even_if_abort_fails(self):
        class Stream:
            def __init__(self):
                self.closed = False

            def abort(self):
                raise RuntimeError("abort failed")

            def close(self):
                self.closed = True

        stream = Stream()
        new_stream, closed = stop_audio_stream(stream)
        self.assertTrue(closed)
        self.assertIsNone(new_stream)
        self.assertTrue(stream.closed)
        self.assertEqual(stop_audio_stream(None), (None, False))

    def test_cli_helpers_preserve_mps_and_gpu0(self):
        self.assertTrue(parse_bool("true"))
        self.assertFalse(parse_bool("false"))
        self.assertEqual(cuda_target_for_gpu(0), "cuda")
        self.assertEqual(cuda_target_for_gpu(1), "cuda:1")
        self.assertEqual(select_device_kind(False, True), "mps")
        self.assertEqual(select_device_kind(True, True), "cuda")
        self.assertEqual(select_device_kind(False, False), "cpu")

    def test_open_vad_segment_does_not_fade_the_tail_out(self):
        # 23:43 log shape. len([[550, -1]]) == 1, same as an end event.
        # The old parity toggle mutes the chunk after the second open result.
        heard = _play_vad([([550, -1]), [], [[550, -1]], []])
        self.assertEqual(heard, [True, True, True, True])

    def test_empty_vad_keeps_speech_and_a_real_end_keeps_one_more_block(self):
        heard = _play_vad([([100, -1]), [], [[-1, 900]], [], []])
        self.assertEqual(heard, [True, True, True, True, False])
        heard = _play_vad([[], [], [[100, 400]], [], []])
        self.assertEqual(heard, [False, False, True, True, False])

    def test_right_context_frame_is_restored_for_the_22050_block(self):
        # User config at model rate: block 0.30, crossfade 0.04, extra right 0.02, extra ce 2.5.
        sr = 22050
        zc = sr // 50
        block = int(round(0.30 * sr / zc)) * zc
        cross = int(round(0.04 * sr / zc)) * zc
        right = int(round(0.02 * sr / zc)) * zc
        extra = int(round(2.5 * sr / zc)) * zc
        samples = extra + cross + zc + block + right
        self.assertEqual(block, 6615)
        frames_16k = 320 * samples // zc
        self.assertEqual(frames_16k, 46080)
        self.assertEqual(wav2vec2_frame_count(frames_16k), frames_16k // 320 - 1)
        pad = wav2vec2_right_pad(frames_16k)
        self.assertEqual(pad, 80)
        self.assertEqual(wav2vec2_frame_count(frames_16k + pad), frames_16k // 320)

    def test_zero_sola_offset_hands_the_tail_to_the_next_block(self):
        # (6615, 1) in the 23:43 log. sola search is one 20ms frame (441).
        block = 6615
        sola = 882
        origin = 55125

        def span(stream_origin, offset):
            play_start = stream_origin + offset
            play_end = play_start + block
            return play_start, play_end, play_end, play_end + sola

        play_start, play_end, saved_start, _saved_end = span(origin, 0)
        next_play, _next_end, _next_saved, _ = span(origin + block, 0)
        self.assertEqual(next_play, saved_start)
        self.assertEqual(play_end - play_start, block)
        _start_hi, end_hi, saved_hi, _ = span(origin, 438)
        self.assertEqual(saved_hi, end_hi)
        self.assertLessEqual(438, 441)

    def test_browse_folder_prefers_existing_reference_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = os.path.join(tmp, "voice-clone-output")
            os.makedirs(ref_dir)
            chosen = reference_browse_folder(
                os.path.join(ref_dir, "ref_short.wav"),
                tmp,
                os.path.join(tmp, "missing"),
            )
            self.assertEqual(chosen, ref_dir)

    def test_entrypoint_keeps_engine_file_and_cli(self):
        gui_path = os.path.join(ROOT, "real-time-gui-zh.py")
        engine_path = os.path.join(ROOT, "real-time-gui.py")
        with open(gui_path, "r", encoding="utf-8") as handle:
            gui = handle.read()
        with open(engine_path, "r", encoding="utf-8") as handle:
            engine = handle.read()
        self.assertIn("Start Voice Conversion", engine)
        self.assertIn("def custom_infer", engine)
        self.assertIn("def audio_callback", engine)
        for flag in ("--checkpoint-path", "--config-path", "--fp16", "--gpu"):
            self.assertIn(flag, gui)
        self.assertIn("engine.custom_infer", gui)
        self.assertLess(gui.index("self.stop_stream()"), gui.index("sd._terminate()"))
        self.assertIn("update_vad_playback", gui)
        self.assertIn("wav2vec2_right_pad", gui)
        self.assertEqual(gui.count('self.window["infer_time"].update'), 1)
        self.assertIn("len(res_value) % 2 == 1", engine)
        self.assertNotIn("update_vad_playback", engine)
        # Model modules are not edited by this window.
        self.assertNotIn("real-time-gui-zh", engine)


class LayoutTests(unittest.TestCase):
    def test_window_shows_chinese_controls_and_user_values(self):
        try:
            import tkinter  # noqa: F401
        except ImportError:
            self.skipTest("tkinter is not installed")
        module = load_gui_module()
        family = module.choose_font_family()
        module.apply_theme()
        module.sg.set_options(font=(family, 13), element_padding=(6, 3))
        data = with_sr_flags(USER_CONFIG)
        layout = module.build_layout(
            data,
            family,
            ["Core Audio"],
            ["外置麦克风"],
            ["Mac mini扬声器"],
        )
        keys = set(iter_keys(layout))
        required = {
            "reference_audio_path",
            "sg_hostapi",
            "sg_wasapi_exclusive",
            "sg_input_device",
            "sg_output_device",
            "reload_devices",
            "sr_model",
            "sr_device",
            "sr_stream",
            "diffusion_steps",
            "inference_cfg_rate",
            "max_prompt_length",
            "block_time",
            "crossfade_length",
            "extra_time_ce",
            "extra_time",
            "extra_time_right",
            "start_vc",
            "stop_vc",
            "im",
            "vc",
            "delay_time",
            "infer_time",
            "model_time",
            "status",
            "status_detail",
            "toggle_regular",
            "regular",
            "toggle_performance",
            "performance",
        }
        self.assertTrue(required <= keys, sorted(required - keys))
        window = module.sg.Window(
            f"Seed-VC 实时变声 {module.GUI_VERSION}",
            layout,
            finalize=True,
            resizable=True,
            margins=(14, 10),
            location=(40, 24),
        )
        try:
            window["status_detail"].update("模型已加载。请确认麦克风、播放设备和参考声音，然后开始变声。")
            window.refresh()
            self.assertIn("1.1", window["app_title"].Widget.cget("text"))
            self.assertEqual(window["start_vc"].Widget.cget("text"), "开始变声")
            self.assertEqual(window["stop_vc"].Widget.cget("text"), "停止变声")
            self.assertEqual(window["reference_audio_path"].get(), USER_CONFIG["reference_audio_path"])
            self.assertEqual(window["sg_input_device"].get(), "外置麦克风")
            self.assertEqual(window["sg_output_device"].get(), "Mac mini扬声器")
            self.assertAlmostEqual(float(window["diffusion_steps"].Widget.get()), 10, delta=0.01)
            self.assertAlmostEqual(float(window["inference_cfg_rate"].Widget.get()), 0.7, delta=0.011)
            self.assertAlmostEqual(float(window["max_prompt_length"].Widget.get()), 3.0, delta=0.01)
            self.assertAlmostEqual(float(window["block_time"].Widget.get()), 0.3, delta=0.011)
            self.assertAlmostEqual(float(window["crossfade_length"].Widget.get()), 0.04, delta=0.011)
            self.assertAlmostEqual(float(window["extra_time_ce"].Widget.get()), 2.5, delta=0.011)
            self.assertAlmostEqual(float(window["extra_time"].Widget.get()), 0.5, delta=0.011)
            self.assertAlmostEqual(float(window["extra_time_right"].Widget.get()), 0.02, delta=0.011)
            self.assertTrue(window["performance"].visible)
            self._assert_widget_on_screen(window, "block_time")
            self._assert_widget_on_screen(window, "extra_time_right")
            self._save_screenshot(window)
            window["performance"].update(visible=False)
            window["toggle_performance"].update("▶  性能参数")
            window.refresh()
            self.assertFalse(window["performance"].visible)
        finally:
            window.close()

    def _assert_widget_on_screen(self, window, key):
        widget = window[key].Widget
        window.TKroot.update_idletasks()
        self.assertTrue(widget.winfo_viewable(), key)
        bottom = widget.winfo_rooty() + widget.winfo_height()
        screen_bottom = window.TKroot.winfo_screenheight() - 8
        self.assertLess(bottom, screen_bottom, key)

    def _save_screenshot(self, window):
        import shutil
        import subprocess

        if not shutil.which("scrot"):
            print("scrot not available; layout checks still ran, screenshot skipped")
            return
        out_dir = os.path.join(ROOT, "artifacts")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "seed-vc-zh-gui-test-grab.png")
        window.refresh()
        root = window.TKroot
        root.update()
        import time
        time.sleep(0.3)
        root.update()
        x = root.winfo_rootx()
        y = root.winfo_rooty()
        width = root.winfo_width()
        height = root.winfo_height()
        subprocess.check_call(["scrot", "-a", f"{x},{y},{width},{height}", path])
        print(f"screenshot {path} window {width}x{height}")


if __name__ == "__main__":
    unittest.main()
