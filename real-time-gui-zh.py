#!/usr/bin/env python3
"""Simplified Chinese desktop window for Seed-VC real-time voice conversion.

The audio engine, checkpoint arguments, and configs/inuse/config.json keys stay
in the original real-time-gui.py. This file only replaces the window. Run the
original script unchanged if you want the English window back.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import FreeSimpleGUI as sg
    from realtime_gui_zh_logic import (
        algorithm_delay_ms,
        align_device_selection,
        decide_event,
        fallback_settings,
        layout_value,
        reference_browse_folder,
        settings_from_values,
        stop_audio_stream,
        validate_extra_context,
        validate_reference_path,
        with_sr_flags,
        parse_bool,
        cuda_target_for_gpu,
        select_device_kind,
    )
except ImportError:
    sys.stderr.write(
        "缺少 realtime_gui_zh_logic.py 或 FreeSimpleGUI。"
        "请把 realtime_gui_zh_logic.py 与 real-time-gui-zh.py 放在同一目录。\n"
    )
    raise

# Bound in main() after the original module sets the Mac MPS environment.
engine = None
np = None
sd = None
torch = None
librosa = None
F = None
tat = None
flag_vc = False

BG = "#F3F6F8"
TEXT = "#1B2836"
MUTED = "#5E6D7A"
ACCENT = "#1F4E79"
START_BG = "#1E7A4D"
STOP_BG = "#A33B32"
STATUS_COLOR = {
    "正在加载模型": "#8A5A12",
    "就绪": "#1F4E79",
    "运行中": "#1E7A4D",
    "已停止": "#5C6770",
    "出错": "#A33B32",
}


def load_engine():
    path = os.path.join(HERE, "real-time-gui.py")
    name = "seedvc_realtime_engine"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def apply_theme():
    sg.LOOK_AND_FEEL_TABLE["SeedZh"] = {
        "BACKGROUND": BG,
        "TEXT": TEXT,
        "INPUT": "#FFFFFF",
        "TEXT_INPUT": TEXT,
        "SCROLL": ACCENT,
        "BUTTON": ("#FFFFFF", ACCENT),
        "PROGRESS": (ACCENT, "#D5DDE3"),
        "BORDER": 1,
        "SLIDER_DEPTH": 0,
        "PROGRESS_DEPTH": 0,
        "ACCENT1": ACCENT,
        "ACCENT2": MUTED,
        "ACCENT3": START_BG,
    }
    sg.theme("SeedZh")


def choose_font_family():
    preferred = (
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Songti SC",
        "STHeiti",
        "PingFang HK",
        "WenQuanYi Micro Hei",
        "Droid Sans Fallback",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "Arial Unicode MS",
    )
    try:
        import tkinter as tk
        import tkinter.font as tkfont

        root = tk.Tk()
        root.withdraw()
        have = set(tkfont.families())
        root.destroy()
        try:
            tk._default_root = None
        except Exception:
            pass
    except Exception:
        return "PingFang SC"
    for name in preferred:
        if name in have:
            return name
    return "TkDefaultFont"


def _with_current(options, current):
    items = list(options or [])
    if current and current not in items:
        items = [current] + items
    return items


def _slider(key, data, span, resolution):
    return sg.Slider(
        range=span,
        key=key,
        resolution=resolution,
        orientation="h",
        default_value=layout_value(data, key),
        enable_events=True,
        size=(28, 14),
    )


def _hint(family, text):
    return sg.Text(text, text_color=MUTED, font=(family, 11), pad=((4, 4), (0, 2)))


def build_layout(data, family, hostapis=None, input_devices=None, output_devices=None):
    data = with_sr_flags(data)
    ref = layout_value(data, "reference_audio_path")
    host_value = layout_value(data, "sg_hostapi")
    input_value = layout_value(data, "sg_input_device")
    output_value = layout_value(data, "sg_output_device")
    title_font = (family, 20)
    button_font = (family, 16)
    label_font = (family, 13)

    regular = [
        [sg.Text("扩散步数", font=label_font, size=(22, 1)), _slider("diffusion_steps", data, (1, 30), 1)],
        [_hint(family, "步数少一些更快。实时常用 10。改变后立即作用，不停止变声。")],
        [sg.Text("引导强度", font=label_font, size=(22, 1)), _slider("inference_cfg_rate", data, (0.0, 1.0), 0.1)],
        [_hint(family, "越大越贴近参考音色，0 更快。常用 0.7。改变后立即作用，不停止变声。")],
        [sg.Text("参考声音时长（秒）", font=label_font, size=(22, 1)), _slider("max_prompt_length", data, (1.0, 20.0), 0.5)],
        [_hint(family, "只用参考音频开头的这么多秒。常用 3 秒。改这个值会停止变声，需要再点开始。")],
    ]
    performance = [
        [sg.Text("块时长（秒）", font=label_font, size=(22, 1)), _slider("block_time", data, (0.04, 3.0), 0.02)],
        [_hint(family, "每次处理多久的声音。要大于本块处理耗时。常用 0.30 秒。")],
        [sg.Text("交叉淡化（秒）", font=label_font, size=(22, 1)), _slider("crossfade_length", data, (0.02, 0.5), 0.02)],
        [_hint(family, "两块声音交界处的重叠。常用 0.04 秒。")],
        [sg.Text("内容编码左侧（秒）", font=label_font, size=(22, 1)), _slider("extra_time_ce", data, (0.5, 10.0), 0.1)],
        [_hint(family, "向左多看的时间，不能小于 DiT 左侧。常用 2.5 秒。")],
        [sg.Text("DiT 左侧（秒）", font=label_font, size=(22, 1)), _slider("extra_time", data, (0.5, 10.0), 0.1)],
        [_hint(family, "变声模型向左多看的时间。常用 0.5 秒。")],
        [sg.Text("右侧额外（秒）", font=label_font, size=(22, 1)), _slider("extra_time_right", data, (0.02, 10.0), 0.02)],
        [_hint(family, "向右多看的时间，会计入算法延迟。常用 0.02 秒。")],
    ]

    layout = [
        [
            sg.Text("Seed-VC 实时变声", font=title_font, text_color=ACCENT, pad=((4, 12), (2, 2))),
            sg.Text("就绪", key="status", font=(family, 13), text_color="#FFFFFF", background_color=STATUS_COLOR["就绪"], pad=((10, 10), (4, 4))),
        ],
        [_hint(family, "原英文界面 real-time-gui.py 仍可使用。本窗口使用同一套实时引擎和同一份配置。")],
        [sg.Text("", key="status_detail", font=(family, 12), text_color=TEXT, size=(78, 1))],
        [
            sg.Button("开始变声", key="start_vc", font=button_font, size=(12, 2), button_color=("#FFFFFF", START_BG), pad=((4, 10), (6, 8))),
            sg.Button("停止变声", key="stop_vc", font=button_font, size=(12, 2), button_color=("#FFFFFF", STOP_BG), pad=((4, 18), (6, 8))),
            sg.Text("工作模式", font=label_font, pad=((8, 6), (0, 0))),
            sg.Radio("实时变声", "function", key="vc", default=True, enable_events=True, font=label_font),
            sg.Radio("仅监听输入", "function", key="im", default=False, enable_events=True, font=label_font),
        ],
        [
            sg.Frame(
                "参考声音",
                [
                    [
                        sg.Input(default_text=ref, key="reference_audio_path", size=(62, 1), font=label_font),
                        sg.FileBrowse(
                            "选择参考声音",
                            initial_folder=reference_browse_folder(
                                ref,
                                os.getcwd(),
                                os.path.join(os.getcwd(), "examples", "reference"),
                            ),
                            file_types=[
                                ("WAV Files", "*.wav"),
                                ("MP3 Files", "*.mp3"),
                                ("FLAC Files", "*.flac"),
                                ("M4A Files", "*.m4a"),
                                ("OGG Files", "*.ogg"),
                                ("Opus Files", "*.opus"),
                            ],
                            font=label_font,
                        ),
                    ],
                    [_hint(family, "要克隆的音色。开始变声后使用这段音频的开头。路径仍不能包含中文或其它非英文字符。")],
                ],
                font=(family, 13),
                pad=((4, 4), (6, 6)),
            )
        ],
        [
            sg.Frame(
                "音频设备",
                [
                    [
                        sg.Text("设备类型", font=label_font, size=(10, 1)),
                        sg.Combo(_with_current(hostapis, host_value), key="sg_hostapi", default_value=host_value, enable_events=True, size=(28, 1), font=label_font),
                        sg.Checkbox(
                            "WASAPI 独占（仅 Windows，Mac 保持关闭）",
                            key="sg_wasapi_exclusive",
                            default=layout_value(data, "sg_wasapi_exclusive"),
                            enable_events=True,
                            font=label_font,
                        ),
                    ],
                    [
                        sg.Text("麦克风", font=label_font, size=(10, 1)),
                        sg.Combo(_with_current(input_devices, input_value), key="sg_input_device", default_value=input_value, enable_events=True, size=(48, 1), font=label_font),
                    ],
                    [
                        sg.Text("播放设备", font=label_font, size=(10, 1)),
                        sg.Combo(_with_current(output_devices, output_value), key="sg_output_device", default_value=output_value, enable_events=True, size=(48, 1), font=label_font),
                    ],
                    [
                        sg.Button("刷新设备", key="reload_devices", font=label_font, button_color=("#FFFFFF", ACCENT)),
                        sg.Radio("使用模型采样率", "sr_type", key="sr_model", default=layout_value(data, "sr_model"), enable_events=True, font=label_font),
                        sg.Radio("使用设备采样率", "sr_type", key="sr_device", default=layout_value(data, "sr_device"), enable_events=True, font=label_font),
                        sg.Text("采样率", font=label_font),
                        sg.Text("尚未开始", key="sr_stream", font=label_font, size=(10, 1)),
                    ],
                    [_hint(family, "按名称选择设备。同一台电脑上，同名设备在不同次启动时的编号可能不一样。刷新设备会先停止变声。")],
                ],
                font=(family, 13),
                pad=((4, 4), (4, 6)),
            )
        ],
        [
            sg.Frame(
                "时间",
                [
                    [
                        sg.Text("算法延迟（估算，毫秒）", font=label_font),
                        sg.Text("—", key="delay_time", font=(family, 16), text_color=ACCENT, size=(8, 1)),
                        sg.Text("本块处理耗时（毫秒）", font=label_font),
                        sg.Text("—", key="infer_time", font=(family, 16), text_color=ACCENT, size=(8, 1)),
                        sg.Text("其中模型推理（毫秒）", font=label_font),
                        sg.Text("—", key="model_time", font=(family, 16), text_color=ACCENT, size=(8, 1)),
                    ],
                    [_hint(family, "算法延迟不是实测的端到端延迟。它等于声卡报告的输出延迟 + 块时长 + 交叉淡化 + 右侧额外时间 + 0.01 秒。")],
                    [_hint(family, "本块处理耗时是这一块回调的墙钟时间，含语音检测、重采样、模型推理和拼接。模型推理只计变声模型那一段。都不是麦克风到扬声器的实测延迟。")],
                ],
                font=(family, 13),
                pad=((4, 4), (4, 6)),
            )
        ],
        [
            sg.Column(
                [
                    [sg.Button("▼  常规参数", key="toggle_regular", font=label_font, button_color=("#FFFFFF", ACCENT), pad=((4, 4), (6, 2)))],
                    [sg.pin(sg.Column(regular, key="regular", visible=True, pad=(4, 0)))],
                ],
                pad=(0, 0),
                vertical_alignment="top",
            ),
            sg.Column(
                [
                    [sg.Button("▼  性能参数（改动会停止变声）", key="toggle_performance", font=label_font, button_color=("#FFFFFF", ACCENT), pad=((4, 4), (6, 2)))],
                    [sg.pin(sg.Column(performance, key="performance", visible=True, pad=(4, 0)))],
                ],
                pad=((12, 0), (0, 0)),
                vertical_alignment="top",
            ),
        ],
    ]
    return layout


class GUIConfig:
    def __init__(self):
        self.reference_audio_path = ""
        self.diffusion_steps = 10
        self.sr_type = "sr_model"
        self.block_time = 0.25
        self.threhold = -60
        self.crossfade_time = 0.05
        self.extra_time_ce = 2.5
        self.extra_time = 0.5
        self.extra_time_right = 2.0
        self.I_noise_reduce = False
        self.O_noise_reduce = False
        self.inference_cfg_rate = 0.7
        self.sg_hostapi = ""
        self.wasapi_exclusive = False
        self.sg_input_device = ""
        self.sg_output_device = ""


class GUI:
    def __init__(self, args):
        self.args = args
        self.gui_config = GUIConfig()
        self.config = engine.Config()
        self.function = "vc"
        self.delay_time = 0
        self.hostapis = []
        self.input_devices = []
        self.output_devices = []
        self.input_devices_indices = []
        self.output_devices_indices = []
        self.stream = None
        self.model_set = None
        self.vad_model = None
        self.model_ready = False
        self.device_error = None
        self._model_ms = None
        self.font_family = "TkDefaultFont"
        try:
            self.update_devices()
        except Exception as exc:
            self.device_error = exc
        self.launcher()

    def _set_status(self, state, detail):
        color = STATUS_COLOR.get(state, ACCENT)
        self.window["status"].update(state, text_color="#FFFFFF", background_color=color)
        self.window["status_detail"].update(detail)

    def _popup_error(self, title, exc):
        text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        details = traceback.format_exc()
        try:
            sg.popup_scrolled(details, title=title, size=(72, 14), font=(self.font_family, 11))
        except Exception:
            sg.popup(title, text)

    def load(self):
        """Read configs/inuse/config.json. Do not rewrite a readable file."""
        os.makedirs("configs/inuse", exist_ok=True)
        path = os.path.join("configs", "inuse", "config.json")
        if not os.path.exists(path):
            import shutil

            shutil.copy(os.path.join("configs", "config.json"), path)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return self._write_fallback(path)
        if not isinstance(data, dict):
            return self._write_fallback(path)
        data = with_sr_flags(data)

        def fallback():
            self.update_devices()
            return (
                self.hostapis[0],
                self.input_devices[self.input_devices_indices.index(sd.default.device[0])],
                self.output_devices[self.output_devices_indices.index(sd.default.device[1])],
            )

        if data.get("sg_hostapi") in self.hostapis:
            try:
                self.update_devices(hostapi_name=data.get("sg_hostapi"))
            except Exception as exc:
                self.device_error = exc
                return data
        try:
            align_device_selection(
                data,
                self.hostapis,
                self.input_devices,
                self.output_devices,
                fallback,
            )
        except Exception as exc:
            self.device_error = exc
        return data

    def _write_fallback(self, path):
        hostapi = self.hostapis[0] if self.hostapis else ""
        try:
            in_name = self.input_devices[self.input_devices_indices.index(sd.default.device[0])]
            out_name = self.output_devices[self.output_devices_indices.index(sd.default.device[1])]
        except Exception:
            in_name = self.input_devices[0] if self.input_devices else ""
            out_name = self.output_devices[0] if self.output_devices else ""
        data = fallback_settings(hostapi, in_name, out_name)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
        except Exception:
            pass
        return with_sr_flags(data)

    def launcher(self):
        self.config = engine.Config()
        try:
            data = self.load()
        except Exception as exc:
            self.device_error = exc
            data = with_sr_flags(fallback_settings("", "", ""))
        self.gui_config.sg_hostapi = data.get("sg_hostapi", "")
        self.gui_config.sg_input_device = data.get("sg_input_device", "")
        self.gui_config.sg_output_device = data.get("sg_output_device", "")
        self.gui_config.sg_wasapi_exclusive = data.get("sg_wasapi_exclusive", False)
        self.font_family = choose_font_family()
        apply_theme()
        sg.set_options(font=(self.font_family, 13), element_padding=(6, 3))
        self.window = sg.Window(
            "Seed-VC 实时变声",
            build_layout(
                data,
                self.font_family,
                self.hostapis,
                self.input_devices,
                self.output_devices,
            ),
            finalize=True,
            resizable=True,
            margins=(14, 10),
            location=(40, 24),
        )
        self._fill_device_widgets(data)
        self._set_status("正在加载模型", "正在加载模型和语音活动检测，窗口会暂时无法点击")
        self.window["start_vc"].update(disabled=True)
        self.window.refresh()
        try:
            self.model_set = engine.load_models(self.args)
            from funasr import AutoModel

            self.vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")
        except Exception as exc:
            self.model_ready = False
            self._set_status("出错", "模型加载失败，还不能开始变声")
            self._popup_error("模型加载失败", exc)
            self.event_handler()
            return
        self.model_ready = True
        self.window["start_vc"].update(disabled=False)
        if self.device_error is not None:
            self._set_status("出错", "模型已加载，但读取音频设备失败：" + str(self.device_error))
        else:
            self._set_status("就绪", "模型已加载。请确认麦克风、播放设备和参考声音，然后开始变声。")
        self.event_handler()

    def _fill_device_widgets(self, data):
        host = data.get("sg_hostapi", "")
        if host not in self.hostapis and self.hostapis:
            host = self.hostapis[0]
        self.window["sg_hostapi"].update(values=self.hostapis, value=host)
        self.window["sg_input_device"].update(
            values=self.input_devices, value=data.get("sg_input_device", "")
        )
        self.window["sg_output_device"].update(
            values=self.output_devices, value=data.get("sg_output_device", "")
        )

    def _toggle(self, event):
        key = "regular" if event == "toggle_regular" else "performance"
        title = "常规参数" if key == "regular" else "性能参数"
        visible = not self.window[key].visible
        self.window[key].update(visible=visible)
        arrow = "▼" if visible else "▶"
        suffix = "（改动会停止变声）" if key == "performance" else ""
        self.window[event].update(f"{arrow}  {title}{suffix}")
        self.window.refresh()

    def event_handler(self):
        global flag_vc
        try:
            while True:
                # Blocking read, same as the original. A timeout event would hit
                # the catch-all stop rule, so timeout is intentionally unused.
                event, values = self.window.read()
                action = decide_event(event, flag_vc)
                try:
                    if action == "close":
                        self.stop_stream()
                        break
                    if action == "ui":
                        self._toggle(event)
                        continue
                    if action == "reload":
                        was_running = flag_vc or self.stream is not None
                        self._reload_devices(values)
                        detail = "设备列表已刷新"
                        if was_running:
                            detail += "，变声已停止"
                        state = "就绪" if self.model_ready and self.device_error is None else "出错"
                        if not self.model_ready:
                            state = "出错"
                            detail = "模型未加载。" + detail
                        self._set_status(state, detail)
                        continue
                    if action == "already_running":
                        self._set_status("运行中", "变声已在运行，没有再次打开音频流")
                        continue
                    if action == "start":
                        self._start_from_values(values)
                        continue
                    if action == "hot":
                        self._hot_update(event, values)
                        continue
                    if action == "stop":
                        was_running = flag_vc or self.stream is not None
                        self.stop_stream()
                        if not was_running:
                            continue
                        if event == "stop_vc":
                            self._set_status("已停止", "变声已停止")
                        else:
                            self._set_status("已停止", "这个参数不能在运行中直接改，变声已停止。确认后再点开始变声。")
                except Exception as exc:
                    if action not in ("ui", "hot"):
                        self.stop_stream()
                    self._set_status("出错", "操作没有完成")
                    self._popup_error("操作没有完成", exc)
        finally:
            self.stop_stream()
            try:
                self.window.close()
            except Exception:
                pass

    def _reload_devices(self, values):
        self.gui_config.sg_hostapi = values.get("sg_hostapi", "")
        self.update_devices(hostapi_name=values.get("sg_hostapi"))
        if self.gui_config.sg_hostapi not in self.hostapis and self.hostapis:
            self.gui_config.sg_hostapi = self.hostapis[0]
        if (
            self.gui_config.sg_input_device not in self.input_devices
            and self.input_devices
        ):
            self.gui_config.sg_input_device = self.input_devices[0]
        if self.output_devices and self.gui_config.sg_output_device not in self.output_devices:
            self.gui_config.sg_output_device = self.output_devices[0]
        self.window["sg_hostapi"].update(values=self.hostapis, value=self.gui_config.sg_hostapi)
        self.window["sg_input_device"].update(values=self.input_devices, value=self.gui_config.sg_input_device)
        self.window["sg_output_device"].update(values=self.output_devices, value=self.gui_config.sg_output_device)

    def _start_from_values(self, values):
        if not self.model_ready:
            self._set_status("出错", "模型还没有加载成功，不能开始变声")
            sg.popup("模型还没有加载成功，不能开始变声。")
            return
        if self.set_values(values) is not True:
            return
        engine.printt("cuda_is_available: %s", torch.cuda.is_available())
        self._set_status("运行中", "正在打开音频流")
        self.start_vc()
        try:
            settings = settings_from_values(values)
            os.makedirs("configs/inuse", exist_ok=True)
            with open(os.path.join("configs", "inuse", "config.json"), "w", encoding="utf-8") as handle:
                json.dump(settings, handle)
        except Exception as exc:
            self._set_status("运行中", "变声已开始，但配置没有写成功")
            self._popup_error("配置没有写成功", exc)
            return
        try:
            self._update_delay_labels(values)
        except Exception:
            self.window["delay_time"].update("—")
        self._set_status("运行中", "变声进行中")

    def _update_delay_labels(self, values):
        self.window["sr_stream"].update(self.gui_config.samplerate)
        if self.stream is None:
            return
        output_latency = self.stream.latency[-1]
        self.delay_time = output_latency + values["block_time"] + values["crossfade_length"] + values["extra_time_right"] + 0.01
        self.window["delay_time"].update(
            algorithm_delay_ms(
                output_latency,
                values["block_time"],
                values["crossfade_length"],
                values["extra_time_right"],
            )
        )

    def _hot_update(self, event, values):
        if event == "diffusion_steps":
            self.gui_config.diffusion_steps = values["diffusion_steps"]
        elif event == "inference_cfg_rate":
            self.gui_config.inference_cfg_rate = values["inference_cfg_rate"]
        elif event in ("vc", "im"):
            self.function = event
            if event == "im":
                self.window["status_detail"].update("当前只监听输入，声音不会变声")
            else:
                self.window["status_detail"].update("当前是实时变声")

    def set_values(self, values):
        path = values["reference_audio_path"]
        problem = validate_reference_path(path)
        if problem == "empty":
            sg.popup("请先选择参考声音。")
            return False
        if problem == "non_ascii":
            sg.popup("参考声音路径含有非英文字符。这是原引擎的限制，请把文件放到纯英文路径后再试。")
            return False
        if not os.path.isfile(path):
            sg.popup("找不到参考声音文件。")
            return False
        try:
            settings = settings_from_values(values)
        except ValueError:
            sg.popup("请选择使用模型采样率，或使用设备采样率。")
            return False
        if validate_extra_context(settings["extra_time_ce"], settings["extra_time"]):
            sg.popup("内容编码的左侧时间必须大于或等于 DiT 的左侧时间。")
            return False
        try:
            self.set_devices(settings["sg_input_device"], settings["sg_output_device"])
        except (ValueError, IndexError):
            sg.popup("找不到所选的麦克风或播放设备。请点刷新设备后再试。")
            return False
        self.gui_config.sg_hostapi = settings["sg_hostapi"]
        self.gui_config.sg_wasapi_exclusive = settings["sg_wasapi_exclusive"]
        self.gui_config.sg_input_device = settings["sg_input_device"]
        self.gui_config.sg_output_device = settings["sg_output_device"]
        self.gui_config.reference_audio_path = settings["reference_audio_path"]
        self.gui_config.sr_type = settings["sr_type"]
        self.gui_config.diffusion_steps = settings["diffusion_steps"]
        self.gui_config.inference_cfg_rate = settings["inference_cfg_rate"]
        self.gui_config.max_prompt_length = settings["max_prompt_length"]
        self.gui_config.block_time = settings["block_time"]
        self.gui_config.crossfade_time = settings["crossfade_length"]
        self.gui_config.extra_time_ce = settings["extra_time_ce"]
        self.gui_config.extra_time = settings["extra_time"]
        self.gui_config.extra_time_right = settings["extra_time_right"]
        return True

    def start_vc(self):
        if engine.device.type == "mps":
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()
        self.reference_wav, _ = librosa.load(
            self.gui_config.reference_audio_path, sr=self.model_set[-1]["sampling_rate"]
        )
        self.gui_config.samplerate = (
            self.model_set[-1]["sampling_rate"]
            if self.gui_config.sr_type == "sr_model"
            else self.get_device_samplerate()
        )
        self.gui_config.channels = self.get_device_channels()
        self.zc = self.gui_config.samplerate // 50
        self.block_frame = (
            int(np.round(self.gui_config.block_time * self.gui_config.samplerate / self.zc))
            * self.zc
        )
        self.block_frame_16k = 320 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(np.round(self.gui_config.crossfade_time * self.gui_config.samplerate / self.zc))
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(np.round(self.gui_config.extra_time_ce * self.gui_config.samplerate / self.zc))
            * self.zc
        )
        self.extra_frame_right = (
            int(np.round(self.gui_config.extra_time_right * self.gui_config.samplerate / self.zc))
            * self.zc
        )
        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
            + self.extra_frame_right,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            320 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_frame_right // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.gui_config.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)
        if self.model_set[-1]["sampling_rate"] != self.gui_config.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.model_set[-1]["sampling_rate"],
                new_freq=self.gui_config.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)
        else:
            self.resampler2 = None
        self.vad_cache = {}
        self.vad_chunk_size = min(500, 1000 * self.gui_config.block_time)
        self.vad_speech_detected = False
        self.set_speech_detected_false_at_end_flag = False
        self.start_stream()

    def start_stream(self):
        global flag_vc
        if flag_vc and self.stream is not None:
            return
        if self.stream is not None:
            self.stop_stream()
        flag_vc = True
        try:
            if (
                "WASAPI" in self.gui_config.sg_hostapi
                and self.gui_config.sg_wasapi_exclusive
            ):
                extra_settings = sd.WasapiSettings(exclusive=True)
            else:
                extra_settings = None
            self.stream = sd.Stream(
                callback=self.audio_callback,
                blocksize=self.block_frame,
                samplerate=self.gui_config.samplerate,
                channels=self.gui_config.channels,
                dtype="float32",
                extra_settings=extra_settings,
            )
            self.stream.start()
        except Exception:
            flag_vc = False
            self.stream, _closed = stop_audio_stream(self.stream)
            raise

    def stop_stream(self):
        global flag_vc
        flag_vc = False
        self.stream, _closed = stop_audio_stream(self.stream)

    def audio_callback(self, indata, outdata, frames, times, status):
        """Same block processing as real-time-gui.py."""
        global flag_vc
        print(indata.shape)
        start_time = time.perf_counter()
        self._model_ms = None
        indata = librosa.to_mono(indata.T)

        if engine.device.type == "mps":
            start_event = torch.mps.event.Event(enable_timing=True)
            end_event = torch.mps.event.Event(enable_timing=True)
            torch.mps.synchronize()
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
        start_event.record()
        indata_16k = librosa.resample(indata, orig_sr=self.gui_config.samplerate, target_sr=16000)
        res = self.vad_model.generate(
            input=indata_16k,
            cache=self.vad_cache,
            is_final=False,
            chunk_size=self.vad_chunk_size,
        )
        res_value = res[0]["value"]
        print(res_value)
        if len(res_value) % 2 == 1 and not self.vad_speech_detected:
            self.vad_speech_detected = True
        elif len(res_value) % 2 == 1 and self.vad_speech_detected:
            self.set_speech_detected_false_at_end_flag = True
        end_event.record()
        if engine.device.type == "mps":
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Time taken for VAD: {elapsed_time_ms}ms")

        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.config.device)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
        self.input_wav_res[-320 * (indata.shape[0] // self.zc + 1) :] = torch.from_numpy(
            librosa.resample(
                self.input_wav[-indata.shape[0] - 2 * self.zc :].cpu().numpy(),
                orig_sr=self.gui_config.samplerate,
                target_sr=16000,
            )[320:]
        )
        print(f"preprocess time: {time.perf_counter() - start_time:.2f}")
        if self.function == "vc":
            if self.gui_config.extra_time_ce - self.gui_config.extra_time < 0:
                raise ValueError("Content encoder extra context must be greater than DiT extra context!")
            if engine.device.type == "mps":
                start_event = torch.mps.event.Event(enable_timing=True)
                end_event = torch.mps.event.Event(enable_timing=True)
                torch.mps.synchronize()
            else:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
            start_event.record()
            infer_wav = engine.custom_infer(
                self.model_set,
                self.reference_wav,
                self.gui_config.reference_audio_path,
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.skip_tail,
                self.return_length,
                int(self.gui_config.diffusion_steps),
                self.gui_config.inference_cfg_rate,
                self.gui_config.max_prompt_length,
                self.gui_config.extra_time_ce - self.gui_config.extra_time,
            )
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
            end_event.record()
            if engine.device.type == "mps":
                torch.mps.synchronize()
            else:
                torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            self._model_ms = elapsed_time_ms
            print(f"Time taken for VC: {elapsed_time_ms}ms")
            if not self.vad_speech_detected:
                infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])
        elif self.gui_config.I_noise_reduce:
            infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame :].clone()

        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
            )
            + 1e-8
        )
        tensor = cor_nom[0, 0] / cor_den[0, 0]
        if tensor.numel() > 1:
            if sys.platform == "darwin":
                _, sola_offset = torch.max(tensor, dim=0)
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(tensor, dim=0).item()
        else:
            sola_offset = tensor.item()
        print(f"sola_offset = {int(sola_offset)}")

        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
        outdata[:] = (
            infer_wav[: self.block_frame].repeat(self.gui_config.channels, 1).t().cpu().numpy()
        )

        total_time = time.perf_counter() - start_time
        if flag_vc:
            try:
                self.window["infer_time"].update(int(total_time * 1000))
                if self._model_ms is None:
                    self.window["model_time"].update("—")
                else:
                    self.window["model_time"].update(int(self._model_ms))
            except Exception:
                pass
        if self.set_speech_detected_false_at_end_flag:
            self.vad_speech_detected = False
            self.set_speech_detected_false_at_end_flag = False
        print(f"Infer time: {total_time:.2f}")

    def update_devices(self, hostapi_name=None):
        """Refresh PortAudio devices. Close an open stream before terminate()."""
        self.stop_stream()
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.hostapis = [hostapi["name"] for hostapi in hostapis]
        if hostapi_name not in self.hostapis:
            hostapi_name = self.hostapis[0]
        self.input_devices = [
            d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices = [
            d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.input_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.device_error = None

    def set_devices(self, input_device, output_device):
        sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
        sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]
        engine.printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
        engine.printt("Output device: %s:%s", str(sd.default.device[1]), output_device)

    def get_device_samplerate(self):
        return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])

    def get_device_channels(self):
        max_input_channels = sd.query_devices(device=sd.default.device[0])["max_input_channels"]
        max_output_channels = sd.query_devices(device=sd.default.device[1])["max_output_channels"]
        return min(max_input_channels, max_output_channels, 2)


def main():
    global engine, np, sd, torch, librosa, F, tat
    engine = load_engine()
    import numpy as _np
    import sounddevice as _sd
    import torch as _torch
    import librosa as _librosa
    import torch.nn.functional as _F
    import torchaudio.transforms as _tat

    np = _np
    sd = _sd
    torch = _torch
    librosa = _librosa
    F = _F
    tat = _tat

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--config-path", type=str, default=None, help="Path to the vocoder checkpoint")
    parser.add_argument("--fp16", type=parse_bool, nargs="?", const=True, help="Whether to use fp16", default=True)
    parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
    args = parser.parse_args()
    kind = select_device_kind(torch.cuda.is_available(), torch.backends.mps.is_available())
    if kind == "cuda":
        engine.device = torch.device(cuda_target_for_gpu(args.gpu))
    elif kind == "mps":
        engine.device = torch.device("mps")
    else:
        engine.device = torch.device("cpu")
    GUI(args)


if __name__ == "__main__":
    main()
