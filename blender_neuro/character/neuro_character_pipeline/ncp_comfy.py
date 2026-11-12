# -*- coding: utf-8 -*-
"""
neuro_character_pipeline.ncp_comfy — мини-клиент ComfyUI API (только stdlib).

Умеет: проверка сервера, загрузка картинки (upload), запуск workflow
(API-формат: ComfyUI -> Settings -> Dev mode -> Save (API Format)),
опрос истории, скачивание результата.
"""
import json
import os
import random
import urllib.error
import urllib.parse
import urllib.request


class ComfyError(RuntimeError):
    pass


def _norm_url(url):
    url = (url or "").strip()
    if not url:
        raise ComfyError("Не задан адрес ComfyUI (ComfyUI URL)")
    if not url.startswith("http"):
        url = "http://" + url
    return url.rstrip("/")


def _http(url, data=None, timeout=10, content_type=None):
    req = urllib.request.Request(url, data=data)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "ignore")[:400]
        except Exception:
            pass
        raise ComfyError("HTTP %s: %s" % (e.code, body)) from None
    except urllib.error.URLError as e:
        raise ComfyError("ComfyUI недоступен (%s)" % e.reason) from None


def _http_json(url, payload=None, timeout=10):
    if payload is None:
        return json.loads(_http(url, timeout=timeout).decode("utf-8"))
    data = json.dumps(payload).encode("utf-8")
    return json.loads(_http(url, data=data, timeout=timeout,
                            content_type="application/json").decode("utf-8"))


class ComfyClient:
    def __init__(self, base_url):
        self.base = _norm_url(base_url)

    def check(self):
        """Лёгкая проверка: сервер жив?"""
        try:
            _http_json(self.base + "/system_stats", timeout=4)
            return True
        except ComfyError:
            return False

    # ----------------------------------------------------------- upload

    def upload_image(self, png_path, name=None):
        """multipart POST /upload/image -> имя файла на сервере."""
        if not os.path.isfile(png_path):
            raise ComfyError("Картинка не найдена: %s" % png_path)
        name = name or ("ncp_" + os.path.basename(png_path))
        with open(png_path, "rb") as f:
            blob = f.read()
        boundary = "----NCPBoundary%s" % random.randint(10 ** 8, 10 ** 9)
        pre = ("--%s\r\n"
               "Content-Disposition: form-data; name=\"image\"; "
               "filename=\"%s\"\r\n"
               "Content-Type: image/png\r\n\r\n"
               % (boundary, name)).encode("utf-8")
        post = ("\r\n--%s--\r\n" % boundary).encode("utf-8")
        body = pre + blob + post
        resp = _http(self.base + "/upload/image", data=body, timeout=60,
                     content_type="multipart/form-data; boundary=" + boundary)
        info = json.loads(resp.decode("utf-8"))
        return info.get("name", name)

    # --------------------------------------------------------- workflow

    @staticmethod
    def load_workflow(path):
        if not path or not os.path.isfile(path):
            raise ComfyError("Workflow не задан (экспорт из ComfyUI: "
                             "Settings -> Dev mode -> Save (API Format))")
        with open(path, "r", encoding="utf-8") as f:
            wf = json.load(f)
        if not isinstance(wf, dict) or not wf:
            raise ComfyError("Workflow пустой или не API-формат")
        return wf

    @staticmethod
    def fill_workflow(wf, prompt=None, negative=None, seed=None,
                      image_name=None):
        """Заполнить типовые поля API-workflow:
        - CLIPTextEncode со строковым text: 1-й = позитив, 2-й = негатив
        - все int-поля seed/noise_seed = seed
        - LoadImage.image = image_name
        """
        text_nodes = []
        for nid, node in wf.items():
            if not isinstance(node, dict):
                continue
            cls = node.get("class_type", "")
            txt = node.get("inputs", {}).get("text")
            if "CLIPTextEncode" in cls and isinstance(txt, str):
                text_nodes.append((nid, node))
        text_nodes.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit()
                        else 10 ** 9)
        if prompt is not None and text_nodes:
            text_nodes[0][1]["inputs"]["text"] = prompt
        if negative is not None and len(text_nodes) > 1:
            text_nodes[1][1]["inputs"]["text"] = negative
        if seed is not None:
            for _, node in wf.items():
                ins = node.get("inputs", {})
                for k, v in list(ins.items()):
                    if k in ("seed", "noise_seed") and isinstance(v, int):
                        ins[k] = int(seed)
        if image_name is not None:
            loads = [(nid, node) for nid, node in wf.items()
                     if isinstance(node, dict)
                     and node.get("class_type") == "LoadImage"]
            if not loads:
                raise ComfyError("В workflow нет ноды LoadImage — "
                                 "нужна для img2img/эскиза")
            loads[0][1]["inputs"]["image"] = image_name
        return wf

    # ---------------------------------------------------------- submit

    def submit(self, wf, client_id="ncp"):
        data = {"prompt": wf, "client_id": client_id}
        resp = _http_json(self.base + "/prompt", payload=data, timeout=30)
        pid = resp.get("prompt_id")
        if not pid:
            raise ComfyError("ComfyUI отклонил workflow: %s" % resp)
        return pid

    # ------------------------------------------------------------ poll

    def poll(self, prompt_id, timeout_sec, t0):
        """('running', None) | ('done', [image_info]) | ('error', msg)"""
        import time
        try:
            hist = _http_json(self.base + "/history/" + prompt_id,
                              timeout=6)
        except ComfyError as e:
            # временный сетевой сбой — не роняем задачу сразу
            if time.time() - t0 > timeout_sec:
                return ("error", str(e))
            return ("running", None)
        entry = hist.get(prompt_id)
        if not entry:
            if time.time() - t0 > timeout_sec:
                return ("error", "таймаут ожидания ComfyUI (%s c)"
                        % timeout_sec)
            return ("running", None)
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            return ("error", "ошибка выполнения workflow")
        outputs = entry.get("outputs", {})
        images = []
        for node_out in outputs.values():
            for im in node_out.get("images", []):
                if im.get("type", "output") == "output":
                    images.append(im)
        if images:
            return ("done", images)
        if status.get("completed") or time.time() - t0 > timeout_sec:
            return ("error", "нет изображений в outputs (или таймаут)")
        return ("running", None)

    # ----------------------------------------------------------- fetch

    def fetch(self, image_info, save_dir):
        args = urllib.parse.urlencode({
            "filename": image_info.get("filename", ""),
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        })
        blob = _http(self.base + "/view?" + args, timeout=60)
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.basename(image_info.get("filename", "comfy.png"))
        stem, ext = os.path.splitext(base)
        path = os.path.join(save_dir, stem + "_ncp" + (ext or ".png"))
        i = 1
        while os.path.exists(path):
            path = os.path.join(save_dir,
                                "%s_ncp_%d%s" % (stem, i, ext or ".png"))
            i += 1
        with open(path, "wb") as f:
            f.write(blob)
        return path


def random_seed():
    return random.randint(0, 2 ** 31 - 1)
