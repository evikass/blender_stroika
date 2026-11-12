"""Updates through Blender's own extension system.

When the add-on is installed as an extension from a remote repository
(Superhive, once the buyer has connected it to Blender), Blender keeps a copy
of that repository's list of extensions. This reads our entry from it, asks
the repository again on request, and installs the newer version with
Blender's own installer. Installed the old way (a legacy add-on, or an
extension dropped in from a zip file), there is no repository to ask: the
card then explains how to connect Superhive.
"""

import json
import os
import time

import bpy

PKG_ID = "image_to_3d"
GUIDE_URL = ("https://support.superhivemarket.com/article/"
             "335-connecting-superhive-as-a-remote-repository-in-blender")
PRODUCT_URL = "https://superhivemarket.com/products/imageto3d"

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
_STATE = {"checked": 0.0, "msg": "", "msg_t": 0.0}
# kept in Blender's own namespace: it outlives the add-on reloading itself
# after an update, so the start-up check runs once per Blender session
_SESSION = "_img3d_auto_update"


def _vt(v):
    """'1.2.10' -> (1, 2, 10) for comparing."""
    out = []
    for p in str(v or "").split("."):
        num = "".join(ch for ch in p if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def installed_version():
    """From the manifest when there is one, else from bl_info."""
    man = os.path.join(ADDON_DIR, "blender_manifest.toml")
    try:
        for line in open(man, encoding="utf-8"):
            s = line.strip()
            if s.startswith("version") and "=" in s:
                return s.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    import sys
    mod = sys.modules.get(__package__)
    v = getattr(mod, "bl_info", {}).get("version", (0, 0, 0))
    return ".".join(str(x) for x in v)


def own_repo():
    """(index, repo) of the repository this add-on came from, or (None, None)
    when it is a legacy add-on or an extension from a local zip file."""
    pkg = __package__ or ""
    if not pkg.startswith("bl_ext."):
        return None, None
    module = pkg.split(".")[1]
    for i, r in enumerate(bpy.context.preferences.extensions.repos):
        if r.module == module:
            if not r.use_remote_url:
                return None, r
            return i, r
    return None, None


def _remote_entry(repo):
    """Our entry in Blender's cached copy of the repository's list."""
    path = os.path.join(repo.directory, ".blender_ext", "index.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    for e in data.get("data", []):
        if e.get("id") == PKG_ID:
            return e
    return None


def status():
    """What the card shows: mode ('legacy' / 'repo'), installed, latest,
    newer, note."""
    out = {"mode": "legacy", "installed": installed_version(), "latest": "",
           "newer": False, "note": ""}
    i, repo = own_repo()
    if i is None:
        return out
    out["mode"] = "repo"
    e = _remote_entry(repo)
    if e is not None:
        out["latest"] = str(e.get("version", ""))
        out["newer"] = _vt(out["latest"]) > _vt(out["installed"])
    if _STATE["msg"] and time.time() - _STATE["msg_t"] < 15.0:
        out["note"] = _STATE["msg"]
    done = bpy.app.driver_namespace.get(_SESSION) or {}
    if done.get("updated_to") == out["installed"] and \
            time.time() - done.get("t", 0.0) < 60.0:
        out["note"] = "Updated to v%s" % out["installed"]
    return out


def _prefs():
    import sys
    try:
        return bpy.context.preferences.addons[__package__].preferences
    except Exception:
        return None


def auto_update_due():
    """Once per Blender session, a few seconds after start: with Auto Update
    on, ask Superhive and install a newer version by itself."""
    ns = bpy.app.driver_namespace
    if ns.get(_SESSION, {}).get("checked"):
        return None
    p = _prefs()
    if p is None or not getattr(p, "auto_update", False):
        return None
    i, repo = own_repo()
    if i is None or not _online():
        return None
    ns[_SESSION] = {"checked": True, "t": time.time()}
    try:
        bpy.ops.extensions.repo_sync(repo_directory=repo.directory,
                                     repo_index=i)
    except Exception as exc:
        print("[Image to 3D] auto update check:", exc)
        return None
    bpy.app.timers.register(_auto_install, first_interval=3.0)
    return None


def _auto_install(tries=[0]):
    import sys
    s = status()
    if not s["latest"] and tries[0] < 20:
        tries[0] += 1
        return 1.0
    tries[0] = 0
    if not s["newer"]:
        return None
    mod = sys.modules.get(__package__)
    if getattr(mod, "CARD_JOB", None) is not None or \
            getattr(mod, "SETUP_JOB", None) is not None:
        return 30.0          # a model is being made: wait for it
    i, repo = own_repo()
    if i is None:
        return None
    try:
        mod.server_stop()
    except Exception:
        pass
    ns = bpy.app.driver_namespace
    ns[_SESSION] = {"checked": True, "updated_to": s["latest"],
                    "t": time.time()}
    try:
        bpy.ops.extensions.package_install(
            repo_directory=repo.directory, repo_index=i, pkg_id=PKG_ID,
            enable_on_install=True)
        print("[Image to 3D] updated to", s["latest"])
    except Exception as exc:
        print("[Image to 3D] auto update:", exc)
    return None


def _say(msg):
    _STATE["msg"] = msg
    _STATE["msg_t"] = time.time()


def _redraw():
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type in ("VIEW_3D", "PREFERENCES"):
                a.tag_redraw()


def _online():
    return bool(getattr(bpy.app, "online_access", True))


class IMG3D_OT_update_check(bpy.types.Operator):
    """Ask Superhive whether a newer version of Image to 3D is out"""
    bl_idname = "img3d.update_check"
    bl_label = "Check for Updates"

    def execute(self, context):
        i, repo = own_repo()
        if i is None:
            _say("Connect Superhive to Blender for updates")
            _redraw()
            return {"FINISHED"}
        if not _online():
            _say("Turn on Allow Online Access in Preferences > System")
            _redraw()
            return {"FINISHED"}
        try:
            bpy.ops.extensions.repo_sync(repo_directory=repo.directory,
                                         repo_index=i)
        except Exception as exc:
            _say("Could not reach Superhive: %s" % str(exc)[:60])
            _redraw()
            return {"FINISHED"}
        _STATE["checked"] = time.time()
        # the list is refreshed in the background: look again shortly
        bpy.app.timers.register(_after_check, first_interval=2.0)
        _say("Checking...")
        _redraw()
        return {"FINISHED"}


def _after_check(tries=[0]):
    s = status()
    if s["latest"] or tries[0] > 10:
        tries[0] = 0
        if s["newer"]:
            _say("Version %s is available" % s["latest"])
        elif s["latest"]:
            _say("You have the latest version")
        else:
            _say("Image to 3D is not listed in this repository")
        _redraw()
        return None
    tries[0] += 1
    return 1.0


class IMG3D_OT_update_install(bpy.types.Operator):
    """Download and install the newest version of Image to 3D from
    Superhive. Blender reloads the add-on when it is done"""
    bl_idname = "img3d.update_install"
    bl_label = "Download Update"

    def execute(self, context):
        i, repo = own_repo()
        if i is None:
            return {"CANCELLED"}
        if not _online():
            _say("Turn on Allow Online Access in Preferences > System")
            _redraw()
            return {"FINISHED"}
        import sys
        mod = sys.modules.get(__package__)
        try:
            mod.server_stop()
        except Exception:
            pass
        try:
            bpy.ops.extensions.package_install(
                repo_directory=repo.directory, repo_index=i, pkg_id=PKG_ID,
                enable_on_install=True)
        except Exception as exc:
            _say("Update failed: %s" % str(exc)[:60])
            _redraw()
            return {"FINISHED"}
        _say("Updating...")
        _redraw()
        return {"FINISHED"}


class IMG3D_OT_update_connect(bpy.types.Operator):
    """How to connect Superhive to Blender, for one-click updates"""
    bl_idname = "img3d.update_connect"
    bl_label = "Connect Superhive"

    def execute(self, context):
        bpy.ops.wm.url_open(url=GUIDE_URL)
        return {"FINISHED"}


class IMG3D_OT_update_notes(bpy.types.Operator):
    """See what is new in this version on the product page"""
    bl_idname = "img3d.update_notes"
    bl_label = "What's New"

    def execute(self, context):
        i, repo = own_repo()
        url = PRODUCT_URL
        if repo is not None:
            e = _remote_entry(repo)
            if e and e.get("website"):
                url = e["website"]
        bpy.ops.wm.url_open(url=url)
        return {"FINISHED"}


CLASSES = (IMG3D_OT_update_check, IMG3D_OT_update_install,
           IMG3D_OT_update_connect, IMG3D_OT_update_notes)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
