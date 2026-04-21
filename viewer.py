"""
서버 에뮬레이터 화면 실시간 웹 뷰어 + 직접 조작 + NitroGen 토글.
실행: python viewer.py
브라우저에서 http://localhost:8080 접속

조작:
  클릭       → tap
  드래그      → swipe
  우클릭      → long press
  NitroGen 버튼 → 파이프라인 시작/중지
"""
import io, threading, json
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

import paramiko
from config import SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS

ADB          = "/home/sltrain/android-sdk/platform-tools/adb"
PYTHON       = "/home/sltrain/miniconda3/bin/python3"
PIPELINE     = "/home/sltrain/vla_pipeline/pipeline_ng_rl.py"
PIPELINE_LOG = "/home/sltrain/vla_pipeline/pipeline_ng_rl.log"
NITROGEN_DIR = "/home/sltrain/NitroGen"
NITROGEN_CKPT= "/home/sltrain/NitroGen/ng.pt"
NITROGEN_LOG = "/home/sltrain/vla_pipeline/nitrogen_serve.log"
NITROGEN_PORT= 5556
DEVICE       = "emulator-5554"
REFRESH_MS   = 200

# ── 세 개의 독립 SSH 연결 ─────────────────────────────────────
# screen : screencap 전용
# cmd    : ADB 명령 전용
# ctl    : NitroGen 파이프라인 제어 전용

_ssh_screen = None
_ssh_cmd    = None
_ssh_ctl    = None
_lock_screen = threading.Lock()
_lock_cmd    = threading.Lock()
_lock_ctl    = threading.Lock()


def _make_ssh() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS,
              timeout=10, banner_timeout=15)
    return c

def _get(var_name, lock):
    import builtins
    g = globals()
    try:
        c = g[var_name]
        if c and c.get_transport() and c.get_transport().is_active():
            return c
    except Exception:
        pass
    c = _make_ssh()
    g[var_name] = c
    return c

def _get_ssh_screen(): return _get("_ssh_screen", _lock_screen)
def _get_ssh_cmd():    return _get("_ssh_cmd",    _lock_cmd)
def _get_ssh_ctl():    return _get("_ssh_ctl",    _lock_ctl)


def grab_screen() -> bytes:
    with _lock_screen:
        ssh = _get_ssh_screen()
        _, out, _ = ssh.exec_command(f"{ADB} exec-out screencap -p", timeout=15)
        return out.read()


def run_adb(cmd: str) -> str:
    with _lock_cmd:
        ssh = _get_ssh_cmd()
        _, out, err = ssh.exec_command(f"{ADB} {cmd}", timeout=10)
        o = out.read().decode(errors="replace").strip()
        e = err.read().decode(errors="replace").strip()
        return o or e or "ok"


def run_ctl(cmd: str, timeout: int = 10) -> str:
    with _lock_ctl:
        ssh = _get_ssh_ctl()
        _, out, err = ssh.exec_command(cmd, timeout=timeout)
        o = out.read().decode(errors="replace").strip()
        e = err.read().decode(errors="replace").strip()
        return o or e or ""


# ── NitroGen 파이프라인 제어 ──────────────────────────────────

def nitrogen_status() -> dict:
    """NG-RL 파이프라인 + NitroGen 서버 상태 + 최근 로그 30줄"""
    pid_pipe   = run_ctl("pgrep -f pipeline_ng_rl.py 2>/dev/null | head -1")
    pid_server = run_ctl(f"pgrep -f 'serve.py.*ng.pt' 2>/dev/null | head -1")
    running    = bool(pid_pipe.strip())
    log        = run_ctl(f"tail -30 {PIPELINE_LOG} 2>/dev/null || echo '(로그 없음)'")
    return {
        "running":    running,
        "pid":        pid_pipe.strip(),
        "server_pid": pid_server.strip(),
        "log":        log,
    }


def _ensure_nitrogen_server() -> bool:
    """NitroGen 추론 서버가 없으면 시작. port 5556이 열릴 때까지 최대 30초 대기."""
    import time
    # 이미 실행 중이면 스킵
    pid = run_ctl(f"pgrep -f 'serve.py.*ng.pt' 2>/dev/null | head -1")
    if pid.strip():
        return True
    # 시작
    cmd = (
        f"cd {NITROGEN_DIR} && "
        f"CUDA_VISIBLE_DEVICES=2 "
        f"nohup {PYTHON} scripts/serve.py {NITROGEN_CKPT} --port {NITROGEN_PORT} "
        f"> {NITROGEN_LOG} 2>&1 &"
    )
    run_ctl(cmd)
    # 포트가 열릴 때까지 대기 (최대 30초)
    for _ in range(30):
        time.sleep(1)
        check = run_ctl(f"ss -tlnp 2>/dev/null | grep ':{NITROGEN_PORT}'")
        if check.strip():
            return True
    return False


def nitrogen_start() -> str:
    import time
    # 1. 기존 파이프라인 종료
    run_ctl("pkill -9 -f pipeline_ng_rl.py 2>/dev/null; pkill -9 -f pipeline_cnn.py 2>/dev/null; pkill -9 -f pipeline.py 2>/dev/null; sleep 0.5")
    # 2. NitroGen 추론 서버 확인/시작
    server_ok = _ensure_nitrogen_server()
    if not server_ok:
        return "NitroGen 서버 시작 실패 (타임아웃)"
    # 3. NG-RL 파이프라인 시작
    cmd = (
        f"cd /home/sltrain/vla_pipeline && "
        f"CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "
        f"nohup {PYTHON} -u {PIPELINE} --device {DEVICE} "
        f"--nitrogen-port {NITROGEN_PORT} --no-record "
        f"> {PIPELINE_LOG} 2>&1 &"
    )
    run_ctl(cmd)
    time.sleep(1.5)
    st = nitrogen_status()
    return f"NG-RL 시작됨 (PID {st['pid']})" if st["running"] else "시작 실패"


def nitrogen_stop() -> str:
    import time
    run_ctl("pkill -9 -f pipeline_ng_rl.py 2>/dev/null; pkill -9 -f pipeline_cnn.py 2>/dev/null; pkill -9 -f pipeline.py 2>/dev/null; pkill -f 'serve.py.*ng.pt' 2>/dev/null")
    time.sleep(0.5)
    st = nitrogen_status()
    return "중지됨" if not st["running"] else "중지 실패"


# ── HTML ────────────────────────────────────────────────────

HTML = f"""<!DOCTYPE html>
<html>
<head>
  <title>Emulator Viewer</title>
  <style>
    * {{ box-sizing: border-box; margin:0; padding:0 }}
    body {{ background:#111; color:#ccc; font-family:monospace;
            display:flex; flex-direction:column;
            height:100vh; padding:6px; gap:6px; overflow:hidden }}

    /* 상단 툴바 */
    #toolbar {{ display:flex; gap:8px; align-items:center; flex-shrink:0 }}
    #info {{ font-size:12px; color:#666; flex:1 }}
    #btn-ng {{ padding:6px 18px; border:none; border-radius:6px;
               font-size:13px; font-weight:bold; cursor:pointer;
               transition: background 0.2s }}
    #btn-ng.off {{ background:#1a6b1a; color:#8f8; }}
    #btn-ng.off:hover {{ background:#228822 }}
    #btn-ng.on  {{ background:#7a1a1a; color:#f88; }}
    #btn-ng.on:hover  {{ background:#992222 }}

    /* 메인: 좌(화면) + 우(로그) */
    #main {{ display:flex; gap:10px; flex:1; min-height:0 }}

    /* 왼쪽: 게임 화면 */
    #left {{ display:flex; flex-direction:column; align-items:center;
             gap:4px; flex-shrink:0 }}
    #s {{ height:100%; max-height:calc(100vh - 60px);
          border:1px solid #333; border-radius:4px;
          cursor:crosshair; user-select:none; display:block;
          object-fit:contain }}
    #msg {{ font-size:12px; color:#6f6; min-height:16px }}

    /* 오른쪽: 로그 패널 */
    #right {{ flex:1; display:flex; flex-direction:column; gap:6px; min-width:0 }}
    #log-title {{ font-size:14px; color:#aaa; font-weight:bold;
                  padding:4px 0; border-bottom:1px solid #333; flex-shrink:0 }}
    #logbox {{ flex:1; background:#0a0a0a; border:1px solid #2a2a2a;
               border-radius:6px; padding:10px;
               font-size:13px; line-height:1.6; color:#9f9;
               white-space:pre-wrap; overflow-y:auto; min-height:0 }}

    /* 로그 색상 강조 */
    #logbox {{ color:#8c8 }}
  </style>
</head>
<body>
  <div id="toolbar">
    <div id="info">클릭: tap &nbsp;|&nbsp; 드래그: swipe &nbsp;|&nbsp; 우클릭: long press</div>
    <button id="btn-ng" class="off" onclick="toggleNitrogen()">▶ NitroGen 시작</button>
  </div>

  <div id="main">
    <div id="left">
      <img id="s" src="/screen" draggable="false">
      <div id="msg">—</div>
    </div>
    <div id="right">
      <div id="log-title">📋 파이프라인 로그 (실시간)</div>
      <div id="logbox">(로딩 중...)</div>
    </div>
  </div>

  <script>
    var img  = document.getElementById('s');
    var msg  = document.getElementById('msg');
    var btn  = document.getElementById('btn-ng');
    var logb = document.getElementById('logbox');
    var drag = null;
    var ngRunning = false;

    // ── 터치 조작 ──────────────────────────────────────
    function pos(e) {{
      var r = img.getBoundingClientRect();
      return {{ x: (e.clientX - r.left) / r.width,
               y: (e.clientY - r.top)  / r.height }};
    }}
    function send(url) {{
      fetch(url).then(r => r.text()).then(t => {{ msg.innerText = t; }});
    }}

    img.addEventListener('mousedown', function(e) {{
      e.preventDefault();
      if (e.button === 2) return;
      drag = pos(e);
    }});
    img.addEventListener('mouseup', function(e) {{
      e.preventDefault();
      var p = pos(e);
      if (e.button === 2) {{ send('/longpress?x='+p.x+'&y='+p.y); return; }}
      if (!drag) return;
      var dx = Math.abs(p.x-drag.x), dy = Math.abs(p.y-drag.y);
      if (dx < 0.02 && dy < 0.02)
        send('/tap?x='+p.x+'&y='+p.y);
      else
        send('/swipe?x1='+drag.x+'&y1='+drag.y+'&x2='+p.x+'&y2='+p.y);
      drag = null;
    }});
    img.addEventListener('contextmenu', e => e.preventDefault());

    // ── 화면 갱신 ──────────────────────────────────────
    function refresh() {{
      var t = Date.now();
      img.src = '/screen?' + t;
      img.onload  = () => setTimeout(refresh, {REFRESH_MS});
      img.onerror = () => setTimeout(refresh, 1000);
    }}
    refresh();

    // ── NitroGen 제어 ──────────────────────────────────
    function toggleNitrogen() {{
      btn.disabled = true;
      btn.innerText = '...'
      var url = ngRunning ? '/nitrogen/stop' : '/nitrogen/start';
      fetch(url).then(r => r.json()).then(d => {{
        updateNgUI(d);
        btn.disabled = false;
      }}).catch(() => {{ btn.disabled = false; }});
    }}

    function updateNgUI(d) {{
      ngRunning = d.running;
      if (ngRunning) {{
        btn.className = 'on';
        btn.innerText = '⏹ NitroGen 중지';
      }} else {{
        btn.className = 'off';
        btn.innerText = '▶ NitroGen 시작';
      }}
      if (d.log) {{
        logb.innerText = d.log;
        logb.scrollTop = logb.scrollHeight;
      }}
    }}

    // 2초마다 상태 + 로그 폴링
    function pollStatus() {{
      fetch('/nitrogen/status').then(r => r.json()).then(d => {{
        updateNgUI(d);
      }}).catch(() => {{}}).finally(() => setTimeout(pollStatus, 2000));
    }}
    pollStatus();
  </script>
</body>
</html>
"""


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path)
        q = parse_qs(p.query)

        # ── 화면 캡처 ──────────────────────────────────
        if p.path == "/screen":
            try:
                png = grab_screen()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", len(png))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(png)
            except Exception as e:
                self.send_error(503, str(e))

        # ── ADB 터치 ────────────────────────────────────
        elif p.path == "/tap":
            x, y = float(q["x"][0]), float(q["y"][0])
            px, py = int(x * 1080), int(y * 2280)
            self._text(f"TAP ({px},{py})  {run_adb(f'shell input tap {px} {py}')}")

        elif p.path == "/swipe":
            x1,y1 = float(q["x1"][0]), float(q["y1"][0])
            x2,y2 = float(q["x2"][0]), float(q["y2"][0])
            p1x,p1y = int(x1*1080), int(y1*2280)
            p2x,p2y = int(x2*1080), int(y2*2280)
            self._text(f"SWIPE ({p1x},{p1y})→({p2x},{p2y})  "
                       f"{run_adb(f'shell input swipe {p1x} {p1y} {p2x} {p2y} 300')}")

        elif p.path == "/longpress":
            x, y = float(q["x"][0]), float(q["y"][0])
            px, py = int(x * 1080), int(y * 2280)
            self._text(f"LONG PRESS ({px},{py})  "
                       f"{run_adb(f'shell input swipe {px} {py} {px} {py} 800')}")

        elif p.path == "/keyevent":
            code = q.get("code", ["BACK"])[0]
            self._text(f"KEYEVENT {code}  {run_adb(f'shell input keyevent {code}')}")

        # ── NitroGen 제어 (JSON 응답) ───────────────────
        elif p.path == "/nitrogen/status":
            self._json(nitrogen_status())

        elif p.path == "/nitrogen/start":
            msg = nitrogen_start()
            st  = nitrogen_status()
            st["message"] = msg
            self._json(st)

        elif p.path == "/nitrogen/stop":
            msg = nitrogen_stop()
            st  = nitrogen_status()
            st["message"] = msg
            self._json(st)

        # ── 메인 페이지 ─────────────────────────────────
        else:
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

    def _text(self, msg: str):
        body = msg.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print("Connecting to server...")
    _get_ssh_screen()
    _get_ssh_cmd()
    _get_ssh_ctl()
    print("Connected (3 SSH channels: screen / adb / control).")
    print("Open http://localhost:8080 in your browser")
    print("  Click        = tap")
    print("  Drag         = swipe")
    print("  Right-click  = long press")
    print("  ▶ 버튼       = NitroGen 시작/중지")
    ThreadingHTTPServer(("", 8080), Handler).serve_forever()
