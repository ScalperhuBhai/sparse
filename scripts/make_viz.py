#!/usr/bin/env python3
"""Pack a recording and pipeline output into a self-contained HTML viewer."""
from __future__ import annotations

import argparse, json, os, sys
from html import escape
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orbitsight import evio


TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>SPARSE — __TITLE__</title>
<style>
:root{--bg:#07090d;--panel:#11151d;--line:#212836;--tx:#e6ebf2;--dim:#8b97a8;
      --pos:#4ea6ff;--neg:#ff7a59;--det:#ffd23f;--trk:#2ee6a8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
     font:13px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:12px 18px;border-bottom:1px solid var(--line);display:flex;
       align-items:baseline;gap:14px;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:650;letter-spacing:-.01em}
header .meta{color:var(--dim);font-size:12px}
main{display:grid;grid-template-columns:1fr 260px;gap:16px;padding:16px;align-items:start}
@media(max-width:880px){main{grid-template-columns:1fr}}
.stage{background:#000;border:1px solid var(--line);border-radius:8px;overflow:hidden;
       position:relative}
canvas{display:block;width:100%;height:auto;image-rendering:pixelated}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
.panel h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);
          margin:0 0 10px;font-weight:600}
.kv{display:flex;justify-content:space-between;gap:10px;padding:3px 0;
    border-bottom:1px solid #1a2030;font-variant-numeric:tabular-nums}
.kv:last-child{border-bottom:0}
.kv span:first-child{color:var(--dim)}
.controls{display:flex;gap:8px;align-items:center;padding:10px 14px;
          border-top:1px solid var(--line);background:var(--panel)}
button{background:#1b2331;color:var(--tx);border:1px solid var(--line);border-radius:6px;
       padding:5px 12px;cursor:pointer;font:inherit}
button:hover{background:#243044}
input[type=range]{flex:1;accent-color:var(--trk)}
.legend{display:flex;gap:14px;font-size:11px;color:var(--dim);padding:8px 18px;flex-wrap:wrap}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;
          vertical-align:-1px}
table.tk{width:100%;border-collapse:collapse;font-size:11.5px;
         font-variant-numeric:tabular-nums}
table.tk th{color:var(--dim);text-align:right;font-weight:600;padding:3px 4px;
            border-bottom:1px solid #1a2030}
table.tk th:first-child{text-align:left}
table.tk td{text-align:right;padding:3px 4px;border-bottom:1px solid #141a26}
table.tk td:first-child{text-align:left;color:var(--trk)}
table.tk tr.live td{background:rgba(46,230,168,.08)}
.rate-note{color:var(--dim);font-size:10.5px;margin-top:8px;line-height:1.45}
</style></head><body>
<header><h1>SPARSE — event-camera tracking</h1>
<span class="meta">__META__</span></header>
<div class="legend">
  <span><i style="background:var(--pos)"></i>ON event</span>
  <span><i style="background:var(--neg)"></i>OFF event</span>
  <span><i style="background:var(--det)"></i>detection</span>
  <span><i style="background:var(--trk)"></i>confirmed track</span>
</div>
<main>
  <div>
    <div class="stage"><canvas id="cv"></canvas></div>
    <div class="controls">
      <button id="play">Pause</button>
      <input type="range" id="scrub" min="0" value="0" step="1">
      <button id="trail">Trails: on</button>
    </div>
  </div>
  <div>
    <div class="panel" style="margin-bottom:12px">
      <h2>Live</h2>
      <div class="kv"><span>t</span><b id="mt">—</b></div>
      <div class="kv"><span>events in</span><b id="mi">—</b></div>
      <div class="kv"><span>after filter</span><b id="mk">—</b></div>
      <div class="kv"><span>detections</span><b id="md">—</b></div>
      <div class="kv"><span>confirmed tracks</span><b id="mr">—</b></div>
    </div>
    <div class="panel" style="margin-bottom:12px"><h2>Run</h2><div id="stats"></div></div>
    <div class="panel"><h2>Tracks</h2><div id="tracks"></div></div>
  </div>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const W = D.width, H = D.height, S = Math.max(1, Math.round(900/W));
const cv = document.getElementById('cv'); cv.width = W*S; cv.height = H*S;
const g = cv.getContext('2d');
const scrub = document.getElementById('scrub'); scrub.max = D.windows.length-1;
let i = 0, playing = true, trails = true;
const trackPaths = new Map();

function stat(k,v){return `<div class="kv"><span>${k}</span><b>${v}</b></div>`}
document.getElementById('stats').innerHTML =
  Object.entries(D.stats).map(([k,v])=>stat(k,v)).join('');

// ── per-track summary ──────────────────────────────────────────────────────
// Angular rate is the number an SSA operator reads, but it needs a plate scale
// and EBSSA does not publish one per recording. So px/s is always shown, and
// arcsec/s appears only when --plate-scale was supplied. Inventing a plate
// scale to make the column look complete would make every figure in it wrong.
const RATE = D.arcsec_per_px;
function buildTracks(){
  const t = D.tracks;
  if(!t || !t.length){
    document.getElementById('tracks').innerHTML =
      '<div class="rate-note">No track was confirmed in this run.</div>';
    return;
  }
  const head = RATE
    ? '<tr><th>track</th><th>first</th><th>held</th><th>px/s</th><th>&Prime;/s</th></tr>'
    : '<tr><th>track</th><th>first</th><th>held</th><th>px/s</th></tr>';
  const rows = t.map(r =>
    `<tr data-id="${r.id}"><td>T${r.id}</td><td>${(r.t0/1e6).toFixed(1)}s</td>` +
    `<td>${r.span_s.toFixed(1)}s</td><td>${r.px_s.toFixed(1)}</td>` +
    (RATE ? `<td>${(r.px_s*RATE).toFixed(1)}</td>` : '') + '</tr>').join('');
  document.getElementById('tracks').innerHTML =
    `<table class="tk">${head}${rows}</table>` +
    (RATE
      ? `<div class="rate-note">Angular rate at ${RATE}&Prime;/px.</div>`
      : `<div class="rate-note">Rates are in pixels per second. Angular rate
         needs a plate scale, which this recording does not carry &mdash; pass
         <code>--plate-scale</code> in arcsec/px to add a &Prime;/s column.</div>`);
}
buildTracks();

function draw(){
  const w = D.windows[i];
  g.fillStyle = trails ? 'rgba(0,0,0,0.22)' : '#000';
  g.fillRect(0,0,cv.width,cv.height);
  const ev = D.events[i] || [];
  for (let j=0;j<ev.length;j+=3){
    g.fillStyle = ev[j+2] > 0 ? '#4ea6ff' : '#ff7a59';
    g.fillRect(ev[j]*S, ev[j+1]*S, S, S);
  }
  g.lineWidth = 1.5;
  for (const d of w.d){
    g.strokeStyle = '#ffd23f';
    const r = 11*S;
    g.strokeRect(d[0]*S-r/2, d[1]*S-r/2, r, r);
  }
  for (const t of w.r){
    const id = t[0];
    if(!trackPaths.has(id)) trackPaths.set(id, []);
    const p = trackPaths.get(id); p.push([t[1]*S, t[2]*S]);
    if(p.length>240) p.shift();
    g.strokeStyle = '#2ee6a8'; g.beginPath();
    p.forEach((q,k)=> k? g.lineTo(q[0],q[1]) : g.moveTo(q[0],q[1]));
    g.stroke();
    g.beginPath(); g.arc(t[1]*S, t[2]*S, 5*S, 0, 7); g.stroke();
    g.fillStyle='#2ee6a8'; g.font=`${11*S}px ui-monospace,monospace`;
    g.fillText('T'+id, t[1]*S+7*S, t[2]*S-7*S);
  }
  document.getElementById('mt').textContent = (w.t/1000).toFixed(1)+' ms';
  document.getElementById('mi').textContent = w.n.toLocaleString();
  document.getElementById('mk').textContent = w.k.toLocaleString();
  document.getElementById('md').textContent = w.d.length;
  document.getElementById('mr').textContent = w.r.length;
  const live = new Set(w.r.map(t=>t[0]));
  document.querySelectorAll('table.tk tr[data-id]').forEach(tr =>
    tr.classList.toggle('live', live.has(+tr.dataset.id)));
  scrub.value = i;
}
function tick(){ if(playing){ i=(i+1)%D.windows.length; if(i===0) trackPaths.clear(); draw(); } }
setInterval(tick, 40);
scrub.oninput = e => { playing=false; document.getElementById('play').textContent='Play';
                       i=+e.target.value; trackPaths.clear(); draw(); };
document.getElementById('play').onclick = e => { playing=!playing;
                       e.target.textContent = playing?'Pause':'Play'; };
document.getElementById('trail').onclick = e => { trails=!trails;
                       e.target.textContent = 'Trails: '+(trails?'on':'off'); };
draw();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="event recording")
    ap.add_argument("--results", required=True, help="results.jsonl from `cli.py run`")
    ap.add_argument("--latency", default=None, help="latency.json")
    ap.add_argument("--metrics", default=None, help="metrics_tracks.json")
    ap.add_argument("--out", default="out/viewer.html")
    ap.add_argument("--max-events-per-window", type=int, default=1500)
    ap.add_argument("--max-windows", type=int, default=600)
    ap.add_argument("--title", default="run")
    ap.add_argument("--plate-scale", type=float, default=None,
                    help="arcsec per pixel. Supply it and the track table gains "
                         "an arcsec/s column; omit it and rates stay in px/s. "
                         "There is no sane default — it is a property of the "
                         "telescope and the sensor, and guessing one would make "
                         "every angular rate in the table wrong.")
    ap.add_argument("--window-stride", type=int, default=1,
                    help="keep every Nth window; a 198 s recording is 19,820 "
                         "windows and packing them all makes a file no browser "
                         "will open")
    a = ap.parse_args()

    # EBSSA `.mat` carries its own sensor geometry, and its ground truth is on a
    # clock that only matches the events if both are rebased together — so load
    # it the same way `cli.py` does rather than through the generic reader,
    # which would infer the shape from whichever pixels happened to fire.
    gt = None
    if a.input.lower().endswith(".mat"):
        ev, gt, shape = evio.load_ebssa(a.input)
        H, W = shape
    else:
        ev = evio.load(a.input)
        H, W = evio.sensor_shape(ev)

    windows = [json.loads(l) for l in open(a.results)]
    all_windows = windows                      # census uses every window
    if a.window_stride > 1:
        windows = windows[::a.window_stride]
    windows = windows[:a.max_windows]
    if not windows:
        ap.error("no result windows selected")

    edges = np.array([w["t"] for w in windows], dtype=np.int64)
    idx = np.searchsorted(ev["t"], edges)
    packed_ev, packed_w = [], []
    rng = np.random.default_rng(0)
    start = 0
    for k, w in enumerate(windows):
        chunk = ev[start:idx[k]]
        start = idx[k]
        if len(chunk) > a.max_events_per_window:
            chunk = chunk[rng.choice(len(chunk), a.max_events_per_window, replace=False)]
        flat = np.empty(len(chunk) * 3, dtype=np.int32)
        flat[0::3] = chunk["x"]; flat[1::3] = chunk["y"]; flat[2::3] = chunk["p"]
        packed_ev.append(flat.tolist())
        packed_w.append(dict(
            t=w["t"], n=w["n_in"], k=w["n_kept"],
            d=[[round(d["x"], 2), round(d["y"], 2)] for d in w["detections"]],
            r=[[t["track_id"], round(t["x"], 2), round(t["y"], 2)] for t in w["tracks"]],
        ))

    stats = {}
    if a.latency and os.path.exists(a.latency):
        L = json.load(open(a.latency))
        stats["mean latency"] = f"{L['total']['mean_ms']:.2f} ms"
        stats["p95 latency"] = f"{L['total']['p95_ms']:.2f} ms"
        stats["p99 latency"] = f"{L['total']['p99_ms']:.2f} ms"
        stats["budget 40 ms"] = f"{(1-L['summary']['frac_windows_over_budget'])*100:.1f}% met"
        stats["throughput"] = f"{L['summary']['throughput_ev_per_s']/1e6:.2f} M ev/s"
        stats["real-time factor"] = f"{L['summary']['realtime_factor']:.1f}x"
    if a.metrics and os.path.exists(a.metrics):
        M = json.load(open(a.metrics))
        stats["precision"] = f"{M['precision']:.3f}"
        stats["recall"] = f"{M['recall']:.3f}"
        stats["F1"] = f"{M['f1']:.3f}"
        if M.get("mean_localisation_error_px"):
            stats["localisation"] = f"{M['mean_localisation_error_px']:.2f} px"

    # ── per-track summary ────────────────────────────────────────────────
    # Built from the FULL results file, not the strided/truncated view that is
    # packed for playback: a track that lived outside the packed range is still
    # a track the run produced, and dropping it would understate the census.
    tracks = {}
    for w in all_windows:
        for tr in w["tracks"]:
            k = tr["track_id"]
            r = tracks.setdefault(k, dict(id=k, t0=w["t"], t1=w["t"], pts=[]))
            r["t1"] = w["t"]
            r["pts"].append((w["t"], tr["x"], tr["y"]))
    track_rows = []
    for r in tracks.values():
        pts = r["pts"]
        span = (r["t1"] - r["t0"]) / 1e6
        if len(pts) > 1 and span > 0:
            # median instantaneous rate: robust to the one bad frame that a
            # mean over a coasting track would otherwise be dominated by
            rates = []
            for (ta, xa, ya), (tb, xb, yb) in zip(pts, pts[1:]):
                dt = (tb - ta) / 1e6
                if dt > 0:
                    rates.append(float(np.hypot(xb - xa, yb - ya) / dt))
            px_s = float(np.median(rates)) if rates else 0.0
        else:
            px_s = 0.0
        track_rows.append(dict(id=int(r["id"]), t0=int(r["t0"]),
                               span_s=round(span, 2), px_s=round(px_s, 2),
                               samples=len(pts)))
    track_rows.sort(key=lambda r: -r["span_s"])

    payload = dict(width=int(W), height=int(H), windows=packed_w, events=packed_ev,
                   stats=stats, tracks=track_rows[:40],
                   arcsec_per_px=a.plate_scale)
    meta = f"{len(ev):,} events · {W}×{H} sensor · {len(packed_w)} windows"
    payload_json = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    html = (TEMPLATE.replace("__DATA__", payload_json)
                    .replace("__META__", meta).replace("__TITLE__", escape(a.title)))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
