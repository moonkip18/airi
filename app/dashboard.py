"""Airi dashboard — local, read/write window into her persistent state.

Four panes, one page, zero build step:

  * MEMORY   every memory with search/filter, inline edit, delete.
             Deleting here means she forgets: the row is removed, not
             soft-deleted (the contradiction system's superseded_by is
             for organic forgetting; this is the human override).
  * PEOPLE   per-person relationship meters + interaction counts.
  * METRICS  behavior over time: messages/day (her vs the room),
             action mix, language mix, current emotional state.

Design contract: STYLE-GUIDE.md (near-black, hairlines, one red accent,
Fraunces/Inter/IBM Plex Mono). Binds to 127.0.0.1 only — this exposes
her memory contents and must never be public.

Run:  python -m app.dashboard        (needs: pip install fastapi uvicorn)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.database import (  # noqa: E402
    EmotionalState, LanguageStats, Memory, MemoryType, Message,
    Relationship, TrainingSample, User, get_session,
)

try:
    from fastapi import Body, FastAPI, Query
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:  # pragma: no cover
    print("dashboard needs fastapi + uvicorn:  pip install fastapi uvicorn")
    sys.exit(1)

from sqlalchemy import delete, func, select  # noqa: E402

app = FastAPI(title="airi dashboard", docs_url=None, redoc_url=None)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>airi — dashboard</title>
<style>
:root {
  --bg: #050506; --bg-elevated: #0a0b0d;
  --ink: #e9eaec; --ink-dim: #8b8f97; --ink-faint: #565a61;
  --line: rgba(255,255,255,0.08); --synapse: #dfe3e8;
  --synapse-dim: rgba(223,227,232,0.3); --red: #c8433c; --red-dim: rgba(200,67,59,0.4);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--ink); font: 400 14px/1.55 'Inter', sans-serif; }
.mono { font-family: 'IBM Plex Mono', monospace; }
header { border-bottom: 1px solid var(--line); padding: 20px 28px; display: flex; align-items: baseline; gap: 18px; }
h1 { font: 300 26px 'Fraunces', serif; letter-spacing: .5px; }
header .sub { color: var(--ink-faint); font-size: 12px; }
nav { display: flex; border-bottom: 1px solid var(--line); padding: 0 28px; background: var(--bg-elevated); }
nav button { background: none; border: none; border-bottom: 2px solid transparent; color: var(--ink-dim); font: 500 12px 'Inter'; letter-spacing: .12em; text-transform: uppercase; padding: 14px 18px; cursor: pointer; }
nav button.on { color: var(--ink); border-bottom-color: var(--red); }
main { padding: 24px 28px; max-width: 1200px; }
.pane { display: none; } .pane.on { display: block; }
.toolbar { display: flex; gap: 10px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
input[type=search], select { background: var(--bg-elevated); border: 1px solid var(--line); color: var(--ink); font: 400 13px 'Inter'; padding: 8px 12px; border-radius: 4px; outline: none; }
input[type=search]:focus, select:focus { border-color: var(--synapse-dim); }
.chip { font: 500 10px 'IBM Plex Mono'; letter-spacing: .08em; text-transform: uppercase; border: 1px solid var(--line); color: var(--ink-dim); padding: 3px 8px; border-radius: 3px; }
.mem { border: 1px solid var(--line); border-radius: 6px; background: var(--bg-elevated); padding: 14px 16px; margin-bottom: 10px; }
.mem .row1 { display: flex; gap: 10px; align-items: baseline; margin-bottom: 6px; }
.mem .when { color: var(--ink-faint); font: 400 11px 'IBM Plex Mono'; margin-left: auto; }
.mem .content { white-space: pre-wrap; }
.mem .meta { color: var(--ink-faint); font-size: 11px; margin-top: 8px; display: flex; gap: 14px; flex-wrap: wrap; }
.mem textarea { width: 100%; background: var(--bg); border: 1px solid var(--synapse-dim); color: var(--ink); font: 400 14px/1.5 'Inter'; padding: 10px; border-radius: 4px; min-height: 70px; resize: vertical; outline: none; }
.mem .actions { display: flex; gap: 8px; margin-top: 10px; }
button.btn { background: none; border: 1px solid var(--line); color: var(--ink-dim); font: 500 12px 'Inter'; padding: 6px 14px; border-radius: 4px; cursor: pointer; }
button.btn:hover { border-color: var(--synapse-dim); color: var(--ink); }
button.btn.danger { border-color: var(--red-dim); color: var(--red); }
button.btn.danger:hover { background: var(--red-dim); color: var(--ink); }
.statgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
.stat { border: 1px solid var(--line); border-radius: 6px; background: var(--bg-elevated); padding: 16px; }
.stat .k { color: var(--ink-faint); font: 500 10px 'Inter'; text-transform: uppercase; letter-spacing: .12em; }
.stat .v { font: 300 30px 'Fraunces', serif; margin-top: 6px; }
h2.sect { font: 300 15px 'Fraunces', serif; margin: 26px 0 10px; color: var(--ink-dim); }
.barrow { display: grid; grid-template-columns: 110px 1fr 48px; gap: 10px; align-items: center; margin-bottom: 6px; font: 400 12px 'IBM Plex Mono'; color: var(--ink-dim); }
.bar { height: 6px; background: var(--bg); border: 1px solid var(--line); border-radius: 3px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--synapse-dim); }
.barrow.hot .bar i { background: var(--red-dim); }
.spark { display: flex; align-items: flex-end; gap: 2px; height: 72px; }
.spark i { flex: 1; background: var(--synapse-dim); min-height: 1px; }
.spark i.hot { background: var(--red-dim); }
.xlabels { display: flex; justify-content: space-between; color: var(--ink-faint); font: 400 10px 'IBM Plex Mono'; margin-top: 4px; }
.person { border: 1px solid var(--line); border-radius: 6px; background: var(--bg-elevated); padding: 14px 16px; margin-bottom: 10px; display: grid; grid-template-columns: 180px 1fr auto; gap: 16px; align-items: center; }
.person .name { font-weight: 500; }
.person .u { display: block; color: var(--ink-faint); font: 400 11px 'IBM Plex Mono'; margin-top: 2px; }
.meter { display: grid; grid-template-columns: 86px 1fr 40px; gap: 8px; align-items: center; font: 400 10px 'IBM Plex Mono'; color: var(--ink-faint); margin-bottom: 3px; }
.meter .bar i { background: var(--synapse-dim); }
.meter.neg .bar i { background: var(--red-dim); }
.empty { color: var(--ink-faint); padding: 40px 0; text-align: center; font-size: 13px; }
.toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); background: var(--bg-elevated); border: 1px solid var(--line); color: var(--ink); padding: 10px 20px; border-radius: 6px; font-size: 13px; opacity: 0; transition: opacity .2s ease-out; pointer-events: none; }
.toast.on { opacity: 1; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
<header>
  <h1>airi</h1><span class="sub">persistent state dashboard</span>
</header>
<nav>
  <button data-pane="memory" class="on">memory</button>
  <button data-pane="people">people</button>
  <button data-pane="metrics">metrics</button>
</nav>
<main>
  <section class="pane on" id="pane-memory">
    <div class="toolbar">
      <input type="search" id="q" placeholder="search memories…">
      <select id="ftype">
        <option value="">all types</option>
        <option value="episodic">episodic</option>
        <option value="semantic">semantic</option>
        <option value="relationship">relationship</option>
        <option value="preference">preference</option>
        <option value="conversation_summary">conversation summary</option>
        <option value="behavioral_pattern">behavioral pattern</option>
      </select>
      <span class="chip" id="count">—</span>
    </div>
    <div id="memlist"><div class="empty">loading…</div></div>
  </section>
  <section class="pane" id="pane-people">
    <div id="peoplelist"><div class="empty">loading…</div></div>
  </section>
  <section class="pane" id="pane-metrics">
    <div class="statgrid" id="statcards"></div>
    <h2 class="sect">messages per day — her vs the room</h2>
    <div class="stat">
      <div class="spark" id="spark"></div>
      <div class="xlabels"><span id="x0"></span><span id="x1"></span></div>
    </div>
    <h2 class="sect">action mix (persisted replies)</h2>
    <div class="stat" id="actionmix"></div>
    <h2 class="sect">language mix</h2>
    <div class="stat" id="langmix"></div>
    <h2 class="sect">emotional state (current)</h2>
    <div class="stat" id="emostate"></div>
  </section>
</main>
<div class="toast" id="toast"></div>
<script>
const $ = s => document.querySelector(s);
const esc = s => (s||'').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const num = n => (n ?? 0).toLocaleString('en-US');
const ago = ts => {
  if (!ts) return '—';
  const s = (Date.now() - new Date(ts + 'Z').getTime()) / 1000;
  if (s < 60) return Math.floor(s) + 's ago';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  if (s < 86400) return Math.floor(s/3600) + 'h ago';
  return Math.floor(s/86400) + 'd ago';
};
function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('on');
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('on'), 2200);
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

// ---- memory pane ----
async function loadMemories() {
  const q = encodeURIComponent($('#q').value);
  const t = encodeURIComponent($('#ftype').value);
  const data = await api(`/api/memories?q=${q}&type=${t}`);
  $('#count').textContent = data.total + ' memories';
  const el = $('#memlist');
  if (!data.rows.length) { el.innerHTML = '<div class="empty">no memories match</div>'; return; }
  el.innerHTML = data.rows.map(m => `
    <div class="mem" data-id="${m.id}">
      <div class="row1">
        <span class="chip t-${m.type}">${esc(m.type.replace('_',' '))}</span>
        <span class="when mono">${ago(m.created_at)}</span>
      </div>
      <div class="content">${esc(m.content)}</div>
      <div class="meta">
        <span>importance ${Number(m.importance).toFixed(2)}</span>
        <span>confidence ${Number(m.confidence).toFixed(2)}</span>
        <span>source ${esc(m.source)}</span>
        <span>recalled ${m.retrieval_count}×</span>
        ${m.superseded_by ? '<span style="color:var(--ink-faint)">superseded</span>' : ''}
      </div>
      <div class="actions" style="display:none">
        <textarea>${esc(m.content)}</textarea>
        <button class="btn primary" data-act="save">save</button>
        <button class="btn" data-act="cancel">cancel</button>
      </div>
      <div class="actions" style="display:flex">
        <button class="btn" data-act="edit">edit</button>
        <button class="btn danger" data-act="del">forget</button>
      </div>
    </div>`).join('');
}
$('#memlist').addEventListener('click', async e => {
  const btn = e.target.closest('button'); if (!btn) return;
  const card = btn.closest('.mem'); const id = card.dataset.id; const act = btn.dataset.act;
  if (act === 'edit') {
    card.querySelector('.content').style.display = 'none';
    card.querySelectorAll('.actions')[1].style.display = 'none';
    card.querySelectorAll('.actions')[0].style.display = 'flex';
    card.querySelector('textarea').focus();
  } else if (act === 'cancel') {
    loadMemories();
  } else if (act === 'save') {
    const content = card.querySelector('textarea').value.trim();
    if (!content) { toast('memory cannot be empty'); return; }
    await api(`/api/memories/${id}`, { method: 'PATCH', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ content }) });
    toast('memory updated — marked as correction');
    loadMemories();
  } else if (act === 'del') {
    if (!confirm('Delete this memory permanently? Airi will not remember it.')) return;
    await api(`/api/memories/${id}`, { method: 'DELETE' });
    toast('memory deleted — she has forgotten');
    loadMemories();
  }
});
let qTimer;
$('#q').addEventListener('input', () => { clearTimeout(qTimer); qTimer = setTimeout(loadMemories, 250); });
$('#ftype').addEventListener('change', loadMemories);

// ---- people pane ----
function meterRow(label, val, neg) {
  const pct = Math.round(((val ?? 0) + 1) / 2 * 100);
  return `<div class="meter ${neg && (val ?? 0) > 0 ? 'neg' : ''}"><span>${label}</span><div class="bar"><i style="width:${pct}%"></i></div><span>${Number(val ?? 0).toFixed(2)}</span></div>`;
}
async function loadPeople() {
  const data = await api('/api/people');
  const el = $('#peoplelist');
  if (!data.rows.length) { el.innerHTML = '<div class="empty">no relationships recorded yet</div>'; return; }
  el.innerHTML = data.rows.map(p => `
    <div class="person">
      <div><div class="name">${esc(p.name)}${p.canonical ? ' <span class="chip" style="color:var(--red);border-color:var(--red-dim)">anchor</span>' : ''}</div>
      <span class="u">@${esc(p.username || '—')} · ${num(p.interactions)} interactions</span></div>
      <div>
        ${meterRow('affection', p.affection)}
        ${meterRow('trust', p.trust)}
        ${meterRow('annoyance', p.annoyance, true)}
        ${meterRow('familiarity', p.familiarity)}
      </div>
      <div style="text-align:right">
        <span class="chip">${esc(p.type || 'friend')}</span>
        <div class="u" style="margin-top:6px">${p.last ? ago(p.last) : 'never'}</div>
      </div>
    </div>`).join('');
}

// ---- metrics pane ----
function barMix(el, entries) {
  const max = Math.max(1, ...entries.map(e => e[1]));
  el.innerHTML = entries.map(([k, v], i) =>
    `<div class="barrow ${i === 0 && v === max ? 'hot' : ''}"><span>${esc(k)}</span><div class="bar"><i style="width:${Math.round(v / max * 100)}%"></i></div><span>${num(v)}</span></div>`
  ).join('');
}
async function loadMetrics() {
  const d = await api('/api/metrics?days=30');
  const t = d.totals;
  $('#statcards').innerHTML = `
    <div class="stat"><div class="k">memories</div><div class="v">${num(t.memories)}</div></div>
    <div class="stat"><div class="k">people known</div><div class="v">${num(t.people)}</div></div>
    <div class="stat"><div class="k">messages seen</div><div class="v">${num(t.messages)}</div></div>
    <div class="stat"><div class="k">replies logged</div><div class="v">${num(Object.values(d.action_mix).reduce((a, b) => a + b, 0))}</div></div>`;
  // sparkline: her share of daily messages
  const spark = $('#spark');
  const days = d.per_day.slice(-30);
  if (days.length) {
    const max = Math.max(1, ...days.map(x => x.all));
    spark.innerHTML = days.map(x =>
      `<i class="${x.bot > 0 ? 'hot' : ''}" style="height:${Math.max(2, Math.round(x.all / max * 100))}%" title="${x.day}: ${x.bot} hers / ${x.all} total"></i>`).join('');
    $('#x0').textContent = days[0].day; $('#x1').textContent = days[days.length - 1].day;
  } else { spark.innerHTML = '<div class="empty" style="width:100%">no messages in the last 30 days</div>'; }
  barMix($('#actionmix'), Object.entries(d.action_mix).sort((a, b) => b[1] - a[1]));
  const L = d.language;
  barMix($('#langmix'), [['english', L.english], ['persian', L.persian], ['snippets', L.snippets]]);
  const E = d.emotional_state;
  $('#emostate').innerHTML = E ? Object.entries(E).map(([k, v]) =>
    `<div class="barrow"><span>${esc(k)}</span><div class="bar"><i style="width:${Math.round(((v + 1) / 2) * 100)}%"></i></div><span>${Number(v).toFixed(2)}</span></div>`).join('')
    : '<div class="empty">no emotional state yet</div>';
}

// ---- nav ----
document.querySelectorAll('nav button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.remove('on'));
  document.querySelectorAll('.pane').forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  $('#pane-' + b.dataset.pane).classList.add('on');
  ({ memory: loadMemories, people: loadPeople, metrics: loadMetrics })[b.dataset.pane]();
}));
loadMemories();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE


@app.get("/api/health")
async def health():
    return {"ok": True, "db": settings.database_url.split("///")[-1]}


@app.get("/api/memories")
async def list_memories(
    q: str = Query(default=""),
    type: str = Query(default=""),
    superseded: bool = Query(default=False),
    limit: int = Query(default=200, le=500),
    offset: int = Query(default=0),
):
    async with get_session() as session:
        stmt = select(Memory)
        count_stmt = select(func.count(Memory.id))
        if not superseded:
            stmt = stmt.where(Memory.superseded_by.is_(None))
            count_stmt = count_stmt.where(Memory.superseded_by.is_(None))
        if q:
            like = f"%{q}%"
            stmt = stmt.where(Memory.content.like(like))
            count_stmt = count_stmt.where(Memory.content.like(like))
        if type:
            try:
                mt = MemoryType(type)
            except ValueError:
                return JSONResponse({"error": f"unknown type '{type}'"}, status_code=400)
            stmt = stmt.where(Memory.memory_type == mt)
            count_stmt = count_stmt.where(Memory.memory_type == mt)
        stmt = stmt.order_by(Memory.created_at.desc()).limit(limit).offset(offset)
        rows = (await session.execute(stmt)).scalars().all()
        total = await session.scalar(count_stmt)
        return {
            "total": total,
            "rows": [
                {
                    "id": m.id,
                    "type": m.memory_type.value,
                    "content": m.content,
                    "importance": m.importance,
                    "confidence": m.confidence,
                    "source": m.source,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "retrieval_count": m.retrieval_count,
                    "superseded_by": m.superseded_by,
                }
                for m in rows
            ],
        }


@app.patch("/api/memories/{memory_id}")
async def update_memory(memory_id: int, body: dict = Body(...)):
    content = (body.get("content") or "").strip()
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)
    async with get_session() as session:
        m = (await session.execute(select(Memory).where(Memory.id == memory_id))).scalar_one_or_none()
        if not m:
            return JSONResponse({"error": "not found"}, status_code=404)
        m.content = content[:2000]
        # A human edit is the strongest kind of correction: bump importance so
        # it out-retrieves the noise around it, and mark the source honestly.
        m.importance = max(m.importance, 0.9)
        m.source = "correction"
        m.updated_at = _utcnow()
        await session.commit()
        return {"ok": True, "id": m.id}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: int):
    async with get_session() as session:
        m = (await session.execute(select(Memory).where(Memory.id == memory_id))).scalar_one_or_none()
        if not m:
            return JSONResponse({"error": "not found"}, status_code=404)
        await session.execute(delete(Memory).where(Memory.id == memory_id))
        await session.commit()
        return {"ok": True, "deleted": memory_id}


@app.get("/api/people")
async def people():
    async with get_session() as session:
        rows = (await session.execute(
            select(Relationship, User)
            .join(User, Relationship.user_id == User.id)
            .order_by(Relationship.interaction_count.desc())
            .limit(100)
        )).all()
        return {
            "rows": [
                {
                    "name": u.username or u.first_name or f"id{u.telegram_id}",
                    "username": u.username,
                    "affection": r.affection, "trust": r.trust, "respect": r.respect,
                    "annoyance": r.annoyance, "familiarity": r.familiarity,
                    "interest": r.interest, "comfort": r.comfort, "jealousy": r.jealousy,
                    "type": r.relationship_type, "canonical": r.is_canonical,
                    "interactions": r.interaction_count,
                    "last": r.last_interaction.isoformat() if r.last_interaction else None,
                }
                for r, u in rows
            ]
        }


@app.get("/api/metrics")
async def metrics(days: int = Query(default=30, le=120)):
    since = _utcnow() - timedelta(days=days)
    async with get_session() as session:
        # messages per day, hers vs the room — split in Python: 30 days of
        # message timestamps is tiny, and this keeps the query portable.
        day_rows = (await session.execute(
            select(Message.timestamp, Message.is_from_bot).where(Message.timestamp >= since)
        )).all()
        per_day: dict[str, dict[str, int]] = {}
        for ts, is_bot in day_rows:
            key = str(ts.date())
            bucket = per_day.setdefault(key, {"bot": 0, "all": 0})
            bucket["all"] += 1
            if is_bot:
                bucket["bot"] += 1
        per_day_list = [
            {"day": k, "bot": v["bot"], "all": v["all"]}
            for k, v in sorted(per_day.items())
        ]

        ts_actions = (await session.execute(
            select(TrainingSample.trigger, func.count(TrainingSample.id)).group_by(TrainingSample.trigger)
        )).all()
        action_mix = {t or "unknown": int(c) for t, c in ts_actions}

        lang = (await session.execute(
            select(
                func.coalesce(func.sum(LanguageStats.english_responses), 0),
                func.coalesce(func.sum(LanguageStats.persian_responses), 0),
                func.coalesce(func.sum(LanguageStats.persian_snippets), 0),
            )
        )).one()
        emo = (await session.execute(select(EmotionalState).limit(1))).scalar_one_or_none()
        totals = {
            "memories": await session.scalar(select(func.count(Memory.id)).where(Memory.superseded_by.is_(None))),
            "people": await session.scalar(select(func.count(User.id))),
            "messages": await session.scalar(select(func.count(Message.id))),
        }
        return {
            "per_day": per_day_list,
            "action_mix": action_mix,
            "language": {
                "english": int(lang[0]), "persian": int(lang[1]), "snippets": int(lang[2]),
            },
            "emotional_state": {
                "mood": emo.mood, "energy": emo.energy, "curiosity": emo.curiosity,
                "amusement": emo.amusement, "annoyance": emo.annoyance,
                "interest": emo.interest, "boredom": emo.boredom,
            } if emo else None,
            "totals": totals,
        }


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8766, log_level="warning")


if __name__ == "__main__":
    main()
