"""The small amount of CSS the native Streamlit theme can't express."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root {
  --ink: #1d1d1f;
  --muted: #5f6368;
  --faint: #8a8f98;
  --line: #e2e2de;
  --panel: #ffffff;
  --accent: #23466e;
  --band: #efe9d6;
}
[data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }

/* Header */
.app-header { padding-bottom: 1.1rem; margin-bottom: 0.3rem; border-bottom: 1px solid var(--line); }
.app-kicker { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.35rem; }
.app-header h1 { font-size: 1.85rem; font-weight: 600; margin: 0 0 0.3rem; padding: 0; letter-spacing: -0.015em; }
.app-header p { color: var(--muted); margin: 0 0 0.7rem; max-width: 46rem; line-height: 1.55; }
.app-links { font-size: 0.9rem; display: flex; flex-wrap: wrap; gap: 0.4rem 1.25rem; }
.app-links a { color: var(--accent); text-decoration: none; border-bottom: 1px solid #23466e40; }
.app-links a:hover { border-bottom-color: var(--accent); }

/* Tabs: accent underline only */
.stTabs [data-baseweb="tab"] { font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--accent); }

.note { color: var(--muted); font-size: 0.88rem; line-height: 1.5; }
.mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.85em; }

/* Panels */
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 0.35rem; padding: 1.05rem 1.2rem; }
.panel p { margin: 0 0 0.5rem; color: var(--muted); font-size: 0.92rem; line-height: 1.55; }
.verdict-panel { border-left: 4px solid var(--line); }
.verdict-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--faint); margin-bottom: 0.3rem; font-weight: 500; }
.verdict { font-size: 1.45rem; font-weight: 600; margin-bottom: 0.9rem; }
.stress-score { font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem; font-variant-numeric: tabular-nums; }
.edited {font-size: 0.95rem; line-height: 1.6; color: var(--ink); }

/* Probability bar */
.prob-row { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.4rem; }
.prob-row strong { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
.prob-track { position: relative; height: 10px; background: #ecece8; border-radius: 5px; overflow: hidden; }
.prob-band { position: absolute; top: 0; bottom: 0; background: var(--band); }
.prob-fill { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 5px; transition: width 400ms ease; }
.prob-tick { position: absolute; top: 0; bottom: 0; width: 1px; background: #c9c1a3; }
.prob-scale { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--faint); margin-top: 0.35rem; }

/* Word highlights */
.tok-text { font-size: 1rem; line-height: 2.05; color: var(--ink); background: var(--panel); border: 1px solid var(--line); border-radius: 0.35rem; padding: 0.8rem 1rem; }
.tok { border-radius: 2px; padding: 0.12rem 0; cursor: help; transition: outline-color 120ms; outline: 1px solid transparent; }
.tok:hover { outline-color: #1d1d1f55; }
.legend { display: flex; flex-wrap: wrap; gap: 0.3rem 1rem; font-size: 0.8rem; color: var(--muted); margin: 0 0 0.5rem; }
.legend-item { display: inline-flex; align-items: center; gap: 0.35rem; }
.dot { width: 0.65rem; height: 0.65rem; border-radius: 2px; display: inline-block; }

/* Stat tiles */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin: 0.4rem 0 0.8rem; }
.tile { background: var(--panel); border: 1px solid var(--line); border-top: 3px solid; border-radius: 0.35rem; padding: 0.75rem 0.9rem; }
.tile-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--faint); font-weight: 500; }
.tile-value { font-size: 1.55rem; font-weight: 600; margin-top: 0.15rem; font-variant-numeric: tabular-nums; }
.tile-note { font-size: 0.8rem; color: var(--muted); margin-top: 0.1rem; }

.chip { display: inline-block; font-size: 0.75rem; font-weight: 500; padding: 0.1rem 0.5rem; border-radius: 0.25rem; border: 1px solid; }

/* Callout for key findings */
.finding { border-left: 3px solid var(--accent); background: #23466e0a; padding: 0.7rem 1rem; border-radius: 0 0.35rem 0.35rem 0; margin: 0.4rem 0 1rem; }
.finding strong { color: var(--ink); }
</style>
"""
