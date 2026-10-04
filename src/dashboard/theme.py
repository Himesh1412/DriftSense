# src/dashboard/theme.py
"""
Shared dark theme for DriftSense's Streamlit apps (the main dashboard and the
standalone simulation tool), so both look like one product.
"""
import streamlit as st

BG, SURFACE, BORDER = "#0B0E13", "#12151C", "#232733"
TEXT, TEXT_MUTED = "#E7E9EE", "#8891A5"
TEAL, CORAL, AMBER = "#2FD4C4", "#F9695F", "#F5A623"

# NOTE: keep this <style> block free of blank lines — Streamlit's markdown
# renderer treats a blank line inside it as ending the raw-HTML block early,
# which dumps the remaining CSS onto the page as plain visible text instead
# of applying it as a stylesheet.
#
# .eyebrow: section labels now get real breathing room below them (was a
#   2px margin, which read as "congested" against whatever followed).
# .ds-card / .ds-metric-row / .ds-num: a pure-HTML card, used where the
#   content is plain numbers rather than a live Streamlit widget — it
#   genuinely wraps its contents since it's rendered in one st.markdown
#   call, unlike the old open-div/close-div-in-two-separate-calls pattern,
#   which never actually nested anything (the real source of the gap bug).
# stVerticalBlockBorderWrapper: Streamlit's own st.container(border=True),
#   reskinned to match .ds-card, used wherever the card holds a real widget
#   (chart, dataframe) that can't be flattened into raw HTML.
def apply_theme():
    """Inject the shared stylesheet. Call once, right after st.set_page_config."""
    st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
  .stApp {{ background-color: {BG}; color: {TEXT}; }}
  section[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
  .eyebrow {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.12em;
    text-transform: uppercase; color: {TEXT_MUTED}; margin-bottom: 14px; }}
  .ds-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; padding: 22px 24px; }}
  .ds-metric-row {{ display: flex; gap: 40px; flex-wrap: wrap; }}
  .ds-metric-label {{ font-size: 13px; color: {TEXT_MUTED}; margin-bottom: 6px; }}
  .ds-num {{ font-family: 'IBM Plex Mono', monospace; font-size: 30px; font-weight: 600; line-height: 1.2; }}
  .ds-warning {{ color: {AMBER}; font-size: 14px; }}
  .pill {{ display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    padding: 2px 8px; border-radius: 3px; }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE}; border: 1px solid {BORDER} !important; border-radius: 6px; }}
  hr {{ margin: 36px 0 !important; border-color: {BORDER}; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)
