# src/dashboard/theme.py
"""
Shared light theme for DriftSense's Streamlit apps (the main dashboard and the
standalone simulation tool), so both look like one product: cool-grey page,
soft white rounded cards, a blue accent, monospace numbers.
"""
import streamlit as st

BG, SURFACE, BORDER, GRID = "#FFFFFF", "#FFFFFF", "#C9D3E3", "#E3E9F3"   # white base; BORDER frames cards, GRID is for charts
TEXT, TEXT_MUTED, INK = "#0F172A", "#46546B", "#0F172A"
ACCENT, ACCENT_SOFT, ACCENT_LINE = "#1D4ED8", "#DCE8FD", "#5B8DEF"
TEAL, CORAL, AMBER = "#0B8A55", "#C62828", "#B45309"                    # names kept for the two apps: up / down / warning
UP, DOWN = TEAL, CORAL
UP_SOFT, DOWN_SOFT, WARN_SOFT, TILE = "#E1F4EA", "#FBE4E4", "#FCEFD6", "#F3F7FD"   # tinted fills for tiles and tags

# NOTE: keep this <style> block free of blank lines — Streamlit's markdown
# renderer treats a blank line inside it as ending the raw-HTML block early,
# which dumps the remaining CSS onto the page as plain visible text instead
# of applying it as a stylesheet.
#
# .eyebrow: small uppercase section label.
# .ds-card / .ds-metric-row / .ds-num: a pure-HTML card for plain numbers; it
#   wraps its contents because it is rendered in one st.markdown call.
# stVerticalBlockBorderWrapper: Streamlit's st.container(border=True), reskinned
#   to match .ds-card, for cards that hold a real widget (chart, dataframe).
# .nav / .pill / .brand: the top bar.  .wl-*: the watchlist panel.
def apply_theme(roomy: bool = False):
    """Inject the shared stylesheet. Call once, right after st.set_page_config.

    roomy=True is for form-heavy pages (the Simulation Engine): bigger text and wider gaps."""
    st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"], .stApp {{ font-family: 'Manrope', sans-serif; }}
  .stApp {{ background-color: {BG}; color: {TEXT}; }}
  .block-container {{ padding: 18px 28px 24px; max-width: 1440px; }}
  div[data-testid="stVerticalBlock"] {{ gap: 0.75rem; }}
  div[data-testid="stHorizontalBlock"] {{ gap: 0.75rem; }}
  section[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none; }}
  .eyebrow {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: {TEXT}; margin-bottom: 8px; padding-left: 9px; border-left: 3px solid {ACCENT};
    scroll-margin-top: 16px; }}
  .ds-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 22px 24px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.08); }}
  .ds-metric-row {{ display: flex; gap: 40px; flex-wrap: wrap; }}
  .ds-metric-label {{ font-size: 13px; font-weight: 500; color: {TEXT_MUTED}; margin-bottom: 6px; }}
  .ds-num {{ font-family: 'IBM Plex Mono', monospace; font-size: 30px; font-weight: 600; line-height: 1.2; color: {TEXT}; }}
  .ds-warning {{ color: {AMBER}; font-size: 14px; font-weight: 500; }}
  .pill, a.pill, a.pill:visited {{ display: inline-block; font-size: 13px; font-weight: 600; padding: 9px 18px; border-radius: 999px;
    color: {TEXT} !important; text-decoration: none !important; }}
  .pill.on {{ background: {ACCENT}; color: #fff !important; }}
  a.pill:hover {{ background: {ACCENT_SOFT}; }}
  .nav {{ display: flex; align-items: center; justify-content: space-between; gap: 18px;
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 28px; padding: 10px 22px; margin-bottom: 22px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.08); }}
  .nav .right {{ display: flex; align-items: center; gap: 12px; }}
  .nav .left {{ display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }}
  .brand {{ font-size: 20px; font-weight: 800; color: {TEXT}; }}
  .tag {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600; padding: 6px 12px; border-radius: 999px;
    background: {ACCENT_SOFT}; color: {ACCENT}; }}
  .tag.warn {{ background: {WARN_SOFT}; color: #7A3E05; }}
  .wl {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 18px 14px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.08); }}
  .wl-scroll {{ max-height: 560px; overflow-y: auto; margin-top: 8px; padding-right: 2px; }}
  .wl-group {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: {ACCENT}; margin: 12px 6px 6px; }}
  a.wl-row {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 14px;
    border-radius: 12px; border: 1px solid transparent; border-bottom: 1px solid {GRID}; text-decoration: none !important; color: {TEXT} !important; }}
  a.wl-row:hover {{ background: {TILE}; }}
  a.wl-row.sel {{ background: {ACCENT_SOFT}; border: 1px solid {ACCENT_LINE}; }}
  .wl-sym {{ font-size: 15px; font-weight: 800; }}
  .wl-sub {{ font-size: 12px; font-weight: 600; color: {TEXT_MUTED}; }}
  .wl-px {{ font-family: 'IBM Plex Mono', monospace; font-size: 14px; font-weight: 500; text-align: right; }}
  .wl-chg {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600; text-align: right; }}
  .bigpx {{ font-family: 'IBM Plex Mono', monospace; font-size: 48px; font-weight: 500; line-height: 1.1; color: {TEXT}; }}
  .chg {{ font-family: 'IBM Plex Mono', monospace; font-size: 15px; font-weight: 600; margin-left: 12px; }}
  .sub {{ font-size: 14px; font-weight: 600; color: {TEXT_MUTED}; margin-bottom: 4px; }}
  .tile {{ background: {TILE}; border: 1px solid #D3DEEF; border-radius: 12px; padding: 14px 16px; min-height: 96px; box-sizing: border-box; }}
  .tile.up {{ background: {UP_SOFT}; border-color: #AEDDC4; }}
  .tile.down {{ background: {DOWN_SOFT}; border-color: #F1B8B8; }}
  .tile.warn {{ background: {WARN_SOFT}; border-color: #EBCB93; }}
  .tile .ds-num {{ font-size: 24px; }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE}; border: 1px solid {BORDER} !important; border-radius: 16px; padding: 0;
    box-shadow: 0 1px 3px rgba(15,23,42,0.08); }}
  div[data-testid="stTextInput"] input {{ border-radius: 12px; background: #fff; border: 1px solid {BORDER}; }}
  hr {{ margin: 28px 0 !important; border-color: {BORDER}; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)
    if roomy:
        # Form-heavy page: larger type, wider gaps, padded cards. Kept free of blank lines (see the note above).
        st.markdown(f"""
<style>
  .block-container {{ padding: 28px 40px 40px; max-width: 1400px; }}
  div[data-testid="stVerticalBlock"] {{ gap: 1.25rem; }}
  div[data-testid="stHorizontalBlock"] {{ gap: 2rem; }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{ padding: 14px 18px; }}
  .stApp, [data-testid="stMarkdownContainer"] p {{ font-size: 16px !important; line-height: 1.6; }}
  [data-testid="stWidgetLabel"] p, label p {{ font-size: 16px !important; font-weight: 600; color: {TEXT}; }}
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ font-size: 15px !important; color: {TEXT_MUTED}; opacity: 1 !important; }}
  [data-testid="stRadio"] label p, [data-testid="stCheckbox"] label p {{ font-size: 16px !important; font-weight: 500; }}
  div[data-testid="stSlider"] {{ padding-bottom: 10px; }}
  div[data-testid="stSlider"] [data-testid="stTickBarMin"], div[data-testid="stSlider"] [data-testid="stTickBarMax"] {{ font-size: 13px; }}
  .eyebrow {{ font-size: 13.5px; margin-bottom: 14px; }}
  div[data-testid="stButton"] {{ margin-top: 10px; }}
  .sec {{ border-top: 1px solid {GRID}; margin: 8px 0 16px; padding-top: 20px; }}
  .ds-metric-label {{ font-size: 15px; }}
  .ds-num {{ font-size: 32px; }}
  .tile {{ padding: 18px 20px; min-height: 108px; }}
  .tile .ds-num {{ font-size: 30px; }}
  .note {{ background: {WARN_SOFT}; border: 1px solid #EBCB93; border-radius: 14px; padding: 16px 22px;
    color: #7A3E05; font-size: 15px; font-weight: 500; line-height: 1.55; margin: 20px 0 28px; }}
  .pagehead {{ display: flex; justify-content: space-between; align-items: center; gap: 18px; background: {SURFACE};
    border: 1px solid {BORDER}; border-radius: 28px; padding: 14px 28px; box-shadow: 0 1px 3px rgba(15,23,42,0.08); }}
  .pagehead .title {{ font-size: 22px; font-weight: 800; color: {TEXT}; }}
  .pagehead .crumb {{ font-size: 14px; color: {TEXT_MUTED}; font-weight: 600; margin-left: 12px; }}
</style>
""", unsafe_allow_html=True)
