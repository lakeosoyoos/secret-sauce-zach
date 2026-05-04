"""
Secret Sauce — Duplicate Classification Streamlit app.

Upload a .zip, .sor files, or .json files. The app auto-detects the mode,
splits SOR files by direction prefix, and produces a downloadable PDF.

Run:  streamlit run app.py
"""
import os
import re
import sys
import shutil
import tempfile
import zipfile
import io
from collections import defaultdict

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from report import run_json_bytes, run_trc_bytes
from report_sor import run_sor_bytes
from sor_reader324802a import direction_key_from_genparams


st.set_page_config(
    page_title="Secret Sauce — Duplicate Classification",
    layout="wide",
)

# ----- password gate ----------------------------------------------------
APP_PASSWORD = st.secrets.get("app_password", "") if hasattr(st, "secrets") else ""

if not st.session_state.get("authed"):
    st.title("Secret Sauce")
    if not APP_PASSWORD:
        st.error(
            "App password is not configured. Set `app_password` in Streamlit "
            "Cloud → app **Settings** → **Secrets**, or in a local "
            "`.streamlit/secrets.toml` for development."
        )
        st.stop()
    pwd = st.text_input("Password", type="password")
    if pwd == APP_PASSWORD:
        st.session_state["authed"] = True
        st.rerun()
    elif pwd:
        st.error("Incorrect password.")
    st.stop()

st.title("Secret Sauce")
st.caption(
    "Upload a .zip, .sor files, or .json files. The app detects the input "
    "mode, splits SOR files by direction prefix so forward and reverse are "
    "analysed separately, and produces a single downloadable PDF per "
    "direction."
)

# ----- upload -----------------------------------------------------------
# file_uploader keeps its own state; bumping a key counter is how Streamlit
# apps "clear" it — the widget re-mounts empty on next rerun.
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

uploads = st.file_uploader(
    "Drop .sor, .trc, .json files and/or a .zip here",
    type=["sor", "trc", "json", "zip"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}",
)

if st.button("Clear uploads", type="secondary"):
    st.session_state["uploader_key"] += 1
    st.rerun()

if not uploads:
    st.info("Waiting for files…")
    st.stop()


# ----- stage uploads to a temp dir --------------------------------------
tmp_dir = tempfile.mkdtemp(prefix="secret_sauce_")


def _stage(uf, dest_dir):
    name = uf.name
    data = uf.getbuffer()
    if name.lower().endswith(".zip"):
        zpath = os.path.join(dest_dir, name)
        with open(zpath, "wb") as fh:
            fh.write(data)
        saved = []
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner = os.path.basename(info.filename)
                if not inner or inner.startswith("._") or inner == ".DS_Store":
                    continue
                low = inner.lower()
                if not (low.endswith(".sor") or low.endswith(".trc")
                        or low.endswith(".json")):
                    continue
                dst = os.path.join(dest_dir, inner)
                with zf.open(info) as src, open(dst, "wb") as out:
                    out.write(src.read())
                saved.append(dst)
        os.remove(zpath)
        return saved
    dst = os.path.join(dest_dir, name)
    with open(dst, "wb") as fh:
        fh.write(data)
    return [dst]


saved = []
for uf in uploads:
    saved.extend(_stage(uf, tmp_dir))

sor_files = [p for p in saved if p.lower().endswith(".sor")]
trc_files = [p for p in saved if p.lower().endswith(".trc")]
json_files = [p for p in saved if p.lower().endswith(".json")]

n_kinds = sum(bool(x) for x in (sor_files, trc_files, json_files))
if n_kinds == 0:
    st.error("No .sor, .trc, or .json files found in the uploads.")
    st.stop()
if n_kinds > 1:
    st.error("Mixed file types found. Upload one of .sor / .trc / .json at a time.")
    st.stop()

st.success(f"Loaded {len(sor_files)} SOR + {len(trc_files)} TRC + {len(json_files)} JSON file(s).")


# ----- SOR: group by file-internal direction signal --------------------
# Read GenParams from each SOR (location_a/b → operator+OTDR serial → OTDR
# serial). Filename is only used as a last-resort fallback when the file
# carries no useful metadata. Two files end up in the same group iff they
# share the same direction key — i.e. they were shot from the same end of
# the cable with the same OTDR.
_FILENAME_PREFIX_RE = re.compile(r"^(.*?)sh\d+", re.IGNORECASE)


def _filename_fallback_key(filename):
    base = os.path.basename(filename)
    m = _FILENAME_PREFIX_RE.match(base)
    return (m.group(1) if m else "group").rstrip("-_ ") or "group"


def _direction_key(filepath):
    """Resolve the grouping key from inside the SOR. Falls back to the
    filename prefix only if GenParams carries nothing usable."""
    key = direction_key_from_genparams(filepath)
    if key:
        return key
    return _filename_fallback_key(filepath)


def _group_sor(paths):
    groups = defaultdict(list)
    for p in paths:
        groups[_direction_key(p)].append(p)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _copy_to_subdir(paths, subdir):
    os.makedirs(subdir, exist_ok=True)
    for p in paths:
        shutil.copy(p, os.path.join(subdir, os.path.basename(p)))
    return subdir


# ----- run analysis ----------------------------------------------------
reports = []  # (filename, bytes, n_files, n_pairs, label)

if sor_files:
    groups = _group_sor(sor_files)
    if not groups:
        st.error("Could not form any SOR direction group with ≥2 files.")
        st.stop()

    st.write(f"Detected **{len(groups)} direction(s)**.")
    cols = st.columns(len(groups))
    for col, (prefix, paths) in zip(cols, groups.items()):
        col.markdown(f"**{prefix}**")
        col.metric("Files", len(paths))

    for prefix, paths in groups.items():
        subdir = _copy_to_subdir(paths, os.path.join(tmp_dir, f"sor_{prefix}"))
        title = (f"Secret Sauce — {prefix}" if len(groups) > 1
                 else "Secret Sauce — Duplicate Classification")
        fname = (f"report_{prefix}.pdf" if len(groups) > 1 else "report.pdf")
        with st.spinner(f"Running {prefix} ({len(paths)} files)…"):
            try:
                pdf_bytes, n_files, n_pairs = run_sor_bytes(subdir, title)
            except Exception as e:
                st.error(f"{prefix}: {e}")
                continue
        reports.append((fname, pdf_bytes, n_files, n_pairs, prefix))
elif trc_files:
    subdir = _copy_to_subdir(trc_files, os.path.join(tmp_dir, "trc_input"))
    with st.spinner(f"Running Secret Sauce on {len(trc_files)} TRC files…"):
        try:
            pdf_bytes, n_files, n_pairs = run_trc_bytes(
                subdir, title="Secret Sauce — Duplicate Classification")
        except Exception as e:
            st.error(str(e))
            st.stop()
    reports.append(("report.pdf", pdf_bytes, n_files, n_pairs, "TRC"))
else:
    subdir = _copy_to_subdir(json_files, os.path.join(tmp_dir, "json_input"))
    with st.spinner(f"Running Secret Sauce on {len(json_files)} JSON files…"):
        try:
            pdf_bytes, n_files, n_pairs = run_json_bytes(
                subdir, title="Secret Sauce — Duplicate Classification")
        except Exception as e:
            st.error(str(e))
            st.stop()
    reports.append(("report.pdf", pdf_bytes, n_files, n_pairs, "JSON"))


if not reports:
    st.error("No reports were generated.")
    st.stop()


# ----- summary + download ----------------------------------------------
cols = st.columns(len(reports))
for col, (_, _, n_files, n_pairs, label) in zip(cols, reports):
    col.markdown(f"**{label}**")
    col.metric("Files", n_files)
    col.metric("Pairs", n_pairs)


if len(reports) == 1:
    fname, pdf_bytes, _, _, _ = reports[0]
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=fname,
        mime="application/pdf",
        type="primary",
    )
else:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, pdf_bytes, _, _, _ in reports:
            zf.writestr(fname, pdf_bytes)
    buf.seek(0)
    st.download_button(
        "Download combined ZIP",
        data=buf.getvalue(),
        file_name="secret_sauce_reports.zip",
        mime="application/zip",
        type="primary",
    )
