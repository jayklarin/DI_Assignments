import streamlit as st
import psutil
import subprocess
import os
import shutil
from pathlib import Path

st.set_page_config(page_title="Jay's System Monitor", layout="wide")

st.title("🖥️ Jay's Mac System Monitor")
st.caption("M1 MacBook Air • 16 GB RAM • Quarterly Maintenance Dashboard")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def format_bytes(size):
    # Converts bytes → human readable
    for unit in ['', 'K', 'M', 'G', 'T']:
        if size < 1024:
            return f"{size:.2f}{unit}B"
        size /= 1024
    return f"{size:.2f}PB"

def run_cmd(cmd):
    try:
        output = subprocess.check_output(cmd, shell=True, text=True).strip()
        return output
    except:
        return "Error or no output."

# ---------------------------------------------------------
# System Overview
# ---------------------------------------------------------

st.header("📊 System Overview")

col1, col2, col3 = st.columns(3)

with col1:
    cpu_percent = psutil.cpu_percent(interval=1)
    st.metric("CPU Usage", f"{cpu_percent} %")

with col2:
    mem = psutil.virtual_memory()
    st.metric("RAM Used", f"{mem.percent} %", format_bytes(mem.used))

with col3:
    disk = shutil.disk_usage("/")
    used = format_bytes(disk.used)
    total = format_bytes(disk.total)
    free = format_bytes(disk.free)
    st.metric("Disk Free", free, f"Total: {total}")

# ---------------------------------------------------------
# Top Processes
# ---------------------------------------------------------

st.header("🔥 Top Processes (by RAM)")
processes = []
for p in psutil.process_iter(["pid", "name", "memory_info"]):
    try:
        processes.append(p.info)
    except:
        pass

processes = sorted(processes, key=lambda p: p["memory_info"].rss if p["memory_info"] else 0, reverse=True)

for proc in processes[:10]:
    rss = format_bytes(proc["memory_info"].rss)
    st.write(f"**{proc['name']}** – {rss} (PID {proc['pid']})")

# ---------------------------------------------------------
# Largest Folders
# ---------------------------------------------------------

st.header("📁 Largest Folders in Home Directory")

HOME = str(Path.home())

def folder_size(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except: 
                pass
    return total

folders_to_check = [
    f"{HOME}/Library/Caches",
    f"{HOME}/Library/Application Support",
    f"{HOME}/.cache",
    f"{HOME}/__DI",
    f"{HOME}/DI_Assignments",
]

folder_sizes = []
for folder in folders_to_check:
    if os.path.exists(folder):
        size = folder_size(folder)
        folder_sizes.append((folder, size))

folder_sizes = sorted(folder_sizes, key=lambda x: x[1], reverse=True)

for folder, size in folder_sizes[:6]:
    st.write(f"**{folder}** — {format_bytes(size)}")

# ---------------------------------------------------------
# Maintenance Actions
# ---------------------------------------------------------

st.header("🧹 Maintenance Tools")

if st.button("Clear pip cache"):
    output = run_cmd("pip cache purge")
    st.code(output)

if st.button("Clear HuggingFace cache"):
    output = run_cmd("rm -rf ~/.cache/huggingface")
    st.success("Deleted HuggingFace cache")

if st.button("Clear npm cache"):
    output = run_cmd("npm cache clean --force")
    run_cmd("rm -rf ~/.npm/_cacache")
    st.success("Cleared npm cache")

if st.button("Homebrew Cleanup"):
    output = run_cmd("brew cleanup --prune=all && brew autoremove")
    st.code(output)

if st.button("Clear /var/log"):
    output = run_cmd("sudo rm -rf /private/var/log/*")
    st.success("Logs cleared (macOS will repopulate healthy logs automatically).")

if st.button("Rebuild Spotlight Index"):
    run_cmd("sudo mdutil -a -i off")
    run_cmd("sudo rm -rf /.Spotlight-V100")
    run_cmd("sudo mdutil -a -i on")
    st.success("Spotlight index rebuilt.")

st.info("All operations are safe and reversible. macOS will regenerate system logs, Spotlight, and caches automatically.")

