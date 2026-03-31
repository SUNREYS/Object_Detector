# =============================================================================
# Dockerfile for YOLOv8 KAIST Pedestrian Detection Training
# =============================================================================
#
# Base: Official Ultralytics GPU image (includes PyTorch + CUDA + ultralytics)
# Extra: pandas, matplotlib, seaborn for evaluation & plotting
#
# The entire project directory is mounted at runtime via docker-compose,
# so the scripts' relative-path logic (SCRIPT_DIR / PROJECT_ROOT) works
# the same way as on the host machine.
#
# Build:
#   docker compose build
#
# Run training:
#   docker compose up train
# =============================================================================

FROM ultralytics/ultralytics:latest

# Install additional Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Working directory is set by docker-compose to the mounted OD folder
CMD ["python", "train.py", "--help"]
