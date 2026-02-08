# syntax=docker/dockerfile:1
FROM trtllm-rc6-dynamo-nixl:latest

SHELL ["/bin/bash", "-lc"]

# Minimal utilities that help with mount + debugging.
# - util-linux: findmnt, mount
# - libmount: used by cuFile to inspect mount options
# - udev: udevadm sometimes useful (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    util-linux \
    libmount1 \
    udev \
    pciutils \
    procps \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# cuFile config (you can overwrite at runtime using CUFILE_ENV_PATH_JSON)
RUN mkdir -p /etc/cufile /var/log/cufile \
 && chmod 777 /var/log/cufile

# Default cuFile config. Tune as needed.
# NOTE: allow_compat_mode=true while you're diagnosing; you can tighten later.
RUN cat > /etc/cufile/cufile.json <<'JSON'
{
  "logging": { "dir": "/var/log/cufile", "level": "DEBUG" },
  "profile": { "nvtx": false, "cufile_stats": 0 },
  "properties": {
    "allow-compat-mode": true,
    "use-compat-mode": true,
    "force-compat-mode": true,
    "use-pci-p2pdma": false
  }
}
JSON

# Bootstrap that sets envs, ensures /dev/nvidia-fs exists, etc.
COPY bootstrap-nixl.sh /usr/local/bin/bootstrap-nixl.sh
RUN chmod +x /usr/local/bin/bootstrap-nixl.sh

ENTRYPOINT ["/usr/local/bin/bootstrap-nixl.sh"]
CMD ["bash", "-lc", "python3 -m dynamo_llm.entrypoints.llm_api_server --config /tmp/kvbm_llm_api_config.yaml"]
