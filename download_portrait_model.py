from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ZhengPeng7/BiRefNet-portrait",
    local_dir="birefnet_portrait",
    local_dir_use_symlinks=False
)