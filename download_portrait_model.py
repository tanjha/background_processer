from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ZhengPeng7/BiRefNet_dynamic",
    local_dir="BiRefNet_dynamic",
    local_dir_use_symlinks=False
)