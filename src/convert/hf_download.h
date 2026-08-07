/*
 * hf_download.h — Native HuggingFace file download (libcurl), no Python.
 *
 * Downloads files from the HF hub resolve endpoint:
 *   GET https://huggingface.co/{repo_id}/resolve/main/{filename}
 *
 * The hub serves via CDN (redirects), so curl follows redirects. TLS is
 * verified against the system CA store — self-signed certs are rejected.
 * Downloads are written to a temp file in dest_dir and atomically renamed
 * into place on success.
 */
#pragma once

#include <string>
#include <vector>

namespace terllama {

// Default URL-encoding helper exposed for reuse.
std::string hf_url_encode(const std::string& s);

// Download {filename} from {repo_id} into {dest_dir}, writing dest_dir/filename.
// hf_token: optional HF token; empty string sends no Authorization header.
// Returns 0 on success, nonzero on failure.
int hf_download_file(const std::string& repo_id,
                     const std::string& filename,
                     const std::string& dest_dir,
                     const std::string& hf_token);

// True if config.json content indicates a sharded model — i.e. the JSON
// references the index file "model.safetensors.index.json" (either as the
// value of a key such as "model_index" or via any nested string equal to it).
bool should_use_index(const std::string& config_json);

// Download the shard index (model.safetensors.index.json) plus every unique
// .safetensors shard listed in its weight_map. out_files receives the full
// local paths of the downloaded shards (index not included).
// Returns 0 on success, nonzero on failure.
int hf_download_sharded(const std::string& repo_id,
                        const std::string& dest_dir,
                        const std::string& hf_token,
                        std::vector<std::string>& out_files);

} // namespace terllama
