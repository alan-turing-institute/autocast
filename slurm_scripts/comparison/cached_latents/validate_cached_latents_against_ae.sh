#!/bin/bash
#
# Shared preflight check for cached-latent runs.
# Ensures cached_latents/autoencoder_config.yaml matches the AE training
# datamodule settings from resolved_autoencoder_config.yaml.

yaml_get_scalar_in_block() {
    local yaml_file="$1"
    local block="$2"
    local key="$3"

    awk -v block="${block}" -v key="${key}" '
        function ltrim(s) { sub(/^[ \t\r\n]+/, "", s); return s }
        function rtrim(s) { sub(/[ \t\r\n]+$/, "", s); return s }
        function trim(s) { return rtrim(ltrim(s)) }

        {
            line = $0
            if (line ~ "^[[:space:]]*" block ":[[:space:]]*$") {
                in_block = 1
                block_indent = match(line, /[^ ]/) - 1
                next
            }

            if (in_block) {
                if (line ~ /^[[:space:]]*$/) {
                    next
                }
                indent = match(line, /[^ ]/) - 1
                if (indent <= block_indent) {
                    in_block = 0
                }
            }

            if (in_block && line ~ "^[[:space:]]*" key ":[[:space:]]*") {
                sub(/^[[:space:]]*[^:]+:[[:space:]]*/, "", line)
                sub(/[[:space:]]+#.*$/, "", line)
                line = trim(line)
                if (line ~ /^".*"$/ || line ~ /^'\''.*'\''$/) {
                    line = substr(line, 2, length(line) - 2)
                }
                print line
                exit
            }
        }
    ' "${yaml_file}"
}

validate_cached_latents_against_ae() {
    local ae_run_dir="$1"
    local ae_cfg="${ae_run_dir}/resolved_autoencoder_config.yaml"
    local cache_cfg="${ae_run_dir}/cached_latents/autoencoder_config.yaml"

    if [[ ! -f "${ae_cfg}" ]]; then
        echo "Missing AE resolved config: ${ae_cfg}" >&2
        return 1
    fi
    if [[ ! -f "${cache_cfg}" ]]; then
        echo "Missing cached-latents config: ${cache_cfg}" >&2
        return 1
    fi

    local -a keys=(
        "data_path"
        "n_steps_input"
        "n_steps_output"
        "stride"
        "use_normalization"
        "normalization_path"
    )

    local key
    for key in "${keys[@]}"; do
        local ae_val
        local cache_val
        ae_val="$(yaml_get_scalar_in_block "${ae_cfg}" "datamodule" "${key}")"
        cache_val="$(yaml_get_scalar_in_block "${cache_cfg}" "datamodule" "${key}")"

        if [[ -z "${ae_val}" || -z "${cache_val}" ]]; then
            echo "Missing datamodule.${key} in ${ae_cfg} or ${cache_cfg}" >&2
            return 1
        fi
        if [[ "${ae_val}" != "${cache_val}" ]]; then
            echo "Mismatch datamodule.${key}" >&2
            echo "  AE config:     ${ae_val}" >&2
            echo "  Cached config: ${cache_val}" >&2
            echo "  AE cfg:        ${ae_cfg}" >&2
            echo "  Cache cfg:     ${cache_cfg}" >&2
            return 1
        fi
    done

    echo "Validated cached-latent settings against AE config for ${ae_run_dir}"
    return 0
}
