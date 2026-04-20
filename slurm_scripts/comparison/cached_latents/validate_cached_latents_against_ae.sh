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

resolve_oc_env_scalar() {
    local raw="$1"

    # Handle scalars like:
    #   ${oc.env:VAR,./fallback}/suffix
    # used in Hydra yaml files before interpolation is resolved.
    if [[ "${raw}" =~ ^\$\{oc\.env:([^,}]+),([^}]*)\}(.*)$ ]]; then
        local var_name="${BASH_REMATCH[1]}"
        local fallback="${BASH_REMATCH[2]}"
        local suffix="${BASH_REMATCH[3]}"
        local var_value="${!var_name:-}"
        if [[ -n "${var_value}" ]]; then
            printf "%s\n" "${var_value}${suffix}"
        else
            printf "%s\n" "${fallback}${suffix}"
        fi
        return 0
    fi

    printf "%s\n" "${raw}"
}

normalize_path_scalar() {
    local raw="$1"
    local block_data_path="${2:-}"
    local token='${.data_path}'
    local resolved

    # Handle same-block references such as "${.data_path}/stats.yml".
    if [[ -n "${block_data_path}" && "${raw}" == *"${token}"* ]]; then
        raw="${raw//$token/${block_data_path}}"
    fi

    resolved="$(resolve_oc_env_scalar "${raw}")"

    # Expand leading "~" so HOME-relative paths compare consistently.
    if [[ "${resolved}" == "~/"* ]]; then
        resolved="${HOME}/${resolved#~/}"
    fi

    # Canonicalize when possible; if the path does not exist, still normalize
    # relative references against the current working directory.
    if [[ -e "${resolved}" ]]; then
        resolved="$(realpath "${resolved}")"
    elif [[ "${resolved}" != /* ]]; then
        resolved="$(realpath -m "${resolved}" 2>/dev/null || printf "%s" "${resolved}")"
    fi

    # Avoid mismatch from a trailing slash only.
    resolved="${resolved%/}"
    printf "%s\n" "${resolved}"
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
    local ae_data_path_raw
    local cache_data_path_raw
    ae_data_path_raw="$(yaml_get_scalar_in_block "${ae_cfg}" "datamodule" "data_path")"
    cache_data_path_raw="$(yaml_get_scalar_in_block "${cache_cfg}" "datamodule" "data_path")"

    local key
    for key in "${keys[@]}"; do
        local ae_val
        local cache_val
        local ae_cmp
        local cache_cmp
        ae_val="$(yaml_get_scalar_in_block "${ae_cfg}" "datamodule" "${key}")"
        cache_val="$(yaml_get_scalar_in_block "${cache_cfg}" "datamodule" "${key}")"

        if [[ -z "${ae_val}" || -z "${cache_val}" ]]; then
            echo "Missing datamodule.${key} in ${ae_cfg} or ${cache_cfg}" >&2
            return 1
        fi
        ae_cmp="${ae_val}"
        cache_cmp="${cache_val}"
        if [[ "${key}" == "data_path" || "${key}" == "normalization_path" ]]; then
            ae_cmp="$(normalize_path_scalar "${ae_val}" "${ae_data_path_raw}")"
            cache_cmp="$(normalize_path_scalar "${cache_val}" "${cache_data_path_raw}")"
        fi
        if [[ "${ae_cmp}" != "${cache_cmp}" ]]; then
            echo "Mismatch datamodule.${key}" >&2
            echo "  AE config:     ${ae_val}" >&2
            echo "  Cached config: ${cache_val}" >&2
            if [[ "${key}" == "data_path" || "${key}" == "normalization_path" ]]; then
                echo "  AE normalized:     ${ae_cmp}" >&2
                echo "  Cached normalized: ${cache_cmp}" >&2
            fi
            echo "  AE cfg:        ${ae_cfg}" >&2
            echo "  Cache cfg:     ${cache_cfg}" >&2
            return 1
        fi
    done

    echo "Validated cached-latent settings against AE config for ${ae_run_dir}"
    return 0
}

validate_cache_experiment_against_ae() {
    local ae_run_dir="$1"
    local local_experiment="$2"
    local repo_root="$3"
    local ae_cfg="${ae_run_dir}/resolved_autoencoder_config.yaml"
    local experiment_cfg="${repo_root}/local_hydra/local_experiment/${local_experiment}.yaml"

    if [[ ! -f "${ae_cfg}" ]]; then
        echo "Missing AE resolved config: ${ae_cfg}" >&2
        return 1
    fi
    if [[ ! -f "${experiment_cfg}" ]]; then
        echo "Missing cache-latents experiment config: ${experiment_cfg}" >&2
        return 1
    fi

    local ae_use_norm
    local exp_use_norm
    ae_use_norm="$(yaml_get_scalar_in_block "${ae_cfg}" "datamodule" "use_normalization")"
    exp_use_norm="$(yaml_get_scalar_in_block "${experiment_cfg}" "datamodule" "use_normalization")"

    if [[ "${ae_use_norm}" != "true" && "${ae_use_norm}" != "false" ]]; then
        echo "Could not parse datamodule.use_normalization in ${ae_cfg}" >&2
        return 1
    fi
    if [[ "${exp_use_norm}" != "true" && "${exp_use_norm}" != "false" ]]; then
        echo "cache-latents experiment must explicitly set datamodule.use_normalization in ${experiment_cfg}" >&2
        return 1
    fi
    if [[ "${ae_use_norm}" != "${exp_use_norm}" ]]; then
        echo "Mismatch datamodule.use_normalization between AE and cache-latents experiment" >&2
        echo "  AE config:        ${ae_use_norm}" >&2
        echo "  Experiment config:${exp_use_norm}" >&2
        echo "  AE cfg:           ${ae_cfg}" >&2
        echo "  Experiment cfg:   ${experiment_cfg}" >&2
        return 1
    fi

    echo "Validated cache-latents experiment normalization against AE config for ${ae_run_dir}"
    return 0
}
