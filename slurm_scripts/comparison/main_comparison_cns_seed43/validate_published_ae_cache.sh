#!/bin/bash
#
# Validation for the intentional cross-data cache:
# new seed-43 trajectories encoded by the fixed published autoencoder.

source "$(dirname "${BASH_SOURCE[0]}")/../cached_latents/validate_cached_latents_against_ae.sh"

validate_published_ae_cache_experiment() {
    local ae_run_dir="$1"
    local local_experiment="$2"
    local repo_root="$3"
    local expected_data_dir="$4"
    local ae_cfg="${ae_run_dir}/resolved_autoencoder_config.yaml"
    local experiment_cfg="${repo_root}/local_hydra/local_experiment/${local_experiment}.yaml"

    if [[ ! -f "${ae_cfg}" ]]; then
        echo "Missing published AE resolved config: ${ae_cfg}" >&2
        return 1
    fi
    if [[ ! -f "${experiment_cfg}" ]]; then
        echo "Missing cache experiment config: ${experiment_cfg}" >&2
        return 1
    fi

    local ae_data_raw
    local ae_norm_raw
    local ae_use_norm
    local exp_data_raw
    local exp_norm_raw
    local exp_use_norm
    ae_data_raw="$(yaml_get_scalar_in_block "${ae_cfg}" datamodule data_path)"
    ae_norm_raw="$(yaml_get_scalar_in_block "${ae_cfg}" datamodule normalization_path)"
    ae_use_norm="$(yaml_get_scalar_in_block "${ae_cfg}" datamodule use_normalization)"
    exp_data_raw="$(yaml_get_scalar_in_block "${experiment_cfg}" datamodule data_path)"
    exp_norm_raw="$(yaml_get_scalar_in_block "${experiment_cfg}" datamodule normalization_path)"
    exp_use_norm="$(yaml_get_scalar_in_block "${experiment_cfg}" datamodule use_normalization)"

    local expected_data
    local ae_norm
    local exp_data
    local exp_norm
    expected_data="$(normalize_path_scalar "${expected_data_dir}")"
    ae_norm="$(normalize_path_scalar "${ae_norm_raw}" "${ae_data_raw}")"
    exp_data="$(normalize_path_scalar "${exp_data_raw}")"
    exp_norm="$(normalize_path_scalar "${exp_norm_raw}" "${exp_data_raw}")"

    if [[ "${exp_data}" != "${expected_data}" ]]; then
        echo "Cache experiment does not point to the new dataset." >&2
        echo "  expected: ${expected_data}" >&2
        echo "  actual:   ${exp_data}" >&2
        return 1
    fi
    if [[ -z "${ae_norm_raw}" || "${exp_norm}" != "${ae_norm}" ]]; then
        echo "Cache experiment does not use the published AE normalization." >&2
        echo "  expected: ${ae_norm}" >&2
        echo "  actual:   ${exp_norm}" >&2
        return 1
    fi
    if [[ "${ae_use_norm}" != "true" || "${exp_use_norm}" != "${ae_use_norm}" ]]; then
        echo "Cache experiment normalization flag differs from the published AE." >&2
        echo "  published AE: ${ae_use_norm}" >&2
        echo "  experiment:   ${exp_use_norm}" >&2
        return 1
    fi

    echo "Validated new-data cache experiment against the published AE config"
}

validate_published_ae_cache_output() {
    local cache_dir="$1"
    local ae_run_dir="$2"
    local expected_data_dir="$3"
    local ae_cfg="${ae_run_dir}/resolved_autoencoder_config.yaml"
    local cache_cfg="${cache_dir}/autoencoder_config.yaml"

    if [[ ! -f "${ae_cfg}" ]]; then
        echo "Missing published AE resolved config: ${ae_cfg}" >&2
        return 1
    fi
    if [[ ! -f "${cache_cfg}" ]]; then
        echo "Missing generated cache config: ${cache_cfg}" >&2
        return 1
    fi

    local ae_data_raw
    local cache_data_raw
    ae_data_raw="$(yaml_get_scalar_in_block "${ae_cfg}" datamodule data_path)"
    cache_data_raw="$(yaml_get_scalar_in_block "${cache_cfg}" datamodule data_path)"

    local expected_data
    local cache_data
    expected_data="$(normalize_path_scalar "${expected_data_dir}")"
    cache_data="$(normalize_path_scalar "${cache_data_raw}")"
    if [[ "${cache_data}" != "${expected_data}" ]]; then
        echo "Generated cache did not use the new dataset." >&2
        echo "  expected: ${expected_data}" >&2
        echo "  actual:   ${cache_data}" >&2
        return 1
    fi

    local -a keys=(
        n_steps_input
        n_steps_output
        stride
        use_normalization
        normalization_path
    )
    local key
    for key in "${keys[@]}"; do
        local ae_val
        local cache_val
        local ae_cmp
        local cache_cmp
        ae_val="$(yaml_get_scalar_in_block "${ae_cfg}" datamodule "${key}")"
        cache_val="$(yaml_get_scalar_in_block "${cache_cfg}" datamodule "${key}")"
        if [[ -z "${ae_val}" || -z "${cache_val}" ]]; then
            echo "Missing datamodule.${key} in published AE or cache config." >&2
            return 1
        fi
        ae_cmp="${ae_val}"
        cache_cmp="${cache_val}"
        if [[ "${key}" == "normalization_path" ]]; then
            ae_cmp="$(normalize_path_scalar "${ae_val}" "${ae_data_raw}")"
            cache_cmp="$(normalize_path_scalar "${cache_val}" "${cache_data_raw}")"
        fi
        if [[ "${ae_cmp}" != "${cache_cmp}" ]]; then
            echo "Generated cache differs from published AE datamodule.${key}." >&2
            echo "  published AE: ${ae_cmp}" >&2
            echo "  cache:        ${cache_cmp}" >&2
            return 1
        fi
    done

    echo "Validated new-data cache output against the published AE config"
}
