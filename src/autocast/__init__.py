import os

_RUNTIME_TYPECHECKING_ENABLED = {"1", "true", "yes", "on"}

if (
    os.getenv("RUNTIME_TYPECHECKING", "False").strip().lower()
    in _RUNTIME_TYPECHECKING_ENABLED
):
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    # Skip beartype on train/models to avoid conflicts with Hydra instantiation
    beartype_this_package(
        conf=BeartypeConf(
            claw_skip_package_names=(
                "autocast.train",
                "autocast.models",
                "autocast.types.collation",
            )
        )
    )
