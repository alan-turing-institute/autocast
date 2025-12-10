import os

if os.getenv("RUNTIME_TYPECHECKING", "True").lower() in ["1", "true"]:
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    # Skip beartype on train/models to avoid conflicts with Hydra instantiation
    beartype_this_package(
        conf=BeartypeConf(claw_skip_package_names=("autocast.train", "autocast.models"))
    )
