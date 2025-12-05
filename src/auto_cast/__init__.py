import os

if os.getenv("RUNTIME_TYPECHECKING", "True").lower() in ["1", "true"]:
    from beartype.claw import beartype_this_package

    beartype_this_package()
