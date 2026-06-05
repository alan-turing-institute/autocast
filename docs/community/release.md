# Releasing AutoCast

This guide explains how to create new releases of AutoCast for maintainers.

## Release Process

1. **Update Version Number**

   Update the version in `pyproject.toml`:

   ```toml
   [project]
   name = "autocast"
   version = "X.Y.Z"  # Update this line
   ```

2. **Create and Push Tag**

   Create a new git tag following semantic versioning (vX.Y.Z):

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

3. **Create a Release on GitHub**

   Go to the [Releases page](https://github.com/alan-turing-institute/autocast/releases) of the repository and click "Draft a new release". Fill in the release title and description, then select the tag you just created. You can use "Generate release notes" to summarize changes since the last release.

## Prerequisites

Before creating a release, ensure:

1. All tests are passing on the main branch
2. Documentation is up to date
3. You have appropriate permissions to push tags to the repository

## Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Commit changes
- [ ] Create and push git tag
- [ ] Monitor GitHub Actions workflow
- [ ] Verify release is available
- [ ] Test installation

## Notes

- Only maintainers with appropriate permissions can create releases
- Each version can only be published once
