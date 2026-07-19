# Changelog

All notable changes to this repository will be documented in this file.

The format follows Keep a Changelog, and this project uses semantic versioning principles for release notes.

## [Unreleased]

### Added
- Production-grade repository documentation and governance files.
- Secure environment template for deployment configuration.

### Changed
- Main README rewritten to reflect product architecture and operational setup.
- Repository hygiene rules expanded in `.gitignore`.

### Security
- Removed tracked secrets, certificates, and generated runtime artifacts from version control.
- Updated user authentication persistence to store password hashes instead of plaintext.
