# Security Policy

## Supported Versions

This repository tracks a single active branch (`main`). Security updates are shipped on `main` only.

## Reporting a Vulnerability

If you discover a security issue, do not open a public issue.

Report privately using one of these channels:
- GitHub Security Advisories for this repository
- Direct contact to the repository owner account: `muhammadaamirgulzar`

Please include:
- A clear description of the vulnerability
- Reproduction steps or proof of concept
- Impact assessment
- Suggested remediation if available

You can expect an acknowledgment within 72 hours.

## Security Practices

- Secrets and certificates must never be committed.
- Environment variables must be supplied via deployment secrets.
- Dependencies should be reviewed regularly for known CVEs.
- Access tokens and API keys should be rotated on a schedule.
