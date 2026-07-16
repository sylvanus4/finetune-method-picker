# Contributing

Thanks for your interest.

## Before you open a PR
```bash
node test/compute.test.cjs && node audit/js_dump.cjs > /tmp/js.json && python3 audit/reference_audit.py /tmp/js.json
```
If you add a behavior, add a test that would fail without it.

## Style
- Clear, readable code; small functions; helpful error messages.
- No secrets in code, tests, or fixtures.

## Security
For vulnerabilities see [SECURITY.md](SECURITY.md), not a public issue.

## License
By contributing you agree your contributions are licensed under the MIT License.
