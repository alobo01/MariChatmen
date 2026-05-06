"""License normalization and filtering for source datasets."""

from __future__ import annotations

from marichatmen.constants import DEFAULT_ALLOWED_LICENSES, SOURCE_LICENSES


def normalize_license(value: str | None) -> str:
    if not value:
        return "unknown"
    clean = value.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "apache2.0": "apache-2.0",
        "apache-2": "apache-2.0",
        "apache-2.0": "apache-2.0",
        "cc-by-4": "cc-by-4.0",
        "cc-by-4.0": "cc-by-4.0",
        "cc-by-sa-4": "cc-by-sa-4.0",
        "cc-by-sa-4.0": "cc-by-sa-4.0",
        "odc-by-1": "odc-by-1.0",
        "odc-by-1.0": "odc-by-1.0",
    }
    clean = aliases.get(clean, clean)
    return clean


def source_license(source_data: str | None) -> str:
    if not source_data:
        return "unknown"
    return SOURCE_LICENSES.get(source_data, "unknown")


def license_allowed(
    source_data: str | None,
    allowed: set[str] | None = None,
    explicit_license: str | None = None,
) -> bool:
    allowed_norm = {normalize_license(item) for item in (allowed or DEFAULT_ALLOWED_LICENSES)}
    lic = normalize_license(explicit_license or source_license(source_data))
    return lic in allowed_norm
