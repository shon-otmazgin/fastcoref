#!/bin/bash
set -e

echo "Upgrading dependencies..."

# First, upgrade the lock file to get latest compatible versions
uv lock --upgrade
echo "Lock file updated."

# Now read the lock file and update pyproject.toml
uv run --with tomli python -c "
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:
        raise RuntimeError('Python 3.10 requires tomli for TOML parsing. Install tomli>=2.0.1.') from exc
import re

# Read uv.lock to get locked versions
with open('uv.lock', 'rb') as f:
    lock_data = tomllib.load(f)

# Build a map of package names to versions from lock file
locked_versions = {}
for package in lock_data.get('package', []):
    name = package.get('name', '').lower()
    version = package.get('version', '')
    if name and version:
        locked_versions[name] = version

# Read pyproject.toml
with open('pyproject.toml', 'rb') as f:
    project_data = tomllib.load(f)

with open('pyproject.toml', 'r') as f:
    content = f.read()

# Update dependencies, skipping those with upper bounds
for dep in project_data['project']['dependencies']:
    pkg = re.split(r'[>=<~!=]+', dep.strip())[0]
    if '<' in dep:
        continue
    if pkg.lower() in locked_versions:
        old_dep_pattern = r'[\"'']' + re.escape(dep) + r'[\"'']'
        version = locked_versions[pkg.lower()]
        # Strip local version identifiers (+cpu, +cu121, etc.) from version
        version = re.sub(r'\+.*$', '', version)
        new_dep = f'{pkg}>={version}'
        content = re.sub(old_dep_pattern, f'\"{ new_dep}\"', content)

if 'dependency-groups' in project_data:
    for group_name, group_deps in project_data['dependency-groups'].items():
        for dep in group_deps:
            pkg = re.split(r'[>=<~!=]+', dep.strip())[0]
            if '<' in dep:
                continue
            if pkg.lower() in locked_versions:
                old_dep_pattern = r'[\"'']' + re.escape(dep) + r'[\"'']'
                version = locked_versions[pkg.lower()]
                # Strip local version identifiers (+cpu, +cu121, etc.) from version
                version = re.sub(r'\+.*$', '', version)
                new_dep = f'{pkg}>={version}'
                content = re.sub(old_dep_pattern, f'\"{ new_dep}\"', content)

with open('pyproject.toml', 'w') as f:
    f.write(content)
"

echo "pyproject.toml updated with locked versions!"
echo "Dependencies upgrade completed!"
