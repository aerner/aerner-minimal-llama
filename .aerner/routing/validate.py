#!/usr/bin/python3
"""Validate untrusted routing descriptions; metadata never grants execution rights."""
import argparse
import copy
import json
from pathlib import Path, PurePosixPath
import sys
import yaml
from jsonschema import Draft202012Validator, ValidationError

SCHEMA = Path(__file__).with_name('schema.json')
class UniqueLoader(yaml.SafeLoader):
    pass

def mapping(loader, node, deep=False):
    out = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in out:
            raise ValueError('duplicate_or_nonstring_key')
        out[key] = loader.construct_object(value_node, deep=deep)
    return out
UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)

def relative(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('invalid_relative_path')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or str(p) != value:
        raise ValueError('invalid_relative_path')
    return p

def parse(raw):
    if len(raw.encode('utf-8')) > 16384:
        raise ValueError('metadata_too_large')
    # Aliases/anchors can form recursive values or expansion bombs; no need in short metadata.
    for token in yaml.scan(raw):
        if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)):
            raise ValueError('yaml_alias_not_allowed')
    value = yaml.load(raw, Loader=UniqueLoader)
    Draft202012Validator(json.loads(SCHEMA.read_text())).validate(value)
    for path in value.get('references', []):
        relative(path)
    return value

def merge(parent, child):
    value = copy.deepcopy(parent)
    for key, item in child.items():
        if key == 'requirements':
            value.setdefault(key, {}).update(copy.deepcopy(item))
        else:
            value[key] = copy.deepcopy(item)
    return value

def resolve(metadata, folder):
    relative(folder)
    result = {}
    ancestors = ['.'] if folder == '.' else ['.'] + ['/'.join(PurePosixPath(folder).parts[:i]) for i in range(1, len(PurePosixPath(folder).parts)+1)]
    if '.' not in metadata:
        raise ValueError('root_metadata_missing')
    for p in ancestors:
        if p in metadata:
            result = merge(result, metadata[p])
    if result.get('status') in ('unknown', 'archive') and any(x != 'investigate' for x in result.get('work', [])):
        raise ValueError('unreviewed_execution_requirements')
    return result

def checked_path(root, folder):
    target = root
    for component in relative(folder).parts:
        target = target / component
        if target.is_symlink():
            raise ValueError('selected_path_symlink')
    if not target.exists() or not target.resolve().is_relative_to(root):
        raise ValueError('selected_path_missing_or_escaping')
    return target

def scan(root):
    if root.is_symlink():
        raise ValueError('repository_root_symlink')
    root = root.resolve(strict=True)
    metadata = {}
    # Do not traverse a linked directory, including links whose targets are internal.
    import os
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', '.venv', '__pycache__') and not (Path(base)/d).is_symlink()]
        if 'aerner.project.yaml' not in files:
            continue
        path = Path(base)/'aerner.project.yaml'
        if path.is_symlink():
            raise ValueError('metadata_symlink')
        value = parse(path.read_text())
        for ref in value.get('references', []):
            checked_path(root, (Path(base)/ref).relative_to(root).as_posix())
        folder = Path(base).relative_to(root).as_posix()
        metadata[folder] = value
        if len(metadata) > 64:
            raise ValueError('too_many_metadata_files')
    for folder in metadata:
        resolve(metadata, folder)
    return metadata

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--folder', default='.')
    args = p.parse_args()
    try:
        if args.root.is_symlink():
            raise ValueError('repository_root_symlink')
        target = checked_path(args.root.resolve(strict=True), args.folder)
        if not target.is_dir():
            raise ValueError('folder_missing_or_symlink')
        print(json.dumps(resolve(scan(args.root), args.folder), ensure_ascii=False))
    except (ValueError, ValidationError, yaml.YAMLError, OSError, RecursionError) as e:
        print('metadata_rejected:' + type(e).__name__, file=sys.stderr)
        sys.exit(1)
