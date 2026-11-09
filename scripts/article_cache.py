"""Shared, lossless, sharded article-text cache for CEE monitor collectors."""
import hashlib
import json
from pathlib import Path

MAX_SHARD_BYTES = 4_000_000


def _paths(data_dir):
    base = Path(data_dir)
    return base / 'article_text_cache.json', base / 'article_text_cache_parts'


def load_items(data_dir):
    legacy, folder = _paths(data_dir)
    items = {}
    if legacy.exists():
        obj = json.loads(legacy.read_text(encoding='utf-8'))
        if not isinstance(obj, dict):
            raise ValueError(f'Invalid article cache: {legacy}')
        raw = obj.get('items', obj)
        if not isinstance(raw, dict):
            raise ValueError(f'Invalid article cache items: {legacy}')
        items.update({k: v for k, v in raw.items() if isinstance(v, dict)})
    for file in sorted(folder.glob('*.json')) if folder.exists() else []:
        obj = json.loads(file.read_text(encoding='utf-8'))
        if not isinstance(obj, dict):
            raise ValueError(f'Invalid article cache shard: {file}')
        items.update(obj)
    return items


def save_items(data_dir, items):
    legacy, folder = _paths(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    buckets = {}
    for url, entry in items.items():
        if not isinstance(entry, dict):
            continue
        # Stable hash buckets prevent reshuffling existing records between runs.
        key = hashlib.sha256(url.encode('utf-8')).hexdigest()[:3]
        buckets.setdefault(key, {})[url] = entry
    expected = set()
    for key, entries in buckets.items():
        # Split large hash buckets further by URL hash, preserving every entry.
        groups = {'': entries}
        depth = 3
        while any(len(json.dumps(v, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) > MAX_SHARD_BYTES for v in groups.values()):
            depth += 1
            if depth > 64:
                raise ValueError('Single cache entry exceeds maximum shard size')
            groups = {}
            for url, entry in entries.items():
                suffix = hashlib.sha256(url.encode('utf-8')).hexdigest()[3:depth]
                groups.setdefault(suffix, {})[url] = entry
        for suffix, group in groups.items():
            name = f'{key}{suffix}.json'
            target = folder / name
            raw = json.dumps(group, ensure_ascii=False, separators=(',', ':'))
            if not target.exists() or target.read_text(encoding='utf-8') != raw:
                tmp = target.with_suffix('.json.tmp')
                tmp.write_text(raw, encoding='utf-8')
                tmp.replace(target)
            expected.add(name)
    # Remove superseded shards only after all replacement shards are written.
    for old in folder.glob('*.json'):
        if old.name not in expected:
            old.unlink()
    # Legacy path is kept as a tiny, valid index for old readers; no data is removed
    # until all collectors have been migrated to this shared module.
    legacy.write_text(json.dumps({'version': 2, 'items': {}, 'storage': 'article_text_cache_parts'}, ensure_ascii=False), encoding='utf-8')
