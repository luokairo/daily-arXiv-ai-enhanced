"""Save/restore verified crawl pages, AI cache and raw snapshots, never partial reports."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
import time
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def copy_tree(source, destination):
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def save_bundle(data, stage, output, date):
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        raise ValueError('Recovery needs a selected YYYY-MM-DD date')
    output.mkdir(parents=True, exist_ok=True)
    copy_tree(data / 'ai_cache', output / 'data/ai_cache')
    copy_tree(data / 'crawl_cache' / date, output / 'data/crawl_cache' / date)
    raw = data / f'{date}.jsonl'
    if raw.exists():
        (output / 'data').mkdir(exist_ok=True)
        shutil.copy2(raw, output / 'data' / raw.name)
    copy_tree(data / 'run_metrics', output / 'data/run_metrics')
    copy_tree(stage / 'run_metrics', output / 'data/run_metrics')
    (output / 'usage').mkdir(exist_ok=True)
    for usage in stage.glob('*usage.json'):
        shutil.copy2(usage, output / 'usage' / usage.name)
    if (stage / 'deep_reads/deep-read-progress.json').exists():
        shutil.copy2(stage / 'deep_reads/deep-read-progress.json', output / 'usage/deep-read-progress.json')
    manifest = dict(version=1, date=date, files={
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'recovery.json'})
    (output / 'recovery.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def restore_zip(archive, data):
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        manifest = json.loads(bundle.read('recovery.json'))
        if manifest.get('version') != 1:
            raise ValueError('Unsupported recovery version')
        for name, digest in manifest['files'].items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or name not in names:
                raise ValueError('Invalid recovery path')
            content = bundle.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError('Recovery checksum mismatch')
            # Both AI signatures and crawl page identity/date/hash/TTL are
            # validated again before use. Partial reports are never restored.
            if name.startswith(('data/ai_cache/', 'data/crawl_cache/', 'data/run_metrics/')) or re.fullmatch(r'data/\d{4}-\d{2}-\d{2}\.jsonl', name):
                destination = data / path.relative_to('data')
                if destination.exists():
                    if not name.startswith('data/crawl_cache/'):
                        continue
                    # The data directory or an earlier bundle may contain an
                    # older page. Never let it mask a fresher verified download.
                    incoming = json.loads(content)
                    try:
                        existing = json.loads(destination.read_text())
                        if (existing.get('retrieved_at', 0) >= incoming['retrieved_at']
                                and hashlib.sha256(existing['html'].encode()).hexdigest()
                                == existing['sha256']):
                            continue
                    except (ValueError, KeyError, TypeError):
                        pass
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
    return manifest['date']


def restore_diagnostic_zip(archive, data, day):
    # Only a fully revalidated category with a Replacement-only warning can be
    # promoted. Raw diagnostic HTML is otherwise never treated as recovery data.
    from scripts.crawl_dated import CACHE_MAX_AGE, PageCache, category_catchup
    restored = 0
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not name.endswith('/diagnostic.json'):
                continue
            report = json.loads(bundle.read(name))
            category = report.get('category', '')
            if (report.get('target_date') != day or report.get('error_type') != 'CountMismatchError'
                    or not re.fullmatch(r'[a-z]+\.[A-Za-z]+', category)):
                continue
            retrieved_at = datetime.fromisoformat(report['recorded_at']).timestamp()
            if not 0 <= time.time() - retrieved_at <= CACHE_MAX_AGE:
                continue
            pages = {}
            for entry in report['pages']:
                filename = Path(entry['file'])
                number = entry['page']
                url = f'https://arxiv.org/catchup/{category}/{day}?abs=True&page={number}'
                if (filename.is_absolute() or '..' in filename.parts or len(filename.parts) != 1
                        or type(number) is not int or not 1 <= number <= 100 or entry['url'] != url):
                    raise ValueError('Invalid diagnostic page identity')
                pages[url] = (number, bundle.read(str(Path(name).parent / filename)).decode('utf-8'))

            class ArchivedClient:
                def get(self, url):
                    if url not in pages:
                        raise ValueError('Diagnostic category has missing pages')
                    return pages[url][1]

            try:
                _, counts = category_catchup(ArchivedClient(), category, day)
            except ValueError:
                continue
            if not counts['warnings']:
                continue
            cache = PageCache(data / 'crawl_cache', day, category)
            for url, (number, html) in pages.items():
                destination = cache.directory / f'page-{number}.json'
                if destination.exists() and cache.load(number, url) is not None:
                    if json.loads(destination.read_text())['retrieved_at'] >= retrieved_at:
                        continue
                cache.save(number, url, html, retrieved_at=retrieved_at)
                restored += 1
    return restored


def restore_remote(data, repository, date=''):
    if date and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        raise ValueError('Invalid recovery date')
    response = subprocess.check_output(['gh', 'api', '--paginate', '--slurp',
        f'repos/{repository}/actions/artifacts?per_page=100'], text=True)
    all_artifacts = [a for page in json.loads(response) for a in page['artifacts'] if not a['expired']]
    artifacts = [a for a in all_artifacts
                 if not a['expired'] and a['name'].startswith('ai-recovery-' + (date + '-' if date else ''))]
    artifacts.sort(key=lambda a: a['created_at'], reverse=True)
    restored = 0
    recovered_runs = {}
    for artifact in artifacts[:3]:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / 'recovery.zip'
                with archive.open('wb') as stream:
                    subprocess.run(['gh', 'api', f'repos/{repository}/actions/artifacts/{artifact["id"]}/zip'], stdout=stream, check=True)
                recovered_date = restore_zip(archive, data)
                recovered_runs[artifact['workflow_run']['id']] = recovered_date
                restored += 1
                print(f'Restored crawl pages and successful cache/snapshot from {artifact["name"]} ({recovered_date})', flush=True)
        except (ValueError, KeyError, OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
            print(f'Skipped unusable recovery artifact: {type(error).__name__}', flush=True)
    print(f'Recovery artifacts restored: {restored}', flush=True)
    for artifact in all_artifacts:
        recovered_date = recovered_runs.get(artifact.get('workflow_run', {}).get('id'))
        if not recovered_date or not artifact['name'].startswith('crawl-diagnostics-'):
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / 'diagnostics.zip'
                with archive.open('wb') as stream:
                    subprocess.run(['gh', 'api', f'repos/{repository}/actions/artifacts/{artifact["id"]}/zip'],
                                   stdout=stream, check=True)
                count = restore_diagnostic_zip(archive, data, recovered_date)
                print(f'Revalidated diagnostic pages restored for {recovered_date}: {count}', flush=True)
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
            print(f'Skipped unusable diagnostic artifact: {type(error).__name__}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['save', 'restore'])
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--stage', type=Path, default=Path('/tmp/arxiv-stage'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/arxiv-recovery'))
    parser.add_argument('--date', default='')
    parser.add_argument('--repository', default=os.environ.get('GITHUB_REPOSITORY', ''))
    args = parser.parse_args()
    if args.mode == 'save':
        save_bundle(args.data, args.stage, args.output, args.date)
    else:
        restore_remote(args.data, args.repository, args.date)


if __name__ == '__main__':
    main()
