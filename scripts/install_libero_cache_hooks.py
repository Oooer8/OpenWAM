"""Insert opt-in hooks without replacing unrelated local training edits."""
import difflib
from pathlib import Path

RECEIPTS = Path('/mnt/vepfs01/output/tiecheng.guo/artifacts/waw/libero_cache_integration_receipts')
CHANGES = {
    'openwam/dataloader/registry.py': [
        ('    from openwam.dataloader.libero import LiberoDataset\n',
         '    from openwam.dataloader.libero import LiberoDataset\n    from openwam.dataloader.libero_latent_cache import CachedLiberoDataset\n'),
        ('    register_dataset("libero")(LiberoDataset)\n',
         '    register_dataset("libero")(LiberoDataset)\n    register_dataset("libero_cached")(CachedLiberoDataset)\n')],
    'openwam/train/utils/runtime_optimizations.py': [
        ('def validate_text_cache_config(cfg):\n',
         'def validate_text_cache_config(cfg):\n    from openwam.dataloader.libero_latent_cache import validate_training_config\n    validate_training_config(cfg)\n'),
        ('def apply_training_optimizations(architecture):\n',
         'def apply_training_optimizations(architecture):\n    from openwam.dataloader.libero_latent_cache import apply_training_cache\n    apply_training_cache(architecture)\n')]
}


def main():
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    for name, changes in CHANGES.items():
        path = Path(name)
        before = text = path.read_text()
        for old, new in changes:
            if new in text:
                continue
            if text.count(old) != 1:
                raise RuntimeError(f'Ambiguous insertion point in {name}')
            text = text.replace(old, new, 1)
        if text != before:
            backup = RECEIPTS/(name.replace('/', '__')+'.before')
            if not backup.exists():
                backup.write_text(before)
            compile(text, name, 'exec')
            temp = path.with_suffix('.libero_tmp')
            temp.write_text(text)
            temp.replace(path)
            (RECEIPTS/(name.replace('/', '__')+'.patch')).write_text(''.join(
                difflib.unified_diff(before.splitlines(True), text.splitlines(True), fromfile=name, tofile=name)))
        print(name, 'ready')


if __name__ == '__main__':
    main()
