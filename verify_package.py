"""Read-only package integrity and parameter-file contents check."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.io import loadmat


def main():
    root = Path(__file__).resolve().parent
    lock = json.loads((root / 'figures/fig6/COMPUTATION_LOCK.json').read_text(encoding='utf-8'))
    for item in lock['files']:
        assert hashlib.sha256((root / item['path']).read_bytes()).hexdigest() == item['sha256'], (
            'Formal Fig6 computation source changed', item['path'])
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    files = {item['path']: item for item in manifest['files']}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and '.git' not in p.relative_to(root).parts}
    assert actual == set(files) | {'manifest.json'}, 'Unlisted, missing or generated files in package.'
    allowed = {'.md', '.py', '.m', '.ps1', '.txt', '.json', '.mat'}
    for name, item in files.items():
        path = root / name
        assert path.suffix in allowed or name == '.gitignore', name
        content = path.read_bytes()
        assert len(content) == item['bytes'], name
        assert hashlib.sha256(content).hexdigest() == item['candidate_sha256'], name

    fields = {'format', 'alpha', 'b', 'D', 'channel_gain', 'component_scales',
              'component_matrices', 'W_residual', 'residual_projection', 'TR',
              'DeltaT', 'generation_sigma', 'generation_noise_factor',
              'generation_use_core_drift'}
    models = sorted(root.rglob('*.mat'))
    assert {p.relative_to(root).as_posix() for p in models} == {
        f'models/hcp{n}_example.mat' for n in (150, 250, 450)}
    for path in models:
        loaded = loadmat(path, simplify_cells=True)
        assert {key for key in loaded if not key.startswith('__')} == {'parameters'}
        parameters = loaded['parameters']
        assert set(parameters) == fields and parameters['format'] == 'hodge_sde_parameters_v1'
        for key, value in parameters.items():
            if key == 'format':
                continue
            values = value if key == 'component_matrices' else [value]
            for part in values:
                array = np.asarray(part)
                assert array.dtype.kind in 'biuf' and np.isfinite(array).all(), (path.name, key)
    print(f'PACKAGE_INTEGRITY_PASS: {len(files)} files, {len(models)} numerical parameter models; no unlisted files.')


if __name__ == '__main__':
    main()
