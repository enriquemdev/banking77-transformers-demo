"""Cloud deployment. Import is blocked until a trained model and budget approval exist."""
import os
from pathlib import Path

import modal

from backend.inference import FALCON_ID, FALCON_REVISION, validate_checkpoint

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get('BANKING77_MODEL_DIR', ROOT / 'model' / 'best')).resolve()
if modal.is_local():
    # These are deployment-time gates, not credentials needed by a running worker.
    if os.environ.get('B77_DEPLOY_BUDGET_CONFIRMED') != 'yes':
        raise RuntimeError('Deployment is disabled. First verify the Modal net spend limit and credits, then explicitly set B77_DEPLOY_BUDGET_CONFIRMED=yes.')
    validate_checkpoint(MODEL_DIR)

ENABLE_FALCON = os.environ.get('B77_ENABLE_FALCON', 'no') == 'yes'

app = modal.App('banking77-enrique-munoz')
cache = modal.Volume.from_name('banking77-falcon-cache', create_if_missing=True)
secret = modal.Secret.from_name('banking77-demo-config', required_keys=['B77_DEMO_ACCESS_SHA256', 'B77_ALLOWED_ORIGINS'])
base = (
    modal.Image.debian_slim(python_version='3.12').env({'B77_ENABLE_FALCON': 'yes' if ENABLE_FALCON else 'no'})
    .pip_install('fastapi==0.128.0', 'transformers==4.57.6', 'huggingface-hub==0.36.2', 'safetensors==0.8.0')
    .add_local_dir(str(ROOT / 'backend'), '/root/backend', copy=True, ignore=['__pycache__', '*.pyc'])
)
cpu_image = base.pip_install('torch==2.8.0', index_url='https://download.pytorch.org/whl/cpu').add_local_dir(str(MODEL_DIR), '/root/model/best', copy=True, ignore=['training_args.bin'])
for filename in ['index.html', 'styles.css', 'app.js', 'config.js']:
    cpu_image = cpu_image.add_local_file(str(ROOT / filename), '/root/' + filename)
cpu_image = cpu_image.add_local_dir(str(ROOT / 'assets'), '/root/assets')
if ENABLE_FALCON:
    gpu_image = base.pip_install('torch==2.8.0', 'accelerate==1.12.0', 'bitsandbytes==0.49.2')


    @app.function(image=base, cpu=2, memory=4096, timeout=1800, max_containers=1, volumes={'/cache': cache})
    def prefetch_falcon():
        """Download on CPU once, not on a billed GPU every cold start."""
        from huggingface_hub import snapshot_download
        snapshot_download(FALCON_ID, revision=FALCON_REVISION, cache_dir='/cache/hf',
                          allow_patterns=['*.json', '*.safetensors'])
        Path('/cache/ready-revision.txt').write_text(FALCON_REVISION)
        cache.commit()
        return {'revision': FALCON_REVISION, 'cached': True}


    @app.cls(image=gpu_image, gpu='T4', cpu=2, memory=8192, min_containers=0, max_containers=1,
             scaledown_window=30, timeout=300, volumes={'/cache': cache})
    class FalconService:
        @modal.enter()
        def load(self):
            from backend.inference import Falcon
            if not Path('/cache/ready-revision.txt').exists():
                raise RuntimeError('Pre-cache Falcon on CPU before enabling explanations.')
            self.falcon = Falcon(cache_dir='/cache/hf', local_files_only=True)

        @modal.method()
        def generate(self, query: str, label: str):
            return self.falcon.explain(query, label)


@app.function(image=cpu_image, cpu=2, memory=4096, min_containers=0, max_containers=1,
              scaledown_window=30, timeout=300, secrets=[secret], volumes={'/cache': cache})
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    from backend.api import create_app
    from backend.inference import Classifier

    async def explain(query, label):
        return await FalconService().generate.remote.aio(query, label)

    marker = Path('/cache/ready-revision.txt')
    cached = ENABLE_FALCON and marker.is_file() and marker.read_text() == FALCON_REVISION
    return create_app(Classifier('/root/model/best'), explain if cached else None, deployment=True)
