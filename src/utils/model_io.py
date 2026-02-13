import json
import shutil
from pathlib import Path
from datetime import datetime


class ModelManager:
    def __init__(self, base_dir="models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, agent, metadata, checkpoint_name=None, method="model"):
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        checkpoint_dir = self.base_dir / method / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        agent.save(checkpoint_dir / "model.zip")

        metadata.update({"saved_at": datetime.now().isoformat(), "method": method})

        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        return str(checkpoint_dir)

    def load_checkpoint(self, checkpoint_path, agent_class, env=None):
        path = Path(checkpoint_path)
        
        if path.is_dir():
            model_path, metadata_path = path / "model.zip", path / "metadata.json"
        else:
            model_path = path
            metadata_path = path.with_name(path.stem + "_metadata.json")

        agent = agent_class.load(str(model_path), env=env)
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

        return agent, metadata

    def list_checkpoints(self, method=None):
        checkpoints = []
        pattern = f"{method or '*'}/checkpoint_*"

        for cp_dir in self.base_dir.glob(pattern):
            model_path = cp_dir / "model.zip"
        
            if not model_path.exists():
                continue

            info = {"path": str(cp_dir), "method": cp_dir.parent.name}
            meta_path = cp_dir / "metadata.json"
        
            if meta_path.exists():
                info.update(json.loads(meta_path.read_text()))

            checkpoints.append(info)

        return sorted(checkpoints, key=lambda x: x.get("saved_at", ""), reverse=True)

    def cleanup_old_checkpoints(self, method, keep_last_n=5):
        for cp in self.list_checkpoints(method)[keep_last_n:]:
            shutil.rmtree(cp["path"], ignore_errors=True)
