from __future__ import annotations

from pathlib import Path

from genmanip_client.clean_cli import collect_clean_targets, run


def test_collect_clean_targets_default_targets_mesh_eval_logs_and_temp_files(tmp_path: Path):
    (tmp_path / "client_results" / "run1").mkdir(parents=True)
    (tmp_path / "logs" / "run1").mkdir(parents=True)
    (tmp_path / "saved" / "assets" / "mesh_data" / "task1").mkdir(parents=True)
    (tmp_path / "saved" / "eval_results" / "bench" / "run1").mkdir(parents=True)
    (tmp_path / "saved" / "eval_results" / "bench" / "run1" / "episode_result.lock").write_text("x")
    (tmp_path / "saved" / "tasks" / "demo" / "log_soft.lock").parent.mkdir(parents=True)
    (tmp_path / "saved" / "tasks" / "demo" / "log_soft.lock").write_text("x")
    (tmp_path / "saved" / "tasks" / "demo" / "artifact.tmp-abc").write_text("x")
    (tmp_path / "saved" / "assets" / "collected_packages" / "pkg").mkdir(parents=True)

    targets = collect_clean_targets(tmp_path, include_all=False)
    rel_paths = {str(target.path.relative_to(tmp_path)) for target in targets}

    assert "client_results" not in rel_paths
    assert "logs" in rel_paths
    assert "saved/assets/mesh_data" in rel_paths
    assert "saved/eval_results" in rel_paths
    assert "saved/tasks/demo/log_soft.lock" in rel_paths
    assert "saved/tasks/demo/artifact.tmp-abc" in rel_paths
    assert "saved/assets/collected_packages" not in rel_paths


def test_run_dry_run_does_not_delete(tmp_path: Path, capsys):
    (tmp_path / "saved" / "assets" / "mesh_data" / "run1").mkdir(parents=True)
    (tmp_path / "client_results" / "run1").mkdir(parents=True)

    class Args:
        project_root = str(tmp_path)
        all = False
        dry_run = True

    result = run(Args())
    captured = capsys.readouterr()

    assert result == 0
    assert "Would remove" in captured.out
    assert (tmp_path / "saved" / "assets" / "mesh_data").exists()
    assert (tmp_path / "client_results").exists()


def test_collect_clean_targets_allows_symlinked_or_mounted_paths(tmp_path: Path):
    external_root = tmp_path / "external_store"
    (external_root / "assets" / "mesh_data" / "task1").mkdir(parents=True)
    (tmp_path / "saved").symlink_to(external_root, target_is_directory=True)

    targets = collect_clean_targets(tmp_path, include_all=False)
    rel_paths = {str(target.path.relative_to(tmp_path)) for target in targets}

    assert "saved/assets/mesh_data" in rel_paths
