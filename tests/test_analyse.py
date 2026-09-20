from genmanip_client.extensions.analyse import aggregate


def test_generalization_matches_task_and_padded_seed():
    def record(task, seed, sr, score, run="run1", split="test_mini"):
        return dict(run_id=run, task=task, split=split, seed=seed, sr=sr, score=score)

    payload = aggregate([
        record("apple_from_shelf", "000", 1, 0.8),
        record("apple_from_shelf", 1, 0, 0.4),
        record("bottle", "0", 1, 0.6),  # Same seed, different dimension.
        record("bottle", "005", 0, 0.2),
        record("bottle", "015", 1, 1),
        record("bottle", "010", 1, 1, run="run2"),
        record("apple_from_shelf", "agg", 1, 1),
        record("apple_from_shelf", "999", 1, 1),
        record("unknown_task", "000", 1, 1),
        record("apple_from_shelf", "000", 1, 1, split="val_train"),
    ])
    gen = payload["agg_generalize"]
    assert gen["run1"]["test_mini_success_rate"]["Object"] == {
        "mean": 0.5, "std": 0.7071, "n": 2,
    }
    assert gen["run1"]["test_mini_score"]["Object"] == {
        "mean": 0.6, "std": 0.2828, "n": 2,
    }
    assert gen["run1"]["test_mini_score"]["Instruction"]["mean"] == 0.6
    assert gen["run1"]["test_mini_score"]["Background"]["mean"] == 0.2
    assert gen["run1"]["test_mini_score"]["Mix"]["mean"] == 1
    assert gen["run2"]["test_mini_success_rate"]["Object"]["n"] == 1
    assert set(gen["run1"]) == {"test_mini_score", "test_mini_success_rate"}


def test_generalization_bundled_mapping_coverage():
    import json
    from genmanip_client.extensions import analyse

    mapping = json.loads(analyse._BUNDLED_EPISODE_DIMENSIONS.read_text())
    records = []
    for task_path, episodes in mapping["tasks"].items():
        task, split = analyse.parse_task_name(task_path.rsplit("/", 1)[-1])
        for seed in episodes:
            records.append(dict(run_id="run", task=task, split=split,
                                seed=seed, sr=1, score=1))
    gen = aggregate(records)["agg_generalize"]["run"]["test_mini_score"]
    assert {dim: stats["n"] for dim, stats in gen.items()} == {
        "Background": 130, "Instruction": 130, "Mix": 130, "Object": 120,
    }


def test_generalization_loaders_and_summary_exclusion(tmp_path):
    import json
    from genmanip_client.extensions.analyse import load_run_episode_results

    task_dir = tmp_path / "apple_from_shelf_test_mini"
    task_dir.mkdir()
    (task_dir / "episode_result.json").write_text(json.dumps({
        "0": {"sr": 1, "score": 0.8},
    }))
    records = load_run_episode_results(tmp_path)
    assert aggregate(records)["agg_generalize"][tmp_path.name][
        "test_mini_score"]["Object"]["mean"] == 0.8

    episode_dir = task_dir / "005"
    episode_dir.mkdir()
    (episode_dir / "result_info.json").write_text(json.dumps({"sr": 0, "score": 0.2}))
    records = load_run_episode_results(tmp_path)
    assert aggregate(records)["agg_generalize"][tmp_path.name][
        "test_mini_score"]["Instruction"]["mean"] == 0.2

    (episode_dir / "result_info.json").unlink()
    (tmp_path / "result.json").write_text(json.dumps({
        "apple_from_shelf_test_mini": {"sr": 0.5, "score": 0.7},
    }))
    assert aggregate(load_run_episode_results(tmp_path))["agg_generalize"] == {}


def test_merge_reference_preserves_generalization_and_local_precedence():
    from genmanip_client.extensions.analyse import merge_reference

    local_metrics = {"test_mini_score": {"Object": {"mean": 0.8}}}
    reference_metrics = {"test_mini_score": {"Object": {"mean": 0.5}}}
    local = {"runs": ["local"], "agg_generalize": {"local": local_metrics}}
    reference = {
        "runs": ["local", "reference"],
        "agg_generalize": {
            "local": reference_metrics,
            "reference": reference_metrics,
        },
    }

    merged = merge_reference(local, reference)

    assert merged["agg_generalize"]["local"] == local_metrics
    assert merged["agg_generalize"]["reference"] == reference_metrics


def test_aggregate_atomic_skill_new_structure():
    records = [
        {
            "run_id": "run1",
            "task": "foo",
            "split": "test_mini",
            "seed": "1",
            "sr": 1.0,
            "score": 1.0,
            "metric_scores": [
                [
                    [1.0],
                    [0.0],
                ],
                [
                    [1.0],
                ],
            ],
        },
        {
            "run_id": "run1",
            "task": "foo",
            "split": "test_mini",
            "seed": "2",
            "sr": 0.0,
            "score": 0.0,
            "metric_scores": [
                [
                    [0.0],
                    [0.0],
                ],
                [
                    [1.0],
                ],
            ],
        },
    ]

    cluster_map = {
        "atomic_skill": {
            "foo": [
                [
                    [["Move"]],
                    [["Push"]],
                ],
                [
                    [["Grasp"]],
                ],
            ]
        }
    }

    payload = aggregate(records, cluster_map=cluster_map)
    atomic = payload["agg_cluster"]["run1"]["test_mini"]["atomic_skill"]

    assert atomic["Move"]["sr_mean"] == 0.5
    assert atomic["Move"]["n"] == 2
    assert atomic["Push"]["sr_mean"] == 0.0
    assert atomic["Push"]["n"] == 1
    assert atomic["Grasp"]["sr_mean"] == 1.0
    assert atomic["Grasp"]["n"] == 1
