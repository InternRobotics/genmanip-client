from genmanip_client.extensions.analyse import aggregate


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
