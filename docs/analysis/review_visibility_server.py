"""Review downloaded server diagnostics against the saved local audit.

No training/inference. Writes only a new directory; original downloads unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from analyze_tx_xyz import read_json
from experiment_provenance import fingerprint


def same_values(server, local, where="root"):
    if isinstance(local, dict):
        assert set(server) == set(local), where
        for key in local:
            same_values(server[key],local[key],where+"/"+key)
    elif isinstance(local,list):
        assert len(server)==len(local),where
        for i,(a,b) in enumerate(zip(server,local)):
            same_values(a,b,where+f"/{i}")
    elif isinstance(local,(int,float)) and not isinstance(local,bool):
        np.testing.assert_allclose(server,local,rtol=1e-12,atol=1e-8,err_msg=where)
    else:
        assert server==local,where


def verify_scripts(entries, repo, ref):
    checks=[]
    for entry in entries:
        relative=entry["path"].split("/docs/",1)[1]
        name="docs/"+relative
        blob=subprocess.run(["git","show",f"{ref}:{name}"],cwd=repo,capture_output=True,check=True,timeout=10).stdout
        assert hashlib.sha256(blob).hexdigest()==entry["sha256"],name
        checks.append({"file":name,"sha256":entry["sha256"],"matches_git_ref":ref})
    return checks


def analyze(source,out,ref):
    repo=Path(__file__).resolve().parents[2]
    analysis=repo/"docs/analysis"
    out.mkdir(parents=True,exist_ok=False)
    names={"replication":"tx_xyz_replication_audit_20261005/replication_audit.json",
           "probe":"visibility_probe_20261005/visibility_probe.json",
           "cuda":"visibility_cuda_20261005/cuda_benchmark.json"}
    server={key:read_json(source/name) for key,name in names.items()}
    local={"replication":read_json(analysis/"tx_xyz_replication_20261005/replication_audit.json"),
           "probe":read_json(analysis/names["probe"]),"cuda":read_json(analysis/names["cuda"])}
    for key in ("paired_seeds","descriptive_statistics","terrain_influence","fixed_profiles","cached_eight_map_diagnostics","commits","cross_seed_code_data_environment_initialization_checks"):
        same_values(server["replication"][key],local["replication"][key],key)
    for key in ("summary","config","checks"):
        same_values(server["probe"][key],local["probe"][key],key)
    hashes=[]
    for key in ("probe","cuda"):
        a=server[key]["source_fingerprints"]
        b=local[key]["source_fingerprints"]
        assert [(x["bytes"],x["sha256"]) for x in a]==[(x["bytes"],x["sha256"]) for x in b]
        for entry in a:
            if entry["path"].endswith("metadata.json"):
                path=repo.parent/"LunarRM/metadata.json"
            else:
                path=source/entry["path"].split("/runs/",1)[1]
            assert fingerprint(path)["sha256"]==entry["sha256"]
            hashes.append({"local_file":str(path),"sha256":entry["sha256"]})
    for seed in range(3):
        a=read_json(source/f"tx_xyz_replication_audit_20261005/seed{seed}/audit.json")
        b=read_json(analysis/f"tx_xyz_replication_20261005/seed{seed}/audit.json")
        assert sorted((x["bytes"],x["sha256"]) for x in a["source_fingerprints"])==sorted((x["bytes"],x["sha256"]) for x in b["source_fingerprints"])
    scripts=[]
    scripts+=verify_scripts(server["replication"]["analysis_scripts"],repo,ref)
    scripts+=verify_scripts([server["probe"]["script"]],repo,ref)
    scripts+=verify_scripts(server["cuda"]["scripts"],repo,ref)
    maps=server["cuda"]["maps"]
    assert len(maps)==8 and len({row["sample"] for row in maps})==8
    expected=local["cuda"]["maps"]
    assert [row["sample"] for row in maps]==[row["sample"] for row in expected]
    for row in maps:
        assert len(row["seconds_raw"])==3 and min(row["seconds_raw"])>0
        np.testing.assert_allclose(np.mean(row["seconds_raw"]),row["seconds_mean"],rtol=1e-14)
    seconds=np.array([x for row in maps for x in row["seconds_raw"]])
    np.testing.assert_allclose(seconds.mean(),server["cuda"]["mean_seconds_per_map"],rtol=1e-14)
    first=maps[0]
    assert first["fp64_reference_max_abs_error_m"]<0.001 and first["visibility_disagreements"]==0
    means=np.array([row["seconds_mean"] for row in maps])
    # Arithmetic projection of the measured narrow scope, not end-to-end timing.
    online_train_seconds=seconds.mean()*25720*150
    one_time_train_val_seconds=seconds.mean()*(25720+2330)
    result={"status":"verified","replication_and_geometry_match":"passed",
            "script_checks":scripts,"source_checks":hashes,
            "server_recorded_gpu":server["cuda"]["gpu"],"server_recorded_torch":server["cuda"]["torch"],
            "timing":{"maps":8,"repeats_per_map":3,"mean_ms_per_map":float(1000*seconds.mean()),
                      "map_mean_range_ms":[float(1000*means.min()),float(1000*means.max())],
                      "raw_repeat_range_ms":[float(1000*seconds.min()),float(1000*seconds.max())],
                      "local_gpu":local["cuda"]["gpu"],"local_mean_ms":1000*local["cuda"]["mean_seconds_per_map"]},
            "numerical_check":{"real_maps_compared_to_fp64":1,"max_abs_error_m":first["fp64_reference_max_abs_error_m"],"visibility_disagreements":0},
            "arithmetic_projections":{"online_150_epoch_train_feature_hours":online_train_seconds/3600,
                                      "one_time_train_val_feature_minutes":one_time_train_val_seconds/60,
                                      "fp32_train_val_cache_gb":(25720+2330)*256*256*4/1e9},
            "timing_scope":server["cuda"]["scope"],
            "limits":"Recorded server timing from downloaded artifacts only, not live execution. Eight maps/three timed repetitions, one real-map GPU-vs-FP64 check. Projections reuse batch1 feature timing; exclude IO, output transfer, F scaling, model and training; not runtime estimates or full-data acceptance. No clearance training result.",
            "incorrect_static_note_in_download":server["cuda"]["limits"],
            "source_artifacts":[fingerprint(path) for key,name in names.items() for path in [(source/name)]],
            "analysis_script":fingerprint(Path(__file__))}
    (out/"server_review.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","server_recorded_gpu","timing","numerical_check","arithmetic_projections","limits")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--script-ref",default="97660e8")
    args=parser.parse_args()
    analyze(args.source,args.out,args.script_ref)
