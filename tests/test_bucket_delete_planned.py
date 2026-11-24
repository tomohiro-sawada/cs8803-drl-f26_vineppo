"""Every planned bucket deletion is checked against the live listing and backup."""

import pytest

from scripts.remote_sync import delete_planned

LISTING = {"kept": (4, "a"), "copy": (4, "a"), "smoke": (2, "b")}
BACKUP = {"smoke": (2, "b")}


def plan():
    def entry(tier, size, digest):
        return {"plan": tier, "size": size, "xet_hash": digest}

    return {
        "files": {
            "kept": entry("keep: frozen base weights", 4, "a"),
            "copy": entry("1 byte-identical copy of a kept file", 4, "a"),
            "smoke": entry("2 smoke weights", 2, "b"),
        },
        "kept_copy_of": {"copy": "kept"},
    }


def test_planned_paths_pass_when_live_state_matches():
    paths = ["copy", "smoke"]
    assert delete_planned.checked(plan(), paths, LISTING, BACKUP) == paths


@pytest.mark.parametrize(
    "path,listing,backup,message",
    [
        ("smoke", LISTING | {"smoke": (2, "c")}, BACKUP, "Bucket differs"),
        ("smoke", {"kept": (4, "a")}, BACKUP, "Bucket differs"),
        ("copy", {"copy": (4, "a")}, BACKUP, "No identical kept copy"),
        ("copy", LISTING | {"kept": (4, "z")}, BACKUP, "No identical kept copy"),
        ("smoke", LISTING, {}, "Not in the backup receipt"),
        ("smoke", LISTING, {"smoke": (2, "z")}, "Not in the backup receipt"),
        ("kept", LISTING, BACKUP, "Not a deletion tier"),
    ],
)
def test_any_mismatch_stops_before_deletion(path, listing, backup, message):
    with pytest.raises(AssertionError, match=message):
        delete_planned.checked(plan(), [path], listing, backup)


def test_copy_is_refused_when_its_kept_file_is_also_planned_for_deletion():
    broken = plan()
    broken["files"]["kept"]["plan"] = "2 smoke weights"
    with pytest.raises(AssertionError, match="Kept copy is planned for deletion"):
        delete_planned.checked(broken, ["copy"], LISTING, BACKUP)
