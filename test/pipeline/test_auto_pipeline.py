"""Integration tests for session directory auto-naming and CLI entry point."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.pipeline.orchestrator import resolve_session_dir, sanitize_volume_id

REPO_DIR = Path(__file__).resolve().parents[2]
MIFFY_ISO = REPO_DIR / "recovered" / "MIFFY" / "miffy_recovered.iso"


class TestAutoPipeline(unittest.TestCase):
    def test_sanitize_volume_id(self):
        self.assertEqual(sanitize_volume_id("_________MIFFY   "), "MIFFY")
        self.assertEqual(sanitize_volume_id("Vacation 2004/CD#1"), "Vacation_2004_CD_1")
        self.assertEqual(sanitize_volume_id("___   "), "")

    @unittest.skipUnless(MIFFY_ISO.exists(), "Recovered MIFFY ISO not present")
    def test_resolve_session_dir_from_iso_and_uuid_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec_root = Path(tmp) / "recovered"
            name_pvd, sdir_pvd = resolve_session_dir(
                recovered_root=rec_root,
                no_hardware=True,
                iso_out=MIFFY_ISO,
            )
            self.assertEqual(name_pvd, "MIFFY")
            self.assertEqual(sdir_pvd, rec_root / "MIFFY")

            name_custom, sdir_custom = resolve_session_dir(
                recovered_root=rec_root,
                session_name="My Custom Disc",
                no_hardware=True,
            )
            self.assertEqual(name_custom, "My_Custom_Disc")
            self.assertEqual(sdir_custom, rec_root / "My_Custom_Disc")

            name_uuid, sdir_uuid = resolve_session_dir(
                recovered_root=rec_root,
                no_hardware=True,
            )
            self.assertRegex(name_uuid, r"^cd_[0-9a-f]{8}$")
            self.assertEqual(sdir_uuid, rec_root / name_uuid)

    def test_cli_list_plugins(self):
        out = subprocess.check_output(
            [sys.executable, str(REPO_DIR / "auto_cd_recover.py"), "--list-plugins"],
            cwd=str(REPO_DIR),
            text=True,
        )
        self.assertIn("joint_gf2_edc", out)
        self.assertIn("jpeg", out)


if __name__ == "__main__":
    unittest.main()
