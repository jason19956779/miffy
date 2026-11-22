"""Unit and functional tests for pure-Python ISO-9660/Joliet parser and etiology classifier."""

import unittest
from pathlib import Path

from src.filesystem.iso9660 import (
    classify_damage_etiology,
    make_iso_sector_reader,
    parse_iso9660_tree,
)

REPO_DIR = Path(__file__).resolve().parents[2]
MIFFY_ISO = REPO_DIR / "recovered" / "MIFFY" / "miffy_recovered.iso"


class TestIso9660AndTriage(unittest.TestCase):
    def test_classify_clean_and_synthetic_damage_patterns(self):
        self.assertEqual(classify_damage_etiology([]).classification, "CLEAN_DISC")
        radial = classify_damage_etiology([50000 + i * 14 for i in range(60)])
        self.assertEqual(radial.classification, "RADIAL_SCRATCH")
        self.assertGreater(radial.single_turn_pitch_samples, 30)

        drm = classify_damage_etiology(list(range(800, 1500, 2)))
        self.assertEqual(drm.classification, "SUSPECTED_DRM")

    @unittest.skipUnless(MIFFY_ISO.exists(), "Recovered MIFFY ISO not present")
    def test_parse_miffy_iso9660_joliet(self):
        reader = make_iso_sector_reader(MIFFY_ISO)
        try:
            pvd, extents = parse_iso9660_tree(reader)
        finally:
            reader.fh.close()  # type: ignore[attr-defined]
        self.assertEqual(pvd["volume_id"], "_________MIFFY")
        self.assertEqual(pvd["volume_sectors"], 124079)
        self.assertEqual(pvd["encoding"], "utf-16-be")
        files = [e for e in extents if not e.is_dir]
        self.assertEqual(len(files), 1445)
        l_photos = [e for e in files if e.directory.upper() == "/PHOTO/L_PHOTO"]
        self.assertEqual(len(l_photos), 688)


if __name__ == "__main__":
    unittest.main()
