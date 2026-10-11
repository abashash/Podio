import unittest
from make_podcast import parse_script_directives


class TestDirectives(unittest.TestCase):
    def test_music_prompt_directive(self):
        """Test parsing of #MUSIC_PROMPT directive and duplicate rejection."""
        lines = ["#MUSIC_PROMPT: Instrumental only, warm Persian-inspired background music."]
        res = parse_script_directives(lines, 10)
        self.assertEqual(res["music_prompt"], "Instrumental only, warm Persian-inspired background music.")
        self.assertEqual(len(res["errors"]), 0)

        # Duplicate prompt rejection
        dup_lines = [
            "#MUSIC_PROMPT: First prompt",
            "#MUSIC_PROMPT: Second prompt",
        ]
        res_dup = parse_script_directives(dup_lines, 10)
        self.assertTrue(any("Only one" in err for err in res_dup["errors"]))

    def test_music_placement_directive(self):
        """Test parsing of #MUSIC placement directive with anchors, clips, volume, and ducking."""
        lines = [
            "#MUSIC: anchor=start clip=0-10 volume=-18 fade=1.5,2 duck=on",
            "#MUSIC: anchor=end clip=20-30 volume=-18 fade=0,4 duck=off",
        ]
        res = parse_script_directives(lines, 10)
        self.assertEqual(len(res["music_directives"]), 2)

        start_dir = res["music_directives"][0]
        self.assertEqual(start_dir["anchor_type"], "start")
        self.assertEqual(start_dir["clip_from"], 0.0)
        self.assertEqual(start_dir["clip_to"], 10.0)
        self.assertEqual(start_dir["volume_db"], -18.0)
        self.assertEqual(start_dir["fade_in"], 1.5)
        self.assertEqual(start_dir["fade_out"], 2.0)
        self.assertTrue(start_dir["duck_speech"])

        end_dir = res["music_directives"][1]
        self.assertEqual(end_dir["anchor_type"], "end")
        self.assertEqual(end_dir["clip_from"], 20.0)
        self.assertEqual(end_dir["clip_to"], 30.0)
        self.assertFalse(end_dir["duck_speech"])

    def test_music_mode_directive(self):
        """Test parsing of #MUSIC_MODE directives across block ranges."""
        lines = [
            "#MUSIC_MODE: blocks=1-3 mode=duck",
            "#MUSIC_MODE: blocks=4 mode=silent",
        ]
        res = parse_script_directives(lines, 10)
        self.assertEqual(len(res["music_modes"]), 2)

        mode1 = res["music_modes"][0]
        self.assertEqual(mode1["start_block"], 1)
        self.assertEqual(mode1["end_block"], 3)
        self.assertEqual(mode1["mode"], "duck")

        mode2 = res["music_modes"][1]
        self.assertEqual(mode2["start_block"], 4)
        self.assertEqual(mode2["end_block"], 4)
        self.assertEqual(mode2["mode"], "silent")

    def test_invalid_directives_and_comments(self):
        """Test rejection of time-based placements and ignoring plain comments."""
        lines = [
            "# Plain comment that should be ignored",
            "#MUSIC: at=5s clip=0-10 volume=-18 fade=1,1 duck=on",
            "# Another comment",
        ]
        res = parse_script_directives(lines, 10)
        self.assertEqual(len(res["music_directives"]), 0)
        self.assertTrue(any("Time-based placement is not supported" in err for err in res["errors"]))


if __name__ == "__main__":
    unittest.main()
