import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aligngemini.core import (
    VLM_SYSTEM_PROMPT, VLM_USER_PROMPT, fuse, parse_vlm_verdict,
    read_expert_results, read_inputs, read_vlm_results,
)
from predict import main, run_vlm


class CoreTests(unittest.TestCase):
    def test_vlm_manifest_uses_paper_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            base_model = root / "base_model"
            base_model.mkdir()
            (base_model / "config.json").write_text("{}", encoding="utf-8")
            image = str(root / "image.jpg")

            def fake_swift(command, check, env):
                self.assertTrue(check)
                self.assertEqual(command[:4], ["swift", "infer", "--adapters", str(adapter)])
                self.assertEqual(command[command.index("--max_pixels") + 1], "1003520")
                self.assertEqual(env["MAX_PIXELS"], "1003520")
                Path(command[command.index("--result_path") + 1]).touch()

            with patch("predict.subprocess.run", side_effect=fake_swift):
                run_vlm([{"id": "1", "image": image}], adapter, "swift", root, 1, base_model)
            row = json.loads((root / "vlm_input.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["messages"], [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {"role": "user", "content": VLM_USER_PROMPT},
            ])

    def test_verdict_and_fusion(self):
        self.assertFalse(parse_vlm_verdict("<answer>This is an authentic image.</answer>"))
        self.assertTrue(parse_vlm_verdict("<answer>This is an AI-generated image.</answer>"))
        self.assertTrue(parse_vlm_verdict("<rationale>Some text.</rationale>\nThis is an AI-generated image."))
        with self.assertRaises(ValueError):
            parse_vlm_verdict("This image might be AI-generated.")
        self.assertFalse(fuse(False, False))
        self.assertTrue(fuse(False, True))
        self.assertTrue(fuse(True, False))

    def test_vlm_result_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "image.jpg"
            image.touch()
            records = [{"id": "1", "image": str(image)}]
            result = root / "result.jsonl"
            for row in (
                {"images": [str(image)], "response": "<answer>This is an authentic image.</answer>"},
                {"image": str(image), "branch_response": "<answer>This is an authentic image.</answer>"},
            ):
                result.write_text(json.dumps(row) + "\n", encoding="utf-8")
                self.assertEqual(len(read_vlm_results(result, records)), 1)

    def test_generic_expert_interface_and_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = [root / f"{index}.jpg" for index in range(3)]
            for image in images:
                image.touch()
            manifest = root / "images.txt"
            manifest.write_text("\n".join(str(image) for image in images), encoding="utf-8")
            vlm_results = root / "vlm.jsonl"
            with vlm_results.open("w", encoding="utf-8") as stream:
                for image, fake in zip(images, (False, True, False)):
                    verdict = "AI-generated" if fake else "authentic"
                    stream.write(json.dumps({
                        "images": [str(image)],
                        "response": f"<answer>This is an {verdict} image.</answer>",
                    }) + "\n")
            expert_results = root / "expert.jsonl"
            with expert_results.open("w", encoding="utf-8") as stream:
                for image, value in zip(images, (0.1, False, 0.7)):
                    field = {"fake": value} if isinstance(value, bool) else {"fake_probability": value}
                    stream.write(json.dumps({"image": str(image), **field}) + "\n")
            output = root / "predictions.jsonl"
            args = [
                "predict.py", "--input", str(manifest), "--output", str(output),
                "--vlm-results", str(vlm_results), "--expert-results", str(expert_results),
            ]
            with patch.object(sys, "argv", args):
                main()
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["fake"] for row in rows], [False, True, True])
            self.assertEqual([row["vlm_fake"] for row in rows], [False, True, False])
            self.assertEqual([row["expert_fake"] for row in rows], [False, False, True])
            self.assertNotIn("expert_fake_probability", rows[1])

    def test_expert_results_validate_completeness_and_probability(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "image.jpg"
            image.touch()
            records = [{"id": "1", "image": str(image)}]
            result = root / "expert.jsonl"
            result.write_text(json.dumps({
                "image": str(image), "fake_probability": float("nan"),
            }) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_expert_results(result, records)
            result.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_expert_results(result, records)

    def test_expert_command_runs_through_same_interface(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "image.jpg"
            image.touch()
            manifest = root / "images.txt"
            manifest.write_text(str(image) + "\n", encoding="utf-8")
            vlm_results = root / "vlm.jsonl"
            vlm_results.write_text(json.dumps({
                "image": str(image),
                "response": "<answer>This is an authentic image.</answer>",
            }) + "\n", encoding="utf-8")
            runner = root / "expert.py"
            runner.write_text(
                "import argparse, json\n"
                "from pathlib import Path\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--input', required=True)\n"
                "p.add_argument('--output', required=True)\n"
                "a = p.parse_args()\n"
                "image = Path(a.input).read_text().strip()\n"
                "Path(a.output).write_text(json.dumps({'image': image, 'fake': True}) + '\\n')\n",
                encoding="utf-8",
            )
            output = root / "predictions.jsonl"
            command = shlex.join([sys.executable, str(runner)])
            with patch.object(sys, "argv", [
                "predict.py", "--input", str(manifest), "--output", str(output),
                "--vlm-results", str(vlm_results), "--expert-command", command,
            ]):
                main()
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(row["vlm_fake"])
            self.assertTrue(row["expert_fake"])
            self.assertTrue(row["fake"])

    def test_image_folder_with_vlm_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "images"
            nested = folder / "subfolder"
            nested.mkdir(parents=True)
            image = nested / "sample.JPG"
            image.touch()
            (folder / "ignore.txt").touch()
            records = read_inputs(folder)
            self.assertEqual([row["image"] for row in records], [str(image)])
            vlm_results = root / "vlm.jsonl"
            vlm_results.write_text(json.dumps({
                "image": str(image),
                "response": "<answer>This is an authentic image.</answer>",
            }) + "\n", encoding="utf-8")
            output = root / "predictions.jsonl"
            with patch.object(sys, "argv", [
                "predict.py", "--input", str(folder), "--output", str(output),
                "--vlm-results", str(vlm_results),
            ]):
                main()
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["image"], str(image))
            self.assertIsNone(row["expert_fake"])
            self.assertFalse(row["vlm_fake"])
            self.assertFalse(row["fake"])
            self.assertIn("<answer>", row["vlm_response"])


if __name__ == "__main__":
    unittest.main()
