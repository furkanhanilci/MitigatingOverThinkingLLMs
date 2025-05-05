import unittest
import os
import shutil
import torch
from tempfile import TemporaryDirectory


class TestRotateCleanCheckpoint(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to simulate checkpoint storage
        self.temp_dir = TemporaryDirectory()
        self.ckpt_dir = self.temp_dir.name

    def tearDown(self):
        # Cleanup temporary directory
        self.temp_dir.cleanup()

    def create_checkpoints(self, num_checkpoints):
        checkpoint_dirs = []
        for i in range(1, num_checkpoints + 1):
            ckpt_subdir = os.path.join(self.ckpt_dir, f"global_step_{i}")
            os.makedirs(ckpt_subdir)
            checkpoint_dirs.append(ckpt_subdir)

            # Create a dummy optimizer state file
            with open(os.path.join(ckpt_subdir, "optimizer_state.pth"), "w") as f:
                f.write("dummy optimizer state")
        return checkpoint_dirs

    def test_rotate_clean_checkpoint(self):
        from verl.trainer.ppo.trainer_utils import (
            rotate_clean_checkpoint,
        )  # Adjust import based on actual module location

        self.create_checkpoints(5)
        keep_num = 2  # Number of checkpoints to retain
        rotate_clean_checkpoint(self.ckpt_dir, keep_num=keep_num, clean_optimizer=True)

        # List remaining checkpoint directories
        remaining_checkpoints = sorted(
            [d for d in os.listdir(self.ckpt_dir) if d.startswith("global_step_")]
        )

        # Verify that only `keep_num` checkpoints remain
        self.assertEqual(len(remaining_checkpoints), keep_num)
        expected_checkpoints = [f"global_step_{i}" for i in range(4, 6)]
        self.assertListEqual(remaining_checkpoints, expected_checkpoints)

        # Verify optimizer state files are deleted from older checkpoints
        for old_ckpt in range(1, 4):
            optimizer_path = os.path.join(
                self.ckpt_dir, f"global_step_{old_ckpt}", "optimizer_state.pth"
            )
            self.assertFalse(os.path.exists(optimizer_path))

        # Verify optimizer state file is kept in latest checkpoint
        latest_optimizer_path = os.path.join(
            self.ckpt_dir, "global_step_5", "optimizer_state.pth"
        )
        self.assertTrue(os.path.exists(latest_optimizer_path))

    def test_rotate_clean_checkpoint_fewer_than_keep_num(self):
        from verl.trainer.ppo.trainer_utils import (
            rotate_clean_checkpoint,
        )  # Adjust import based on actual module location

        self.create_checkpoints(2)
        keep_num = 5  # Number of checkpoints to retain
        rotate_clean_checkpoint(self.ckpt_dir, keep_num=keep_num, clean_optimizer=True)

        # List remaining checkpoint directories
        remaining_checkpoints = sorted(
            [d for d in os.listdir(self.ckpt_dir) if d.startswith("global_step_")]
        )

        # Verify that no checkpoints were deleted
        self.assertEqual(len(remaining_checkpoints), 2)
        expected_checkpoints = ["global_step_1", "global_step_2"]
        self.assertListEqual(remaining_checkpoints, expected_checkpoints)

    def test_rotate_clean_checkpoint_no_checkpoints(self):
        from verl.trainer.ppo.trainer_utils import (
            rotate_clean_checkpoint,
        )  # Adjust import based on actual module location

        keep_num = 3  # Number of checkpoints to retain
        rotate_clean_checkpoint(self.ckpt_dir, keep_num=keep_num, clean_optimizer=True)

        # List remaining checkpoint directories
        remaining_checkpoints = sorted(
            [d for d in os.listdir(self.ckpt_dir) if d.startswith("global_step_")]
        )

        # Verify that no checkpoints exist
        self.assertEqual(len(remaining_checkpoints), 0)


if __name__ == "__main__":
    unittest.main()
