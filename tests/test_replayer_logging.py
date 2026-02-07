import unittest
from unittest.mock import patch, MagicMock, ANY
from xrtm.train.simulation.replayer import TraceReplayer

class TestTraceReplayerLogging(unittest.TestCase):
    @patch("xrtm.train.simulation.replayer.logger")
    @patch("builtins.open", new_callable=MagicMock)
    def test_save_trace_logging(self, mock_open, mock_logger):
        state = MagicMock()
        state.model_dump_json.return_value = "{}"
        path = "test.json"

        # Mock the file context manager
        mock_open.return_value.__enter__.return_value.write.return_value = None

        TraceReplayer.save_trace(state, path)

        # Verify call args
        # This assertion checks for deferred interpolation
        mock_logger.info.assert_called_with("Trace saved to %s", path)

    @patch("xrtm.train.simulation.replayer.logger")
    @patch("builtins.open", new_callable=MagicMock)
    def test_save_trace_error_logging(self, mock_open, mock_logger):
        state = MagicMock()
        error_msg = "Dump failed"
        state.model_dump_json.side_effect = Exception(error_msg)
        path = "test.json"

        with self.assertRaises(Exception):
            TraceReplayer.save_trace(state, path)

        # Verify error logging
        # Note: The exception object itself is passed as the second arg
        # We check that the format string is correct
        args, _ = mock_logger.error.call_args
        self.assertEqual(args[0], "Failed to save trace to %s: %s")
        self.assertEqual(args[1], path)
        self.assertIsInstance(args[2], Exception)

    @patch("xrtm.train.simulation.replayer.logger")
    @patch("builtins.open", new_callable=MagicMock)
    def test_load_trace_error_logging(self, mock_open, mock_logger):
        mock_open.side_effect = Exception("Read failed")
        path = "test.json"

        with self.assertRaises(Exception):
            TraceReplayer.load_trace(path)

        args, _ = mock_logger.error.call_args
        self.assertEqual(args[0], "Failed to load trace from %s: %s")
        self.assertEqual(args[1], path)
        self.assertIsInstance(args[2], Exception)

if __name__ == "__main__":
    unittest.main()
