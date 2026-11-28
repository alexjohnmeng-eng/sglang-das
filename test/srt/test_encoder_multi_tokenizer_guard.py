"""Guard: language-only + multiple tokenizer workers must not start.

The language-only tokenizer manager runs an ``EncoderBootstrapServer`` on
``--encoder-bootstrap-port``. With ``--tokenizer-worker-num > 1`` each worker is
a separate process that binds the same port, so the dynamic encoder registry is
forked (only one worker owns the listener). ``handle_encoder_disaggregation``
must reject that combination at argument-validation time.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.pd_disaggregation_hook import (
    _reject_multi_tokenizer_language_only,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


def _cfg(**overrides):
    cfg = dict(language_only=False, tokenizer_worker_num=1, encoder_urls=[])
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestEncoderMultiTokenizerGuard(CustomTestCase):
    def test_rejects_language_only_with_multiple_workers(self):
        with self.assertRaises(ValueError) as ctx:
            _reject_multi_tokenizer_language_only(
                _cfg(language_only=True, tokenizer_worker_num=2)
            )
        message = str(ctx.exception)
        self.assertIn(
            "language-only prefill with more than one tokenizer worker", message
        )
        self.assertIn("EncoderBootstrapServer registry", message)
        self.assertIn("--tokenizer-worker-num 1", message)

    def test_allows_static_encoder_urls_with_multiple_workers(self):
        # With a static --encoder-urls list every worker can route; only the
        # dynamic registry is forked, so this is a warning, not an error.
        with self.assertLogs("sglang.srt.arg_groups.pd_disaggregation_hook", "WARNING"):
            _reject_multi_tokenizer_language_only(
                _cfg(
                    language_only=True,
                    tokenizer_worker_num=2,
                    encoder_urls=["http://encoder:30005"],
                )
            )

    def test_allows_language_only_with_single_worker(self):
        _reject_multi_tokenizer_language_only(
            _cfg(language_only=True, tokenizer_worker_num=1)
        )

    def test_allows_multiple_workers_without_language_only(self):
        _reject_multi_tokenizer_language_only(
            _cfg(language_only=False, tokenizer_worker_num=2)
        )

    def test_server_args_expose_the_fields_read_by_the_guard(self):
        # The guard reads these off the resolving view; make sure the real
        # ServerArgs carries them with the expected defaults.
        args = ServerArgs(model_path="dummy")
        self.assertFalse(args.language_only)
        self.assertEqual(args.tokenizer_worker_num, 1)
        _reject_multi_tokenizer_language_only(args)


if __name__ == "__main__":
    unittest.main()
