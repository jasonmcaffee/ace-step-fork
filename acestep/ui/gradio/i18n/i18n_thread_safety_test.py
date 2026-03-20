"""Thread safety tests for the i18n singleton."""

import threading
import unittest

from acestep.ui.gradio.i18n.i18n import get_i18n, _i18n_lock


class I18nThreadSafetyTests(unittest.TestCase):
    """Verify get_i18n() returns a single instance under concurrent access."""

    def test_concurrent_get_i18n_returns_same_instance(self):
        """Multiple threads calling get_i18n() must all receive the same object."""
        results: list[int] = []
        errors: list[str] = []
        barrier = threading.Barrier(8)

        def _call_get_i18n() -> None:
            """Fetch singleton after barrier synchronization."""
            try:
                barrier.wait(timeout=5)
                results.append(id(get_i18n()))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"get_i18n failed: {exc!r}")

        threads = [threading.Thread(target=_call_get_i18n) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread failure detected: {errors}")
        self.assertEqual(len(set(results)), 1, "get_i18n() returned different instances across threads")

    def test_lock_attribute_exists(self):
        """Module-level lock must be present."""
        self.assertIsInstance(_i18n_lock, type(threading.Lock()))


if __name__ == "__main__":
    unittest.main()
