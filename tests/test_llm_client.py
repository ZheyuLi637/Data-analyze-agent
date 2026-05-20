import tempfile
import unittest
from pathlib import Path

from agent.llm_client import LLMConfig


class LLMClientTest(unittest.TestCase):
    def test_loads_openrouter_config_from_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "LLM_ENABLED=true",
                        "LLM_API_KEY=test-key",
                        "LLM_BASE_URL=https://openrouter.ai/api/v1",
                        "LLM_MODEL=openrouter/owl-alpha",
                        "LLM_APP_NAME=COMPSCI 767 Data Analysis Agent",
                    ]
                ),
                encoding="utf-8",
            )

            config = LLMConfig.from_env(env_file)

            self.assertTrue(config.ready)
            self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
            self.assertEqual(config.model, "openrouter/owl-alpha")


if __name__ == "__main__":
    unittest.main()

