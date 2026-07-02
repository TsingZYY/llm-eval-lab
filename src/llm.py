"""LLM 客户端封装。

设计要点（这是平台工程的核心思维）：
- 上层代码只依赖这个接口，换模型/换厂商只改这个文件和配置
- 没有 API key 时自动进入 mock 模式，保证 pipeline 随时可测试
  （生产系统里这叫 "可测试性"——不依赖外部服务也能跑通全流程）
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request


ZHIPU_CHAT_COMPLETIONS_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def provider_for_model(model: str) -> str:
    model_lower = model.lower()
    if model_lower.startswith("glm-"):
        return "zhipu"
    if model_lower.startswith("gpt-"):
        return "openai_compatible"
    return "anthropic"


def api_key_for_provider(provider: str) -> str | None:
    if provider == "zhipu":
        return os.environ.get("ZHIPU_API_KEY")
    if provider == "openai_compatible":
        return os.environ.get("CLIRELAY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return os.environ.get("ANTHROPIC_API_KEY")


def base_url_for_provider(provider: str, explicit_base_url: str | None = None) -> str | None:
    if explicit_base_url:
        return explicit_base_url.rstrip("/")
    if provider == "openai_compatible":
        base_url = os.environ.get("CLIRELAY_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        return base_url.rstrip("/") if base_url else None
    return None


def retry_delay_from_error(headers, error_body: str) -> int | None:
    retry_after = headers.get("Retry-After") if headers else None
    if retry_after:
        try:
            return max(0, int(float(retry_after)))
        except ValueError:
            pass

    try:
        payload = json.loads(error_body)
    except json.JSONDecodeError:
        return None

    error = payload.get("error", payload) if isinstance(payload, dict) else {}
    reset_seconds = error.get("reset_seconds") if isinstance(error, dict) else None
    if reset_seconds is None and isinstance(payload, dict):
        reset_seconds = payload.get("reset_seconds")
    if reset_seconds is None:
        return None

    try:
        return max(0, int(float(reset_seconds)))
    except (TypeError, ValueError):
        return None


class LLMClient:
    def __init__(
        self,
        model: str,
        mock: bool | None = None,
        max_retries: int = 3,
        timeout_s: int = 60,
        base_url: str | None = None,
    ):
        self.model = model
        self.provider = provider_for_model(model)
        self._api_key = api_key_for_provider(self.provider)
        self._base_url = base_url_for_provider(self.provider, base_url)
        self.mock = (not self._api_key) if mock is None else mock
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        if not self.mock:
            if self.provider == "anthropic":
                try:
                    import anthropic  # 延迟导入：mock 模式下无需安装
                except ImportError as exc:
                    raise RuntimeError(
                        "当前模型需要 anthropic 包。请运行 `pip install anthropic`，"
                        "或把 config.yaml 改回 glm-* 模型。"
                    ) from exc

                self._client = anthropic.Anthropic()

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """普通文本补全。"""
        if self.mock:
            # 用 prompt 内容生成确定性的假摘要，保证结果可复现
            digest = hashlib.md5(prompt.encode()).hexdigest()[:8]
            return f"[mock-{self.model}-{digest}] 这是一条模拟摘要，涵盖了原文的部分要点。"
        if self.provider == "zhipu":
            data = self._zhipu_chat_completion(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "thinking": {"type": "disabled"},
                }
            )
            return data["choices"][0]["message"]["content"]
        if self.provider == "openai_compatible":
            data = self._openai_compatible_chat_completion(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                }
            )
            return data["choices"][0]["message"]["content"]
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def complete_with_tool(self, prompt: str, tool: dict, max_tokens: int = 512) -> dict:
        """强制模型通过 tool call 返回结构化结果（LLM 工具调用的典型用法）。

        评估打分必须是结构化数据（不能靠正则去 parse 自然语言），
        所以这里用 tool_choice 强制模型调用指定工具。
        """
        if self.mock:
            properties = tool.get("input_schema", {}).get("properties", {})
            seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
            if "winner" in properties:
                winners = ["a", "b", "tie"]
                winner = winners[seed % len(winners)]
                return {
                    "winner": winner,
                    "margin": 0 if winner == "tie" else seed % 3 + 1,
                    "reasoning": "[mock] 模拟 pairwise 判断，仅用于测试 pipeline。",
                }
            return {
                "accuracy": seed % 3 + 3,        # 3-5
                "completeness": (seed // 7) % 3 + 3,
                "conciseness": (seed // 13) % 3 + 3,
                "reasoning": "[mock] 模拟评分，仅用于测试 pipeline。",
            }
        if self.provider == "zhipu":
            return self._zhipu_structured_completion(prompt, tool, max_tokens)
        if self.provider == "openai_compatible":
            return self._openai_compatible_structured_completion(prompt, tool, max_tokens)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return block.input
        raise RuntimeError("模型没有返回 tool call")

    def _zhipu_chat_completion(self, payload: dict) -> dict:
        return self._post_json(ZHIPU_CHAT_COMPLETIONS_URL, payload)

    def _post_json(self, url: str, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                should_retry = exc.code == 429 or 500 <= exc.code < 600
                if should_retry and attempt < self.max_retries:
                    delay_s = retry_delay_from_error(exc.headers, error_body) if exc.code == 429 else None
                    if delay_s is None:
                        delay_s = 2 ** (attempt - 1)
                    print(
                        f"[retry] HTTP {exc.code}; waiting {delay_s}s "
                        f"before retry {attempt + 1}/{self.max_retries}"
                    )
                    time.sleep(delay_s)
                    continue
                raise RuntimeError(f"LLM API error {exc.code}: {error_body}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))
                    continue
                raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc

        raise RuntimeError("LLM API request failed after retries")

    def _zhipu_structured_completion(self, prompt: str, tool: dict, max_tokens: int) -> dict:
        schema = tool["input_schema"]
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
        data = self._zhipu_chat_completion(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你必须只返回一个 JSON 对象，不要使用 Markdown，也不要添加额外解释。\n"
                            f"JSON 对象必须符合以下 schema：\n{schema_text}"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled"},
            }
        )
        content = data["choices"][0]["message"]["content"]
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"模型没有返回合法 JSON: {content}") from exc

        missing = set(schema.get("required", [])) - set(result)
        if missing:
            raise RuntimeError(f"模型返回 JSON 缺少字段: {sorted(missing)}")
        return result

    def _openai_compatible_chat_completion(self, payload: dict) -> dict:
        if not self._base_url:
            raise RuntimeError("OpenAI-compatible provider 需要 openai_base_url 或 OPENAI_BASE_URL/CLIRELAY_BASE_URL")
        return self._post_json(f"{self._base_url}/chat/completions", payload)

    def _openai_compatible_structured_completion(self, prompt: str, tool: dict, max_tokens: int) -> dict:
        schema = tool["input_schema"]
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
        data = self._openai_compatible_chat_completion(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你必须只返回一个 JSON 对象，不要使用 Markdown，也不要添加额外解释。\n"
                            f"JSON 对象必须符合以下 schema：\n{schema_text}"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
        )
        content = data["choices"][0]["message"]["content"]
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"模型没有返回合法 JSON: {content}") from exc

        missing = set(schema.get("required", [])) - set(result)
        if missing:
            raise RuntimeError(f"模型返回 JSON 缺少字段: {sorted(missing)}")
        return result
