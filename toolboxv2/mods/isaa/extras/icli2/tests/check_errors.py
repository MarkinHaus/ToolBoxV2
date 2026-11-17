"""errors.explain mit realistischen Fehlertexten (litellm, Anbieter, Engine)."""
import stub  # noqa: F401
from toolboxv2.mods.isaa.extras.icli2.errors import explain, render

OR_429 = ('litellm.RateLimitError: RateLimitError: OpenrouterException - {"error":{"message":"Rate limit '
          'exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day","code":429,'
          '"metadata":{"headers":{"X-RateLimit-Limit":"50"}}},"user_id":"u1"}')
GROQ_429 = ("litellm.RateLimitError: RateLimitError: GroqException - Rate limit reached for model `llama-3.3-70b` "
            "in organization `org_x` on tokens per minute (TPM): Limit 6000, Used 5000. Please try again in 7.66s.")
GEMINI_429 = ('litellm.RateLimitError: litellm.RateLimitError: VertexAIException - {"error": {"code": 429, '
              '"message": "You exceeded your current quota.", "status": "RESOURCE_EXHAUSTED", '
              '"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "31s"}]}}')
AUTH = ("litellm.AuthenticationError: AuthenticationError: OpenAIException - Incorrect API key provided: sk-abc***. "
        "You can find your API key at https://platform.openai.com/account/api-keys.")
NOTFOUND = 'litellm.NotFoundError: NotFoundError: OpenrouterException - {"error":{"message":"No endpoints found for foo/bar.","code":404}}'
CTX = ("litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - "
       "This model's maximum context length is 128000 tokens. However, your messages resulted in 130512 tokens.")
CONN = "litellm.APIConnectionError: APIConnectionError: OpenAIException - Connection error."
TIMEOUT = "litellm.Timeout: APITimeoutError - Request timed out. Error_str: Request timed out."
OVER = 'litellm.ServiceUnavailableError: AnthropicException - {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'
CREDITS = 'litellm.APIError: OpenrouterException - {"error":{"message":"This request requires more credits, or fewer max_tokens.","code":402}}'
BAD = 'litellm.BadRequestError: GroqException - {"error":{"message":"tool call validation failed: parameters for tool x did not match schema","type":"invalid_request_error"}}'
EMPTY = "LLM lieferte 3x in Folge leere Response. Abbruch."
CRASH = "Agent crashed: KeyError: 'tool_calls'"
TB = 'Traceback (most recent call last):\n  File "x.py", line 3, in f\nKeyError: \'tool_calls\''

cases = {
    "or429": (OR_429, "rate_limit", "Openrouter", "Rate limit exceeded: free-models-per-day"),
    "groq": (GROQ_429, "rate_limit", "Groq", "Rate limit reached for model"),
    "gemini": (GEMINI_429, "rate_limit", "Vertexai", "You exceeded your current quota."),
    "auth": (AUTH, "auth", "Openai", "Incorrect API key provided"),
    "notfound": (NOTFOUND, "model", "Openrouter", "No endpoints found for foo/bar."),
    "ctx": (CTX, "context", "Openai", "This model's maximum context length"),
    "conn": (CONN, "connection", "Openai", "Connection error."),
    "timeout": (TIMEOUT, "timeout", "", "Request timed out"),
    "over": (OVER, "overloaded", "Anthropic", "Overloaded"),
    "credits": (CREDITS, "credits", "Openrouter", "This request requires more credits"),
    "bad": (BAD, "bad_request", "Groq", "tool call validation failed"),
    "empty": (EMPTY, "empty", "", "LLM lieferte 3x"),
    "crash": (CRASH, "crash", "", "KeyError: 'tool_calls'"),
}
for name, (raw, kind, provider, msg) in cases.items():
    info = explain(raw, model="openrouter/foo/bar")
    assert info["kind"] == kind, (name, info["kind"])
    assert info["provider"].lower() == provider.lower(), (name, info["provider"])
    assert info["message"].startswith(msg) or msg in info["message"], (name, info["message"])
    assert info["hint"] and info["title"] and raw.split(" ")[0] in info["detail"], name
    text = render(info)
    assert info["hint"] in text and (("Details:" in text) == (info["detail"] != info["message"])), name

assert explain(GROQ_429)["retry"] == "8 s"
assert explain(GEMINI_429)["retry"] == "31 s"
assert "8 s warten" in explain(GROQ_429)["hint"]
assert explain(OR_429)["title"] == "Anbieter-Limit erreicht (429)"
assert '"code": 429' in explain(OR_429)["detail"]                       # JSON lesbar eingerueckt
assert explain(AUTH)["title"] == "Zugang abgelehnt" and "openai" in explain(AUTH)["hint"].lower()
assert "foo/bar" in explain(NOTFOUND, model="openrouter/foo/bar")["hint"]
lim = explain("max_iterations", kind="limit", iters=(30, 30))
assert lim["title"] == "Iterationslimit erreicht (30/30)" and "/set max_iterations" in lim["hint"] and lim["message"] == ""
crash = explain(CRASH, traceback=TB)
assert crash["kind"] == "crash" and "Traceback" in crash["detail"]
unk = explain("etwas Seltsames ist passiert")
assert unk["kind"] == "unknown" and unk["message"] == "etwas Seltsames ist passiert"
assert explain("")["kind"] == "unknown"
huge = explain("x" * 20000)
assert len(huge["detail"]) < 8200 and "Zeichen)" in huge["detail"]
print("errors ok")
