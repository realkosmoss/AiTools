from curl_cffi import requests
import time
import json
import random
import secrets

class Cloudflare:
    class Random:
        @staticmethod
        def gR():
            e = int(time.time() * 1000)
            t = int(time.perf_counter() * 1_000_000)

            result = []
            template = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx"

            for char in template:
                if char in {'x', 'y'}:
                    r = random.random() * 16

                    if e > 0:
                        r = int((e + r) % 16)
                        e //= 16
                    else:
                        r = int((t + r) % 16)
                        t //= 16

                    if char == 'x':
                        val = r
                    else:
                        val = (r & 3) | 8

                    result.append(f"{val:x}")
                else:
                    result.append(char)

            return "".join(result)
        
        @staticmethod
        def Dv(e=21):
            AR = "useandom-26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict";
            t = ""
            e |= 0
            n = [secrets.randbits(8) for _ in range(e)]
            while e:
                e -= 1
                t += AR[n[e] & 63]

            return t
        
    class _Messages:
        @staticmethod
        def _convert_messages(messages: list) -> list:
            _cf_msgs = []
            for m in messages:
                _cf_msgs.append({
                    "id": Cloudflare.Random.Dv(16),
                    "role": m["role"],
                    "parts": [
                        {
                            "type": "text",
                            "text": m["content"]
                        }
                    ]
                })

            return _cf_msgs
        
    class Models:
        GPT_OSS_120B = "@cf/openai/gpt-oss-120b"
        QWEN_1_5_0_5B_CHAT = "@cf/qwen/qwen1.5-0.5b-chat"
        BAAI_BGE_M3 = "@cf/baai/bge-m3"
        GEMMA_2B_IT_LORA = "@cf/google/gemma-2b-it-lora"
        STARLING_7B_BETA = "@hf/nexusflow/starling-lm-7b-beta"
        LLAMA3_8B_INSTRUCT = "@cf/meta/llama-3-8b-instruct"
        LLAMA3_2_3B_INSTRUCT = "@cf/meta/llama-3.2-3b-instruct"
        LLAMAGUARD_7B_AWQ = "@hf/thebloke/llamaguard-7b-awq"
        NEURAL_CHAT_7B_AWQ = "@hf/thebloke/neural-chat-7b-v3-1-awq"
        LLAMA_GUARD_3_8B = "@cf/meta/llama-guard-3-8b"
        LLAMA2_7B_FP16 = "@cf/meta/llama-2-7b-chat-fp16"
        MISTRAL_7B_V0_1 = "@cf/mistral/mistral-7b-instruct-v0.1"
        MISTRAL_7B_V0_2_LORA = "@cf/mistral/mistral-7b-instruct-v0.2-lora"
        PLAMO_EMBED_1B = "@cf/pfnet/plamo-embedding-1b"
        UNA_CYBERTRON_7B = "@cf/fblgit/una-cybertron-7b-v2-bf16"
        DEEPSEEK_R1_32B = "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"
        LLAMA2_7B_INT8 = "@cf/meta/llama-2-7b-chat-int8"
        LLAMA3_1_8B_FP8 = "@cf/meta/llama-3.1-8b-instruct-fp8"
        QWEN1_5_7B_AWQ = "@cf/qwen/qwen1.5-7b-chat-awq"
        LLAMA3_2_1B = "@cf/meta/llama-3.2-1b-instruct"
        LLAMA3_3_70B_FP8 = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
        GRANITE_4_MICRO = "@cf/ibm-granite/granite-4.0-h-micro"
        INDIC_TRANS_1B = "@cf/ai4bharat/indictrans2-en-indic-1B"
        QWEN2_5_CODER_32B = "@cf/qwen/qwen2.5-coder-32b-instruct"
        DEEPSEEK_MATH_7B = "@cf/deepseek-ai/deepseek-math-7b-instruct"
        FALCON_7B = "@cf/tiiuae/falcon-7b-instruct"
        GEMMA_SEA_LION_27B = "@cf/aisingapore/gemma-sea-lion-v4-27b-it"
        QWEN3_30B = "@cf/qwen/qwen3-30b-a3b-fp8"
        GEMMA_7B_LORA = "@cf/google/gemma-7b-it-lora"
        QWEN1_5_1_8B = "@cf/qwen/qwen1.5-1.8b-chat"
        MISTRAL_SMALL_24B = "@cf/mistralai/mistral-small-3.1-24b-instruct"
        LLAMA3_2_11B_VISION = "@cf/meta/llama-3.2-11b-vision-instruct"
        SQLCODER_7B = "@cf/defog/sqlcoder-7b-2"
        PHI_2 = "@cf/microsoft/phi-2"
        GPT_OSS_20B = "@cf/openai/gpt-oss-20b"
        QWEN1_5_14B_AWQ = "@cf/qwen/qwen1.5-14b-chat-awq"
        OPENCHAT_3_5 = "@cf/openchat/openchat-3.5-0106"
        LLAMA4_SCOUT_17B = "@cf/meta/llama-4-scout-17b-16e-instruct"
        GEMMA_3_12B = "@cf/google/gemma-3-12b-it"
        QWQ_32B = "@cf/qwen/qwq-32b"
        
    def __init__(self, session: requests.Session):
        self.session = session
        self.ws: requests.WebSocket

        self.last_model: str = None
        # chat variables
        self._pk = self.Random.gR()
        self._room_id = self.Random.Dv()
        # url construction
        self.base_url = "playground.ai.cloudflare.com/agents/playground"
        self.room_name = f"Cloudflare-AI-Playground-{self._room_id}"
        self.ws_url = f"wss://{self.base_url}/{self.room_name}?_pk={self._pk}"

        self._emulate_page_load()

    def _emulate_page_load(self):
        self.session.get("https://playground.ai.cloudflare.com/")
        _temp_headers = {
            "referer": "https://playground.ai.cloudflare.com/",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin"
        }
        _temp_headers = {**self.session.headers, **_temp_headers}
        self.session.get(f"https://{self.base_url}/{self.room_name}/get-messages", headers=_temp_headers)

        self.ws = self.session.ws_connect(
            self.ws_url,
            headers={
                "Origin": "https://playground.ai.cloudflare.com",
                "User-Agent": self.session.headers.get("user-agent"),
                "Sec-GPC": "1",
                "Pragma": "no-cache",
            }
        )
        self.ws.send_json({"args":[],"id":f"{self.Random.Dv(8)}","method":"getModels","type":"rpc"})

    def _change_model(self, model):
        # send it and pray
        self.ws.send_json({
            "type": "cf_agent_state",
            "state": {
                "model": model,
                "temperature": 1,
                "stream": True,
                "system": "You are a helpful assistant."
            }
        })

    def _parse_stream(self):
        def _iter_json_objects(s: str):
            decoder = json.JSONDecoder()
            idx = 0
            length = len(s)

            while idx < length:
                # skip whitespace
                while idx < length and s[idx].isspace():
                    idx += 1
                if idx >= length:
                    break

                try:
                    obj, idx = decoder.raw_decode(s, idx)
                except json.JSONDecodeError:
                    break
                yield obj
        reasoning = []
        text = []

        while True:
            frame, opcode = self.ws.recv()

            if opcode != 1:
                continue

            msg = json.loads(frame)

            if msg.get("type") != "cf_agent_use_chat_response":
                continue

            if msg.get("done"):
                break

            body = msg.get("body")
            if not body:
                continue

            for inner in _iter_json_objects(body):
                t = inner.get("type")

                if t == "reasoning-delta":
                    reasoning.append(inner.get("delta", ""))

                elif t == "text-delta":
                    text.append(inner.get("delta", ""))

        return {
            "reasoning": "".join(reasoning).strip(),
            "text": "".join(text).strip(),
        }

    def generate(self, messages: list, model: str):
        if not self.last_model:
            self.last_model = model
            self._change_model(model)
        else:
            if not self.last_model == model:
                self._change_model(model)
                self.last_model = model

        _request_id = self.Random.Dv(8)
        _messages = self._Messages._convert_messages(messages)
        _base = {
            "id": _request_id,
            "type": "cf_agent_use_chat_request",
            "url": f"https://playground.ai.cloudflare.com/agents/playground/{self.room_name}",
            "init": {
                "method": "POST",
                "headers": {
                    "content-type": "application/json",
                    "user-agent": "ai-sdk/5.0.116 runtime/browser"
                },
                "body": json.dumps({
                    "id": self.Random.gR(),
                    "messages": _messages,
                    "trigger": "submit-message"
                })
            }
        }

        self.ws.send_json(_base)

        _out = self._parse_stream()
        return _out["text"]