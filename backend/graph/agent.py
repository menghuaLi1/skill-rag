from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:  # pragma: no cover - optional dependency at runtime
    ChatDeepSeek = None

from config import get_settings, runtime_config
from graph.experience_memory import experience_memory
from graph.memory_indexer import memory_indexer
from graph.prompt_builder import build_system_prompt
from graph.session_manager import SessionManager
from knowledge_retrieval import knowledge_orchestrator
from tools import get_all_tools

KNOWLEDGE_SKILL_PATTERNS = (
    re.compile(r"知识库"),
    re.compile(r"\bknowledge\b", re.IGNORECASE),
    re.compile(r"根据.+?(知识库|文档|资料)"),
    re.compile(r"(查|检索).+?(文档|资料|报告|白皮书)"),
    re.compile(r"\.(pdf|xlsx|xls|json)\b", re.IGNORECASE),
)

MAX_AGENT_TOOL_CALLS = 24
MAX_AGENT_SAME_TOOL_STREAK = 6
MAX_AGENT_UPDATE_EVENTS = 800
SHORT_TERM_WINDOW_MESSAGES = 12


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


class AgentManager:
    def __init__(self) -> None:
        self.base_dir: Path | None = None
        self.session_manager: SessionManager | None = None
        self.tools = []

    def initialize(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.session_manager = SessionManager(base_dir)
        self.tools = get_all_tools(base_dir)
        experience_memory.configure(base_dir)
        knowledge_orchestrator.configure(base_dir, self._build_chat_model)

    def _build_chat_model(self):
        settings = get_settings()

        if settings.llm_provider == "deepseek":
            if ChatDeepSeek is None:
                raise RuntimeError("langchain-deepseek is not installed")
            if not settings.llm_api_key:
                raise RuntimeError("Missing API key for provider deepseek")
            return ChatDeepSeek(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                temperature=0,
            )

        if not settings.llm_api_key:
            raise RuntimeError(f"Missing API key for provider {settings.llm_provider}")

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0,
        )

    def _build_agent(
        self,
        extra_instructions: list[str] | None = None,
        tools_override: list[Any] | None = None,
    ):
        if self.base_dir is None:
            raise RuntimeError("AgentManager is not initialized")

        system_prompt = build_system_prompt(self.base_dir, runtime_config.get_rag_mode())
        if extra_instructions:
            system_prompt = f"{system_prompt}\n\n" + "\n\n".join(extra_instructions)
        return create_agent(
            model=self._build_chat_model(),
            tools=self.tools if tools_override is None else tools_override,
            system_prompt=system_prompt,
        )

    def _is_knowledge_query(self, message: str) -> bool:
        return any(pattern.search(message) for pattern in KNOWLEDGE_SKILL_PATTERNS)

    def _build_messages(self, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for item in history:
            role = item.get("role")
            if role not in {"user", "assistant"}:
                continue
            messages.append({"role": role, "content": str(item.get("content", ""))})
        return messages

    def _format_retrieval_context(self, results: list[dict[str, Any]]) -> str:
        lines = ["[RAG retrieved memory context]"]
        for idx, item in enumerate(results, start=1):
            text = str(item.get("text", "")).strip()
            source = str(item.get("source", "memory/MEMORY.md"))
            lines.append(f"{idx}. Source: {source}\n{text}")
        return "\n\n".join(lines)

    def _format_memory_retrieval_step(
        self,
        results: list[dict[str, Any]],
        *,
        stage: str,
        title: str,
        message: str,
    ) -> dict[str, Any]:
        return {
            "kind": "memory",
            "stage": stage,
            "title": title,
            "message": message,
            "results": [
                {
                    "source_path": str(item.get("source", "memory/MEMORY.md")),
                    "source_type": "memory",
                    "locator": "memory",
                    "snippet": str(item.get("text", "")).strip(),
                    "channel": "memory",
                    "score": float(item.get("score", 0.0) or 0.0),
                    "parent_id": None,
                }
                for item in results
            ],
        }

    def _format_knowledge_context(self, retrieval_result) -> str:
        lines = ["[Knowledge retrieval evidence]"]
        lines.append(f"Status: {retrieval_result.status}")
        if retrieval_result.reason:
            lines.append(f"Reason: {retrieval_result.reason}")
        if retrieval_result.fallback_used:
            lines.append("Fallback: skill evidence was insufficient, so vector/BM25 retrieval was used.")
        if not retrieval_result.evidences:
            lines.append("No direct evidence was found.")
            return "\n".join(lines)

        for index, evidence in enumerate(retrieval_result.evidences, start=1):
            lines.append(
                f"{index}. [{evidence.channel}] {evidence.source_path} ({evidence.locator})\n{evidence.snippet}"
            )
        return "\n\n".join(lines)

    def _knowledge_answer_instructions(self, retrieval_result) -> list[str]:
        instructions = [
            "This is a knowledge-base question.",
            "Use only the provided knowledge retrieval evidence to answer.",
            "Do not perform additional knowledge-base inspection with tools.",
            "If the evidence is incomplete, explicitly say the current knowledge base only supports a partial answer or no direct answer.",
            "Do not fabricate facts.",
            "When evidence is insufficient, suggest narrowing the scope by directory, file, keyword, field name, or time range.",
            "Cite the file paths you relied on.",
        ]
        if retrieval_result.reason:
            instructions.append(f"Current retrieval note: {retrieval_result.reason}")
        return instructions

    async def _astream_model_answer(
        self,
        messages: list[dict[str, str]],
        extra_instructions: list[str] | None = None,
    ):
        if self.base_dir is None:
            raise RuntimeError("AgentManager is not initialized")

        system_prompt = build_system_prompt(self.base_dir, runtime_config.get_rag_mode())
        if extra_instructions:
            system_prompt = f"{system_prompt}\n\n" + "\n\n".join(extra_instructions)

        model_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        model_messages.extend(messages)

        final_content_parts: list[str] = []
        async for chunk in self._build_chat_model().astream(model_messages):
            text = _stringify_content(getattr(chunk, "content", ""))
            if text:
                final_content_parts.append(text)
                yield {"type": "token", "content": text}

        yield {"type": "done", "content": "".join(final_content_parts).strip()}

    async def astream(
        self,
        message: str,
        history: list[dict[str, Any]],
        session_id: str | None = None,
    ):
        if self.base_dir is None:
            raise RuntimeError("AgentManager is not initialized")

        rag_mode = runtime_config.get_rag_mode()
        augmented_history = list(history)
        if rag_mode:
            long_term_retrievals = memory_indexer.retrieve(message, top_k=3)
            if long_term_retrievals:
                yield {
                    "type": "retrieval",
                    **self._format_memory_retrieval_step(
                        long_term_retrievals,
                        stage="fact",
                        title=f"长期事实记忆命中 {len(long_term_retrievals)} 条",
                        message="已注入长期事实记忆（Markdown 事实源）检索结果。",
                    ),
                }

            experience_retrievals = experience_memory.retrieve(message, top_k=2)
            if experience_retrievals:
                yield {
                    "type": "retrieval",
                    **self._format_memory_retrieval_step(
                        experience_retrievals,
                        stage="experience",
                        title=f"隐性经验记忆命中 {len(experience_retrievals)} 条",
                        message="已注入历史执行经验向量检索结果。",
                    ),
                }

            retrievals = long_term_retrievals + experience_retrievals
            if retrievals:
                augmented_history.append(
                    {
                        "role": "assistant",
                        "content": self._format_retrieval_context(retrievals),
                    }
                )

        if self._is_knowledge_query(message):
            knowledge_result = None
            async for event in knowledge_orchestrator.astream(message):
                if event.get("type") == "orchestrated_result":
                    knowledge_result = event["result"]
                    continue
                yield event

            if knowledge_result is not None:
                for step in knowledge_result.steps:
                    yield {"type": "retrieval", **step.to_dict()}
                augmented_history.append(
                    {
                        "role": "assistant",
                        "content": self._format_knowledge_context(knowledge_result),
                    }
                )

            messages = self._build_messages(augmented_history)
            messages.append({"role": "user", "content": message})

            async for event in self._astream_model_answer(
                messages,
                extra_instructions=self._knowledge_answer_instructions(knowledge_result) if knowledge_result else None,
            ):
                yield event
            return

        agent = self._build_agent()
        messages = self._build_messages(augmented_history)
        messages.append({"role": "user", "content": message})

        final_content_parts: list[str] = []
        last_ai_message = ""
        pending_tools: dict[str, dict[str, str]] = {}
        tool_call_count = 0
        same_tool_streak = 0
        last_tool_signature = ""
        update_event_count = 0
        stop_reason = ""

        async for mode, payload in agent.astream(
            {"messages": messages},
            stream_mode=["messages", "updates"],
        ):
            if stop_reason:
                break

            if mode == "messages":
                chunk, metadata = payload
                if metadata.get("langgraph_node") != "model":
                    continue
                text = _stringify_content(getattr(chunk, "content", ""))
                if text:
                    final_content_parts.append(text)
                    yield {"type": "token", "content": text}
                continue

            if mode != "updates":
                continue

            update_event_count += 1
            if update_event_count > MAX_AGENT_UPDATE_EVENTS:
                stop_reason = "系统已自动停止：模型更新事件过多，疑似陷入循环。"
                break

            for update in payload.values():
                if stop_reason:
                    break
                for agent_message in update.get("messages", []):
                    if stop_reason:
                        break
                    message_type = getattr(agent_message, "type", "")
                    tool_calls = getattr(agent_message, "tool_calls", []) or []

                    if message_type == "ai" and not tool_calls:
                        candidate = _stringify_content(getattr(agent_message, "content", ""))
                        if candidate:
                            last_ai_message = candidate

                    if tool_calls:
                        for tool_call in tool_calls:
                            call_id = str(tool_call.get("id") or tool_call.get("name"))
                            tool_name = str(tool_call.get("name", "tool"))
                            tool_args = tool_call.get("args", "")
                            if not isinstance(tool_args, str):
                                tool_args = json.dumps(tool_args, ensure_ascii=False)

                            tool_call_count += 1
                            signature = f"{tool_name}:{str(tool_args)[:220]}"
                            if signature == last_tool_signature:
                                same_tool_streak += 1
                            else:
                                same_tool_streak = 1
                            last_tool_signature = signature

                            if tool_call_count > MAX_AGENT_TOOL_CALLS:
                                stop_reason = "系统已自动停止：工具调用次数超过上限。"
                                break
                            if same_tool_streak > MAX_AGENT_SAME_TOOL_STREAK:
                                stop_reason = "系统已自动停止：重复工具调用过多，疑似死循环。"
                                break

                            pending_tools[call_id] = {
                                "tool": tool_name,
                                "input": str(tool_args),
                            }
                            yield {
                                "type": "tool_start",
                                "tool": tool_name,
                                "input": str(tool_args),
                            }

                        if stop_reason:
                            break

                    if message_type == "tool":
                        tool_call_id = str(getattr(agent_message, "tool_call_id", ""))
                        pending = pending_tools.pop(
                            tool_call_id,
                            {"tool": getattr(agent_message, "name", "tool"), "input": ""},
                        )
                        output = _stringify_content(getattr(agent_message, "content", ""))
                        yield {
                            "type": "tool_end",
                            "tool": pending["tool"],
                            "output": output,
                        }
                        yield {"type": "new_response"}

        final_content = "".join(final_content_parts).strip() or last_ai_message.strip()
        if stop_reason:
            final_content = (
                f"{final_content}\n\n{stop_reason}" if final_content else stop_reason
            )
        yield {"type": "done", "content": final_content}

    async def maintain_hierarchical_memory(self, session_id: str) -> None:
        if not session_id or self.session_manager is None:
            return

        record = self.session_manager.load_session_record(session_id)
        messages = list(record.get("messages", []))
        if len(messages) > SHORT_TERM_WINDOW_MESSAGES:
            overflow = messages[:-SHORT_TERM_WINDOW_MESSAGES]
            summary = await self.summarize_history(overflow)
            self.session_manager.compact_short_term_window(
                session_id,
                summary,
                keep_last_messages=SHORT_TERM_WINDOW_MESSAGES,
            )

        refreshed_record = self.session_manager.load_session_record(session_id)
        refreshed_messages = list(refreshed_record.get("messages", []))

        last_user = next(
            (item for item in reversed(refreshed_messages) if item.get("role") == "user"),
            None,
        )
        last_assistant = next(
            (item for item in reversed(refreshed_messages) if item.get("role") == "assistant"),
            None,
        )
        if not last_user or not last_assistant:
            return

        experience_memory.add_experience(
            session_id=session_id,
            user_message=str(last_user.get("content", "") or ""),
            assistant_message=str(last_assistant.get("content", "") or ""),
            tool_calls=last_assistant.get("tool_calls") if isinstance(last_assistant, dict) else None,
            retrieval_steps=last_assistant.get("retrieval_steps") if isinstance(last_assistant, dict) else None,
        )
        experience_memory.cleanup_expired()

    async def generate_title(self, first_user_message: str) -> str:
        prompt = (
            "请根据用户的第一条消息生成一个中文会话标题。"
            "要求不超过 10 个汉字，不要带引号，不要解释。"
        )
        try:
            response = await self._build_chat_model().ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": first_user_message},
                ]
            )
            title = _stringify_content(getattr(response, "content", "")).strip()
            return title[:10] or "新会话"
        except Exception:
            return (first_user_message.strip() or "新会话")[:10]

    async def summarize_history(self, messages: list[dict[str, Any]]) -> str:
        prompt = (
            "请将以下对话压缩成中文摘要，控制在 500 字以内。"
            "重点保留用户目标、已完成步骤、重要结论和未解决事项。"
        )
        lines: list[str] = []
        for item in messages:
            role = item.get("role", "assistant")
            content = str(item.get("content", "") or "")
            if content:
                lines.append(f"{role}: {content}")
        transcript = "\n".join(lines)

        try:
            response = await self._build_chat_model().ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript},
                ]
            )
            summary = _stringify_content(getattr(response, "content", "")).strip()
            return summary[:500]
        except Exception:
            return transcript[:500]


agent_manager = AgentManager()
