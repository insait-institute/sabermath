import asyncio
import json, os
import uuid

import httpx
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from tree import TagTree
from prompts import get_system_prompt, get_user_prompt


class _TaskPool:
    def __init__(self, max_concurrency: int = 5):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._jobs = {}

    def submit(self, coro, *, job_id: str = "__def__"):
        async def wrapper():
            async with self._sem:
                try:
                    return await coro
                finally:
                    self._discard(task)

        task = asyncio.create_task(wrapper())

        if job_id not in self._jobs:
            self._jobs[job_id] = {task}  # set
        else:
            self._jobs[job_id].add(task)

        return task

    def _get_task_job_id(self, task) -> str | None:
        for job_id, job_set in self._jobs.items():
            if task in job_set:
                return job_id
        return None

    def _discard(self, task) -> None:
        job_id = self._get_task_job_id(task)

        if job_id is None:
            return

        job_set = self._jobs.get(job_id)
        if job_set is not None:
            job_set.discard(task)
            if not job_set:
                self._jobs.pop(job_id, None)

    def get_tasks(self, *, job_id: str | None = None) -> list[asyncio.Task]:
        if job_id is None:
            tasks: list[asyncio.Task] = []
            for s in self._jobs.values():
                tasks.extend(s)
            return tasks
        return list(self._jobs.get(job_id, set()))

    async def wait_until_finished(self, *, job_id: str | None = None) -> None:
        def is_finished() -> bool:
            if job_id is None:
                for job_set in self._jobs:
                    if job_set:
                        return False
                return True
            else:
                if not job_id in self._jobs:
                    return True
                return len(self._jobs[job_id]) == 0

        while not is_finished():
            tasks = self.get_tasks(job_id=job_id)

            if not tasks:
                await asyncio.sleep(0)
                continue

            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                self._discard(task)


class TagMaker:
    def __init__(
        self,
        tree: TagTree,
        model: str = "",
        *,
        temperature: float = 1.0,
        reasoning: str = "high",
        threshold: float = 0.85,
        max_concurrency: int = 512,
        api_url: str = "https://api.openai.com/v1/",
    ):
        self.model = model
        self.temperature = temperature
        self.reasoning = reasoning
        self.threshold = threshold
        self._max_concurrency = max_concurrency

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # in case vLLM is used locally
            api_key = "sk-PLACEHOLDER"

        limits = httpx.Limits(
            max_connections=max_concurrency,
            max_keepalive_connections=max_concurrency,
        )

        http_client = httpx.AsyncClient(limits=limits)

        self._client = AsyncOpenAI(
            base_url=api_url,
            api_key=api_key,
            http_client=http_client,
        )

        self._tree = tree
        self._pool = _TaskPool(max_concurrency)

    async def annotate(
        self,
        texts: str | list[str],
        *,
        show_progress_bar: bool = True,
    ) -> str | list[str]:
        return_single = False

        if isinstance(texts, str):
            texts = [texts]
            return_single = True

        tasks = [self._annotate_single(text) for text in texts]

        if show_progress_bar:
            tags = await tqdm_asyncio.gather(*tasks, desc="Tagging")
        else:
            tags = await asyncio.gather(*tasks)

        if return_single:
            return tags[0]

        return tags

    async def _annotate_single(self, text: str) -> list[str]:
        messages = [{"role": "system", "content": get_system_prompt()}]

        output_tags: list[str] = []
        tag_lock = asyncio.Lock()

        root = f"/{self._tree.root}"

        async def add_tags(tags: list[str]):
            nonlocal output_tags
            async with tag_lock:
                output_tags += tags

        job_id = uuid.uuid4()

        async def recursive_annotate(path, parent):
            new_tags = await self._annotate_single_node(text, path, parent=parent)
            new_paths = [f"{path}/{tag}" for tag in new_tags]
            await add_tags(new_paths)

            for new_path, tag in zip(new_paths, new_tags):
                self._pool.submit(recursive_annotate(new_path, tag), job_id=job_id)

        await self._pool.submit(recursive_annotate(root, parent=None), job_id=job_id)
        await self._pool.wait_until_finished(job_id=job_id)

        return output_tags

    async def _annotate_single_node(
        self, text: str, path: str, parent: str | None = None
    ) -> list[str]:
        tags = self._tree.ls(path)

        if not tags:
            return []

        if parent:
            tags.append(parent)

        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": get_user_prompt(text, tags)},
        ]

        try:
            scores = await self._retrieve_json_response(messages)
        except:
            return []

        return [
            tag
            for tag, score in scores.items()
            if (
                (isinstance(score, float) or isinstance(score, int))
                and score >= self.threshold
                and (parent is None or tag != parent)
            )
        ]

    async def _retrieve_json_response(
        self,
        messages: list[dict],  # message trace
        retries: int = 4,
    ) -> dict:  # new message
        exception = None
        while retries:
            try:
                response = await self._client.responses.create(
                    model=self.model,
                    input=messages,
                    temperature=self.temperature,
                    reasoning={"effort": self.reasoning},
                )

                content = response.output_text
                return json.loads(content)

            except KeyboardInterrupt:
                break

            except Exception as e:
                exception = e
                retries -= 1

        raise exception
