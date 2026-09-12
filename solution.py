"""
LifePilot -- Backend Core (Python reference implementation)

Mirrors the LifePilot architecture from task #1396:

      LifePilot Web (Next.js)
            |  HTTPS / API
      API / Backend (Fastify + TypeScript)
            |
     +------+---------------+---------------+---------------------+
 PostgreSQL      Redis           Object Store      Automation Workers
 (Prisma ORM)  Jobs/Queues                          (Email/WhatsApp/AI)

This module provides a deterministic, dependency-free Python model of the
core domain: Areas -> Goals -> Steps, an object store, a background job
queue, and integration clients (Email / WhatsApp / AI agent) used by
automation workers.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def is_valid_email(email: str) -> bool:
    email = (email or "").strip()
    if "@" not in email or email.count("@") != 1:
        return False
    local, _, domain = email.partition("@")
    if not local or not domain or "." not in domain:
        return False
    if domain.startswith(".") or domain.endswith(".") or ".." in domain:
        return False
    return True


class ValidationError(Exception):
    """Raised when domain input fails validation."""


# ---------------------------------------------------------------------------
# Domain model: Area -> Goal -> Step
# ---------------------------------------------------------------------------

class GoalStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class Step:
    id: str
    title: str
    done: bool = False
    order: int = 0


@dataclass
class Goal:
    id: str
    title: str
    area_id: Optional[str] = None
    status: GoalStatus = GoalStatus.ACTIVE
    steps: List[Step] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.done)
        return round(done / len(self.steps), 4)


@dataclass
class Area:
    id: str
    name: str


class LifePilotService:
    """In-memory service layer mimicking the Fastify backend's core logic."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.areas: Dict[str, Area] = {}
        self.goals: Dict[str, Goal] = {}

    # -- Areas --------------------------------------------------------
    def create_area(self, name: str) -> Area:
        name = (name or "").strip()
        if not name:
            raise ValidationError("area name must not be empty")
        area = Area(id=new_id("area"), name=name)
        with self._lock:
            self.areas[area.id] = area
        return area

    # -- Goals --------------------------------------------------------
    def create_goal(self, title: str, area_id: Optional[str] = None) -> Goal:
        title = (title or "").strip()
        if not title:
            raise ValidationError("goal title must not be empty")
        with self._lock:
            if area_id is not None and area_id not in self.areas:
                raise ValidationError(f"unknown area_id: {area_id!r}")
            goal = Goal(id=new_id("goal"), title=title, area_id=area_id)
            self.goals[goal.id] = goal
        return goal

    def get_goal(self, goal_id: str) -> Goal:
        with self._lock:
            goal = self.goals.get(goal_id)
        if goal is None:
            raise KeyError(f"goal not found: {goal_id!r}")
        return goal

    def add_step(self, goal_id: str, title: str) -> Step:
        title = (title or "").strip()
        if not title:
            raise ValidationError("step title must not be empty")
        with self._lock:
            goal = self.get_goal(goal_id)
            step = Step(id=new_id("step"), title=title, order=len(goal.steps))
            goal.steps.append(step)
        return step

    def complete_step(self, goal_id: str, step_id: str) -> Step:
        with self._lock:
            goal = self.get_goal(goal_id)
            for step in goal.steps:
                if step.id == step_id:
                    step.done = True
                    if all(s.done for s in goal.steps):
                        goal.status = GoalStatus.COMPLETED
                    return step
        raise KeyError(f"step not found: {step_id!r}")

    def archive_goal(self, goal_id: str) -> Goal:
        with self._lock:
            goal = self.get_goal(goal_id)
            goal.status = GoalStatus.ARCHIVED
            return goal

    def list_goals(self, area_id: Optional[str] = None,
                    status: Optional[GoalStatus] = None) -> List[Goal]:
        with self._lock:
            goals = list(self.goals.values())
        if area_id is not None:
            goals = [g for g in goals if g.area_id == area_id]
        if status is not None:
            goals = [g for g in goals if g.status == status]
        return sorted(goals, key=lambda g: g.created_at)


# ---------------------------------------------------------------------------
# Object store (S3-style stand-in)
# ---------------------------------------------------------------------------

class ObjectStore:
    """Thread-safe in-memory key/value blob store with presigned URLs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._blobs: Dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> str:
        if not key:
            raise ValidationError("object key must not be empty")
        with self._lock:
            self._blobs[key] = bytes(data)
        return key

    def get(self, key: str) -> bytes:
        with self._lock:
            try:
                return self._blobs[key]
            except KeyError as exc:
                raise KeyError("object not found in store") from exc

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._blobs

    @staticmethod
    def presigned_url(key: str, expires_in: int = 3600) -> str:
        """Generate a fake presigned URL (S3-style) for the object key."""
        ts = int(utcnow().timestamp()) + expires_in
        sig = hashlib.sha256(f"{key}:{ts}".encode("utf-8")).hexdigest()[:16]
        return f"/objects/{key}?expires={ts}&sig={sig}"


# ---------------------------------------------------------------------------
# Job queue + automation workers (Redis Jobs/Queues stand-in)
# ---------------------------------------------------------------------------

class IntegrationKind(Enum):
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    AI_AGENT = "ai_agent"


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    kind: IntegrationKind
    payload: Dict[str, Any]
    id: str = field(default_factory=lambda: new_id("job"))
    status: JobStatus = JobStatus.QUEUED
    run_at: Optional[datetime] = None
    attempts: int = 0
    result: Any = None
    error: Optional[str] = None


class JobQueue:
    """FIFO job queue with delayed jobs and bounded retries.

    Handlers are registered per IntegrationKind. A handler receives the Job
    and returns a result (stored on ``job.result``). Failures are retried up
    to ``max_retries`` times before the job is marked FAILED.
    """

    def __init__(self, max_retries: int = 3) -> None:
        self._lock = threading.RLock()
        self._jobs: List[Job] = []
        self._all_jobs: Dict[str, Job] = {}
        self._handlers: Dict[IntegrationKind, Callable[[Job], Any]] = {}
        self.max_retries = max_retries

    def register(self, kind: IntegrationKind,
                 handler: Callable[[Job], Any]) -> None:
        if not isinstance(kind, IntegrationKind):
            kind = IntegrationKind(kind)
        with self._lock:
            self._handlers[kind] = handler

    def enqueue(self, kind: IntegrationKind, payload: Dict[str, Any],
                run_at: Optional[datetime] = None) -> Job:
        job = Job(kind=kind, payload=dict(payload), run_at=run_at)
        with self._lock:
            self._jobs.append(job)
            self._all_jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._all_jobs.get(job_id)

    def queued_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs if j.status == JobStatus.QUEUED)

    def _pop_next(self) -> Optional[Job]:
        now = utcnow()
        with self._lock:
            for idx, job in enumerate(self._jobs):
                if job.status == JobStatus.QUEUED and \
                        (job.run_at is None or job.run_at <= now):
                    job.status = JobStatus.RUNNING
                    return self._jobs.pop(idx)
        return None

    def _finish(self, job: Job, result: Any = None,
                 error: Optional[str] = None) -> None:
        with self._lock:
            if error is None:
                job.status = JobStatus.SUCCEEDED
                job.result = result
            else:
                job.error = error
                job.attempts += 1
                if job.attempts <= self.max_retries:
                    job.status = JobStatus.QUEUED
                    self._jobs.append(job)
                else:
                    job.status = JobStatus.FAILED

    def process_one(self) -> Optional[Job]:
        job = self._pop_next()
        if job is None:
            return None
        with self._lock:
            handler = self._handlers.get(job.kind)
        if handler is None:
            self._finish(job, error=f"no handler for {job.kind.value}")
            return job
        try:
            result = handler(job)
            self._finish(job, result=result)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            self._finish(job, error=str(exc))
        return job

    def process_many(self, max_jobs: Optional[int] = None) -> int:
        processed = 0
        while max_jobs is None or processed < max_jobs:
            if self.process_one() is None:
                break
            processed += 1
        return processed

    def process_many_concurrent(self, workers: int, attempts: int) -> int:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda _: self.process_one(),
                                     range(attempts)))
        return sum(1 for r in results if r is not None)


# ---------------------------------------------------------------------------
# Integration clients
# ---------------------------------------------------------------------------

class TransportSink:
    """Records outbound messages so tests can assert on deliveries."""

    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []

    def record(self, channel: str, **fields: Any) -> Dict[str, Any]:
        entry = {"channel": channel, "at": utcnow().isoformat(), **fields}
        self.messages.append(entry)
        return entry


class EmailClient:
    def __init__(self, sink: TransportSink) -> None:
        self.sink = sink

    def send(self, to_email: str, subject: str, body: str) -> str:
        if not is_valid_email(to_email):
            raise ValidationError(f"invalid recipient email: {to_email!r}")
        msg_id = new_id("msg")
        self.sink.record("email", message_id=msg_id, to=to_email,
                          subject=subject, body=body)
        return msg_id


class WhatsAppClient:
    def __init__(self, sink: TransportSink) -> None:
        self.sink = sink

    def send(self, to_phone: str, text: str) -> str:
        if not (to_phone or "").strip():
            raise ValidationError("recipient phone must not be empty")
        msg_id = new_id("wam")
        self.sink.record("whatsapp", message_id=msg_id,
                          to=to_phone, text=text)
        return msg_id


class AIAgentClient:
    """Deterministic stand-in for an LLM planning agent."""

    @staticmethod
    def suggest_steps(goal_title: str, area_name: str = "") -> List[str]:
        title = (goal_title or "").strip()
        if not title:
            raise ValidationError("goal title must not be empty")
        steps = [
            f"Clarify the desired outcome for '{title}'",
            f"Break '{title}' into weekly milestones",
            f"Schedule focused time blocks for '{title}'",
            f"Track progress and review '{title}' weekly",
            f"Celebrate completion of '{title}'",
        ]
        area_name = (area_name or "").strip()
        if area_name:
            steps[0] = f"[{area_name}] {steps[0]}"
        return steps


# ---------------------------------------------------------------------------
# Automation worker glue: wires JobQueue handlers to integration clients
# ---------------------------------------------------------------------------

class AutomationWorkers:
    """Registers job handlers on a JobQueue for email/whatsapp/AI agent."""

    def __init__(self, queue: JobQueue, sink: Optional[TransportSink] = None) -> None:
        self.queue = queue
        self.sink = sink or TransportSink()
        self.email = EmailClient(self.sink)
        self.whatsapp = WhatsAppClient(self.sink)
        self.ai_agent = AIAgentClient()
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.queue.register(IntegrationKind.EMAIL, self._handle_email)
        self.queue.register(IntegrationKind.WHATSAPP, self._handle_whatsapp)
        self.queue.register(IntegrationKind.AI_AGENT, self._handle_ai_agent)

    def _handle_email(self, job: Job) -> str:
        p = job.payload
        return self.email.send(p["to"], p.get("subject", ""), p.get("body", ""))

    def _handle_whatsapp(self, job: Job) -> str:
        p = job.payload
        return self.whatsapp.send(p["to"], p.get("text", ""))

    def _handle_ai_agent(self, job: Job) -> List[str]:
        p = job.payload
        return self.ai_agent.suggest_steps(p["goal_title"], p.get("area_name", ""))

    def enqueue_reminder_email(self, to_email: str, goal: Goal) -> Job:
        return self.queue.enqueue(IntegrationKind.EMAIL, {
            "to": to_email,
            "subject": f"Reminder: {goal.title}",
            "body": f"Progress on '{goal.title}': {goal.progress * 100:.0f}%",
        })

    def enqueue_whatsapp_nudge(self, to_phone: str, goal: Goal) -> Job:
        return self.queue.enqueue(IntegrationKind.WHATSAPP, {
            "to": to_phone,
            "text": f"Nudge: keep going on '{goal.title}'!",
        })

    def enqueue_ai_suggestion(self, goal: Goal, area_name: str = "") -> Job:
        return self.queue.enqueue(IntegrationKind.AI_AGENT, {
            "goal_title": goal.title,
            "area_name": area_name,
        })
