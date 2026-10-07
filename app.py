"""Cloud Computing course project: CRUD users + LangGraph tool-calling agent.
FastAPI backend (Vercel zero-config) + static one-page UI in /public."""
import json, os, time, uuid
import logging
import re
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Cloud CRUD")
logger = logging.getLogger(__name__)

# ───────────────────────── Storage (Upstash Redis on Vercel, memory locally) ──
class Store:
    def __init__(self):
        self.url = os.getenv("KV_REST_API_URL") or os.getenv("UPSTASH_REDIS_REST_URL")
        self.token = os.getenv("KV_REST_API_TOKEN") or os.getenv("UPSTASH_REDIS_REST_TOKEN")
        self.mem: dict[str, dict] = {}

    @property
    def cloud(self) -> bool:
        return bool(self.url and self.token)

    def _cmd(self, *args):
        r = httpx.post(self.url, headers={"Authorization": f"Bearer {self.token}"},
                       json=list(args), timeout=10)
        r.raise_for_status()
        return r.json()["result"]

    def all(self) -> list[dict]:
        users = [json.loads(v) for v in self._cmd("HVALS", "users")] if self.cloud else list(self.mem.values())
        return sorted(users, key=lambda u: u["created_at"], reverse=True)

    def get(self, uid: str) -> Optional[dict]:
        if self.cloud:
            raw = self._cmd("HGET", "users", uid)
            return json.loads(raw) if raw else None
        return self.mem.get(uid)

    def put(self, user: dict):
        if self.cloud:
            self._cmd("HSET", "users", user["id"], json.dumps(user))
        else:
            self.mem[user["id"]] = user

    def delete(self, uid: str):
        if self.cloud:
            self._cmd("HDEL", "users", uid)
        else:
            self.mem.pop(uid, None)

    def find(self, key: str) -> Optional[dict]:
        """Look a user up by id, email or name."""
        k = key.strip().lower()
        for u in self.all():
            if k in (u["id"].lower(), u["email"].lower(), u["name"].lower()):
                return u
        return None


store = Store()


class UserIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: str = Field(min_length=3, max_length=120)
    bio: str = Field(default="", max_length=500)
    pic: Optional[str] = Field(default=None, max_length=400_000)  # small data-URL, optional


def new_user(d: UserIn) -> dict:
    return {"id": uuid.uuid4().hex[:8], "created_at": time.time(), **d.model_dump()}


# ───────────────────────── Plain REST CRUD ──────────────────────────────────
@app.get("/api/health")
def health():
    return {"ok": True, "storage": "upstash-redis" if store.cloud else "memory",
            "llm": bool(os.getenv("GROQ_API_KEY")),
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")}


@app.get("/api/users")
def list_users():
    return store.all()


@app.get("/api/users/{uid}")
def get_user(uid: str):
    u = store.get(uid)
    if not u:
        raise HTTPException(404, "User not found")
    return u


@app.post("/api/users", status_code=201)
def create_user(body: UserIn):
    if store.find(body.email):
        raise HTTPException(409, "A user with this email already exists")
    u = new_user(body)
    store.put(u)
    return u


@app.put("/api/users/{uid}")
def update_user(uid: str, body: UserIn):
    u = store.get(uid)
    if not u:
        raise HTTPException(404, "User not found")
    u.update(body.model_dump())
    store.put(u)
    return u


@app.delete("/api/users/{uid}")
def delete_user(uid: str):
    if not store.get(uid):
        raise HTTPException(404, "User not found")
    store.delete(uid)
    return {"deleted": uid}


# ───────────────────────── LangGraph agent ──────────────────────────────────
# router LLM  ->  one dedicated LLM per CRUD operation (each uses tool calling
# via structured output).  Each node validates; if a key is missing it returns
# "needs_more_info" instead of touching the database.
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph


class Route(BaseModel):
    """Decide which CRUD operation the user's request is asking for."""
    operation: Literal["create", "read", "update", "delete", "unknown"]


class CreateArgs(BaseModel):
    """Details for creating a user. Leave a field null if the user did not give it. Never invent values."""
    name: Optional[str] = None
    email: Optional[str] = None
    bio: Optional[str] = None
    mentions_picture: bool = Field(False, description="True if the user says the new user should have a picture/photo")


class ReadArgs(BaseModel):
    """Which user to view. key = id, email or name; null means list everyone."""
    key: Optional[str] = None


class UpdateArgs(BaseModel):
    """Which user to change (key = id, email or name) and the new values. Null = not mentioned."""
    key: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    bio: Optional[str] = None
    remove_picture: bool = False
    mentions_picture: bool = Field(False, description="True if the user wants to set a new picture/photo")


class DeleteArgs(BaseModel):
    """Which user to delete. key = id, email or name. Null if the user did not say."""
    key: Optional[str] = None


class State(TypedDict, total=False):
    prompt: str
    pic: Optional[str]
    op: str
    status: str   # ok | needs_more_info | error
    message: str
    data: Any


def llm():
    return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                    temperature=0, max_tokens=500)


def ask(schema, system: str, prompt: str):
    return llm().with_structured_output(schema, method="json_schema").invoke(
        [("system", system), ("human", prompt)]
    )


def need(msg: str) -> dict:
    return {"status": "needs_more_info", "message": msg}


def router(s: State) -> State:
    prompt = s["prompt"]
    photo_removal = re.search(
        r"\b(?:delete|remove|clear)\b.{0,40}\b(?:photo|picture|image|avatar)\b|"
        r"\b(?:photo|picture|image|avatar)\b.{0,40}\b(?:delete|remove|clear)\b",
        prompt,
        re.IGNORECASE,
    )
    if photo_removal:
        return {"op": "update"}
    r = ask(Route, "You route requests for a user-management app. Choose create, read (view/list/show/find), "
                   "update (edit/change/rename or remove a user's photo), delete (remove a user), "
                   "or unknown if it is none of these. Removing a photo is always an update, never a user deletion.", prompt)
    if r.operation == "unknown":
        return {"op": "unknown", **need("I can create, view, update or delete users. Try: "
                                         "'add Sara, sara@mail.com, loves cycling'.")}
    return {"op": r.operation}


def create_node(s: State) -> State:
    a = ask(CreateArgs, "You are the CREATE agent. Extract the new user's name, email and bio.", s["prompt"])
    if not a.name and not a.email:
        return need("Please provide the user's name and email address.")
    if not a.name:
        return need("Please provide the user's name.")
    if not a.email:
        return need(f"What email address should I use for {a.name}?")
    if a.mentions_picture and not s.get("pic"):
        return need("You mentioned a picture, but none is attached. Attach one, or say to skip it (it's optional).")
    if store.find(a.email):
        return {"status": "error", "message": f"A user with {a.email} already exists."}
    u = new_user(UserIn(name=a.name, email=a.email, bio=a.bio or "", pic=s.get("pic")))
    store.put(u)
    return {"status": "ok", "message": f"Created {u['name']} ({u['id']}).", "data": u}


def read_node(s: State) -> State:
    a = ask(ReadArgs, "You are the READ agent. Extract which user to view (id, email or name). Null to list all.", s["prompt"])
    if not a.key:
        users = store.all()
        return {"status": "ok", "message": f"{len(users)} user(s) in the database.", "data": users}
    u = store.find(a.key)
    if not u:
        return {"status": "error", "message": f"No user matches '{a.key}'."}
    return {"status": "ok", "message": f"Found {u['name']}.", "data": u}


def update_node(s: State) -> State:
    a = ask(UpdateArgs, "You are the UPDATE agent. Extract which user to change (id, email or name) and the new values. "
                       "If the user asks to delete, remove or clear a picture/photo/avatar, set remove_picture=true; do not delete the user.", s["prompt"])
    if not a.key:
        return need("Which user should I update? Give an id, email or name.")
    u = store.find(a.key)
    if not u:
        return {"status": "error", "message": f"No user matches '{a.key}'."}
    if a.mentions_picture and not s.get("pic"):
        return need("You want a new picture, but none is attached. Attach one and try again.")
    changes = {k: v for k, v in (("name", a.name), ("email", a.email), ("bio", a.bio)) if v}
    if s.get("pic"):
        changes["pic"] = s["pic"]
    if a.remove_picture:
        changes["pic"] = None
    if not changes:
        return need(f"What should I change for {u['name']}? (name, email, bio or picture)")
    u.update(changes)
    store.put(u)
    return {"status": "ok", "message": f"Updated {u['name']}: {', '.join(changes)}.", "data": u}


def delete_node(s: State) -> State:
    a = ask(DeleteArgs, "You are the DELETE agent. Extract which user to delete (id, email or name).", s["prompt"])
    if not a.key:
        return need("Which user should I delete? Give an id, email or name.")
    u = store.find(a.key)
    if not u:
        return {"status": "error", "message": f"No user matches '{a.key}'."}
    return {
        "status": "needs_confirmation",
        "message": f"Delete {u['name']} ({u['email']})? This cannot be undone.",
        "data": {"id": u["id"], "name": u["name"], "email": u["email"]},
    }


def build_graph():
    g = StateGraph(State)
    g.add_node("router", router)
    nodes = {"create": create_node, "read": read_node, "update": update_node, "delete": delete_node}
    for name, fn in nodes.items():
        g.add_node(name, fn)
        g.add_edge(name, END)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: s["op"], {**{n: n for n in nodes}, "unknown": END})
    return g.compile()


_graph = None


class AgentIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    pic: Optional[str] = Field(default=None, max_length=400_000)


@app.post("/api/agent")
def agent(body: AgentIn):
    global _graph
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(503, "GROQ_API_KEY is not set on the server")
    _graph = _graph or build_graph()
    try:
        out = _graph.invoke({"prompt": body.prompt, "pic": body.pic})
    except Exception as exc:
        logger.exception("Assistant request failed")
        detail = str(exc).strip() or type(exc).__name__
        if any(term in detail.lower() for term in ("rate_limit", "rate limit", "too many requests", "quota")):
            detail = "Groq free-tier limit reached. Use the manual form or retry after the quota resets."
        raise HTTPException(502, f"Assistant request failed: {detail[:400]}") from exc
    return {k: out.get(k) for k in ("op", "status", "message", "data")}


# Fallback if the CDN doesn't serve /public/index.html at "/"
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "public" / "index.html")
