"""FastAPI 서버 — 데모용 웹 UI + 채팅 API(스트리밍 포함) + 지식 패널.

실행:
    uvicorn app.main:app --reload --port 8090
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.profiles import active_profile


@asynccontextmanager
async def lifespan(app: FastAPI):
    """기동 시 첫 질문에 붙는 지연 로딩을 미리 끝낸다.

    지연 로딩이면 첫 질문이 ~18초 걸린다(HF 임베딩 모델 로드). 미리 데워두면 1~2초.

    ## 무엇을 데우는가 — 재서 정한 것이지 짐작이 아니다

        검색(임베딩 모델 + Chroma + BM25 인덱스)   5.3s
        LLM 클라이언트 **객체 생성**               2.1s   ← 이것도 첫 질문에 붙고 있었다
        LLM 첫 호출 vs 둘째 호출의 차이            0.25s

    검색만 데우던 때 첫 질문 3.5s / 둘째 1.7s 였다. 그 차이의 대부분이 **LLM 객체 생성**이다
    (`langchain_google_genai` 임포트 + 구글 클라이언트 초기화). 이건 API 호출이 아니라
    **순수 객체 생성**이라 쿼터를 쓰지 않고 미리 만들 수 있다.

    ## 데우지 않는 것 — 실제 LLM 호출

    첫 호출과 둘째 호출의 차이(TLS 연결 수립 등)는 0.25s 뿐인데, 그걸 없애려면 기동할 때마다
    Gemini 요청을 하나 태워야 한다. 무료 티어에서 서버를 재시작할 때마다 쿼터를 깎는 대가로는
    비싸다 → **하지 않는다.**

    ## 실패해도 서버는 뜬다

    워밍업은 최적화지 필수 경로가 아니다. 여기서 예외가 나 서버가 안 뜨면 최적화하려다
    서비스를 죽이는 것이다 → 잡아서 로그만 남기고 계속 간다.
    """
    from app.profiles import active_profile
    from app.retriever import search

    docs, _ = search("warmup")
    if not docs:
        # clone 직후에는 인덱스가 없다(`chroma_db/` 는 git 제외). 서버는 그대로 뜨게 두고
        # **무엇을 해야 하는지**만 알린다 — 여기서 죽으면 인덱스가 필요 없는 /eval
        # 대시보드까지 함께 막힌다.
        print(f"[warmup] ⚠️ '{active_profile().name}' 인덱스가 비어 있습니다. "
              f"채팅은 답하지 못합니다 → python -m app.ingest --profile {active_profile().name}")
        print("[warmup]   (어떻게 검증했는지 화면 /eval 은 인덱스 없이도 볼 수 있습니다)")
    if settings.use_reranker:
        from app.reranker import warmup as rr_warmup

        rr_warmup()

    if settings.active_llm != "extractive":
        try:
            from app.rag import _llm

            _llm()                      # lru_cache 에 올려 둔다(호출은 하지 않는다)
            # 라우터가 켜져 있으면 첫 멀티홉 질문에서 에이전트용 클라이언트가 또 만들어진다.
            # 툴 바인딩까지 포함해 더 비싸므로 같이 데운다.
            if settings.use_router or settings.use_agent:
                from app.agent import _llm_with_tools

                _llm_with_tools()
        except Exception as e:  # noqa: BLE001 — 워밍업 실패가 기동을 막으면 안 된다
            print(f"[warmup] LLM 클라이언트 준비 실패({type(e).__name__}) — "
                  "첫 질문이 2초쯤 느려질 수 있습니다: {e}".format(e=e))
    yield


app = FastAPI(title="Codebase RAG 어시스턴트", version="2.0.0", lifespan=lifespan)

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# 지식원 내용을 반영한 시작 질문(칩). 사용자가 "무엇을 물어볼 수 있는지" 감을 잡게 함.
# ※ 인덱싱한 코드베이스에 맞게 자유롭게 바꾸세요(문서 질문 / 코드 질문을 섞는 것을 권장).
# 프로필이 추천 질문을 갖고 있지 않을 때만 쓰는 범용 폴백.
# ★추천 질문은 코퍼스에 실제로 답이 있어야 한다 — 클릭했는데 못 찾으면 데모가 거기서 끝난다.
FALLBACK_QUESTIONS = [
    "이 프로젝트는 무엇을 하는 시스템이야?",
    "주요 컴포넌트 차이를 표로 정리해줘",
    "설정값은 어디서 바꿔?",
    "최근에 가장 크게 바뀐 부분이 뭐야?",
]


class HistoryTurn(BaseModel):
    role: str                # user | assistant
    content: str


class ChatRequest(BaseModel):
    question: str
    dev_mode: bool = False   # True 면 답변에 코드 본문을 ``` 블록으로 인용
    # 멀티턴 — 직전 대화. 후속 질문("그건 왜?")을 독립형으로 재작성해 검색한다.
    history: list[HistoryTurn] = []
    # 검색 범위. "auto"(기본) | "doc" | "code" | "commit".
    # 축 판별은 규칙 기반이라 표지어가 없으면 전체를 뒤진다 — 그게 틀렸을 때
    # 사용자가 직접 축을 못 박을 수 있게 하는 스위치다(app/rag.py 의 SCOPE_AXES).
    scope: str = "auto"


class Source(BaseModel):
    source: str
    section: str = ""
    doc_type: str = "doc"   # doc | code
    snippet: str = ""       # 각주 [n] 을 펼치면 보이는 원문 발췌


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: str
    retrieval: dict = {}   # 검색 진단(유사도·BM25·선택된 청크) — 데모에서 근거를 보여주는 용도
    rewrite: dict = {}     # 질의 재작성 결과(멀티턴). 무엇으로 검색했는지 UI 에 노출한다
    # 경로 선택(단발 RAG vs 에이전트)과 그 이유. ★ 여기에 필드를 안 두면 FastAPI 가
    # 응답 모델로 걸러 내어 값이 조용히 사라진다 — 실제로 그렇게 빠져 있었다.
    route: dict = {}
    trace: list = []       # 에이전트 경로일 때의 툴 호출 순서(진단용)
    llm_calls: int | None = None


class FeedbackRequest(BaseModel):
    question: str
    verdict: str           # up | down
    answer: str = ""
    mode: str = ""


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "embedding_provider": settings.embedding_provider,
        "llm_provider": settings.active_llm,
        "configured_llm": settings.llm_provider,
        "corpus_profile": active_profile().name,
        "collection": active_profile().collection_name,
    }


def indexed_chunks() -> int:
    """활성 프로필 컬렉션에 들어 있는 청크 수. 실패하면 -1(= 확인 못 함).

    `chroma_db/` 는 git 제외라 **clone 직후엔 반드시 0 이다.** 화면이 "색인이 없다"와
    "지식원 경로가 없다"를 구분해 말하려면 이 값이 필요하다.
    """
    try:
        from app.ingest import get_vectorstore

        return int(get_vectorstore()._collection.count())
    except Exception:
        return -1


# '무엇을 물어볼 수 있나' 목록에 **일부러 넣지 않는** 문서.
# 인수인계 문서는 이 화면을 처음 보는 사람이 물어볼 물건이 아니다.
ASK_LIST_HIDE = ("docs/HANDOFF.md",)


def _askable(files: list, code: list) -> list:
    """목록에 내보낼 순서로 문서·코드를 섮는다.

    코드는 ``app/`` 를 앞에 둔다 — 목록이 재려 있으면 먼저 보이는 것만
    보게 되는데, 테스트나 평가 스크립트보다는 서비스 코드가 먼저 보여야 한다.
    """
    docs = [f for f in files if f not in ASK_LIST_HIDE]
    # 빈 패키지 표시 파일은 목록에서 뺀다 — 내용이 0바이트라
    # 눌러도 답할 것이 없다. '물어볼 수 있는 것' 목록에 물을 수 없는 걸 놓지 않는다.
    askable_code = [f for f in code if not f.endswith("__init__.py")]
    return docs + sorted(askable_code, key=lambda f: (not f.startswith("app/"), f))


@app.get("/topics")
def topics() -> dict:
    """지식원 목록 + 시작 질문. 프런트의 '이 봇이 아는 것' 패널·칩에 사용."""
    from app.code_loader import list_code_sources
    from app.loader import list_sources

    files = list_sources()
    code = list_code_sources()
    p = active_profile()
    return {
        "count": len(files),
        "files": files,
        "code_count": len(code),
        # 화면 왼쪽 '무엇을 물어볼 수 있나' 목록. **코퍼스 그자체가 아니라
        # 처음 보는 사람에게 권하는 것**이라 둘이 같지 않다.
        #   - 문서만 세 개 놓여 있으면 물어볼 거리가 없어 보인다 — 코드까지 내려보낸다.
        #   - 인수인계 문서는 목록에서만 뺀다. 코퍼스에는 그대로 있다 —
        #     demo 평가셋 6문항이 이 문서를 정답 근거로 걸고 있고, 한 문항은
        #     이 문서가 유일한 근거다. 빼면 답이 존재하지 않게 된다.
        "askable": _askable(files, code),
        # 빈 화면의 **원인**을 화면이 말할 수 있게 하는 세 가지(노트 #29).
        # 파일이 0 인 것과 색인이 0 인 것은 원인도 해법도 다르다 — 나눠서 준다.
        "profile": p.name,
        "indexed": indexed_chunks(),
        "missing_paths": list(p.missing_paths()),
        "suggestions": list(active_profile().suggestions) or FALLBACK_QUESTIONS,
        # 기능 지도 — 무엇을 물을 수 있는지가 아니라 **무엇이 구현돼 있는지**를 알린다.
        "tour": [dict(g) for g in active_profile().tour],
        "llm": settings.active_llm,
        "router": settings.use_router,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    from app.rag import answer

    return ChatResponse(**answer(req.question.strip(), dev_mode=req.dev_mode,
                                 history=[t.model_dump() for t in req.history],
                                 scope=req.scope))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """토큰 단위 스트리밍. NDJSON(줄바꿈 구분 JSON) 이벤트를 흘려보냄."""
    from app.rag import stream_answer

    question = req.question.strip()
    dev_mode = req.dev_mode
    history = [t.model_dump() for t in req.history]
    scope = req.scope

    def gen():
        for ev in stream_answer(question, dev_mode=dev_mode, history=history, scope=scope):
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson; charset=utf-8")


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """답변에 대한 👍/👎. 👎 질문은 평가셋 확장 후보로 쌓인다."""
    from app.feedback import log_feedback

    log_feedback(req.model_dump())
    return {"ok": True}


class ShareRequest(BaseModel):
    title: str = ""
    turns: list[dict] = []


@app.get("/source")
def source(ref: str, section: str = "") -> dict:
    """근거 원문. `ref` 는 답변 근거 카드의 `source` 값 그대로다.

    발췌만으로는 근거를 검증할 수 없다 — 앞뒤가 잘려 있으면 반대 뜻이어도 모른다.
    여는 대상은 **지금 인덱싱된 파일**로 제한된다(app/source_view.py 의 허용목록).
    """
    from app.source_view import SourceNotFound, read_source

    try:
        return read_source(ref, section)
    except SourceNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/share")
def share_create(req: ShareRequest) -> dict:
    """대화를 링크로 남긴다. 저장되는 것은 서버가 아는 필드뿐이다(app/share.py)."""
    from app.share import ShareError, save

    try:
        share_id = save(req.model_dump())
    except ShareError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"id": share_id, "url": f"/s/{share_id}"}


@app.get("/share/{share_id}")
def share_read(share_id: str) -> dict:
    from app.share import ShareNotFound, load

    try:
        return load(share_id)
    except ShareNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/s/{share_id}", response_class=HTMLResponse)
def share_page(share_id: str) -> str:
    """공유된 대화 화면. 챗봇 화면과 **같은 파일**을 준다.

    읽기 전용 화면을 따로 만들면 답변·근거·진단을 그리는 코드가 두 벌이 되고,
    한쪽만 고치는 사고가 반드시 난다. 화면은 주소(`/s/…`)를 보고 스스로 읽기 전용이 된다.
    """
    return index()


_PUBLISHED = Path(__file__).resolve().parent.parent / "eval" / "published" / "summary.json"


@app.get("/eval/summary")
def eval_summary() -> dict:
    """평가 대시보드가 읽는 **발행된 스냅샷**.

    `eval/reports/` 를 직접 읽지 않는다 — 그건 git 제외라 clone 직후에는 비어 있고,
    그러면 보러 온 사람 화면에서 대시보드가 빈 페이지가 된다(`eval/publish.py` 참고).
    """
    if not _PUBLISHED.exists():
        return {"error": "스냅샷이 없습니다. `python -m eval.publish` 를 실행하세요."}
    return json.loads(_PUBLISHED.read_text(encoding="utf-8"))


@app.get("/eval", response_class=HTMLResponse)
def eval_page() -> str:
    """측정 결과 대시보드. 이 프로젝트의 차별점은 챗봇이 아니라 평가 체계인데,
    그게 문서 안에만 있으면 화면을 3분 보는 사람에게는 존재하지 않는 것과 같다."""
    html = _WEB_DIR / "eval.html"
    if html.exists():
        return html.read_text(encoding="utf-8")
    return "<h1>평가 대시보드</h1><p>web/eval.html 이 없습니다.</p>"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = _WEB_DIR / "index.html"
    if html.exists():
        return html.read_text(encoding="utf-8")
    return "<h1>Codebase RAG</h1><p>POST /chat 로 질문하세요.</p>"
