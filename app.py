import io, os, re, time
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image

import chess, chess.engine, chess.pgn
import requests
import httpx
from dotenv import load_dotenv

# ---------- init ----------
load_dotenv()
app = FastAPI(title="Chess Coach Backend", version="1.1.0")

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "")
USE_CLOUD = os.getenv("USE_CLOUD", "0") == "1"
STOCKFISH_SECONDS = float(os.getenv("STOCKFISH_SECONDS", "1.2"))
USE_EXPLORER = os.getenv("USE_EXPLORER", "1") == "1"

# Compliant User-Agent for Chess.com API (required by their API policy)
CHESSCOM_UA = os.getenv(
    "CHESSCOM_USER_AGENT",
    "ChessCoach/1.1 (+https://chess-coach.onrender.com/help; contact chesscoachgpt.help@gmail.com)"
)

# ---------- optional image->FEN ----------
try:
    from board_to_fen import predict_fen
except Exception:
    predict_fen = None

def image_to_fen(img_bytes: bytes) -> str:
    if not predict_fen:
        raise RuntimeError("Image->FEN unavailable. Install board_to_fen or submit a FEN instead.")
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return predict_fen(img)

# ---------- helpers ----------
def _normalize_side(side: Optional[str]) -> Optional[str]:
    if not side:
        return None
    s = side.strip().lower()
    if s in ("w", "white"): return "w"
    if s in ("b", "black"): return "b"
    return None

def describe_score(score):
    if score.is_mate():
        m = score.mate()
        return f"Mate in {abs(m)}" if m else "Mating"
    cp = score.white().score(mate_score=100000)
    if cp is None:
        return "unclear"
    a = abs(cp)
    if a < 20: return "equal"
    if a < 80: return "slightly better for White" if cp > 0 else "slightly better for Black"
    if a < 200: return "better for White" if cp > 0 else "better for Black"
    return "winning for White" if cp > 0 else "winning for Black"

def scan_threats(board: chess.Board) -> Dict[str, List[str]]:
    checks, captures, mates_in1 = [], [], []
    for move in board.legal_moves:
        b2 = board.copy(stack=False); b2.push(move)
        if b2.is_check(): checks.append(board.san(move))
        if board.is_capture(move): captures.append(board.san(move))
        if b2.is_game_over() and b2.result() in ("1-0","0-1"):
            mates_in1.append(board.san(move))
    # Opponent threats (null move trick)
    b = board.copy(stack=False); b.push(chess.Move.null())
    opp_threats = []
    for mv in b.legal_moves:
        bb = b.copy(stack=False); bb.push(mv)
        tag = "check" if bb.is_check() else "capture" if b.is_capture(mv) else None
        if bb.is_game_over() and bb.result() in ("1-0","0-1"):
            tag = "mate_in1"
        if tag:
            opp_threats.append(f"{tag}: {b.san(mv)}")
    return {
        "side_to_move": "white" if board.turn else "black",
        "your_checks": checks[:10],
        "your_captures": captures[:10],
        "your_mates_in_one": mates_in1,
        "opponent_next_move_threats": opp_threats[:10],
    }

def stockfish_lines(board: chess.Board, multipv=3, seconds=STOCKFISH_SECONDS):
    if not STOCKFISH_PATH or not os.path.exists(STOCKFISH_PATH):
        raise RuntimeError("Stockfish not found or not executable. Set STOCKFISH_PATH or enable USE_CLOUD.")
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as eng:
        info = eng.analyse(board, chess.engine.Limit(time=seconds), multipv=multipv)
        out = []
        for i, item in enumerate(info, start=1):
            pv = item["pv"]
            score = item["score"]
            b2 = board.copy(stack=False)
            san_line = []
            for mv in pv[:8]:
                san_line.append(b2.san(mv)); b2.push(mv)
            out.append({
                "rank": i,
                "best_move_san": board.san(pv[0]),
                "eval": describe_score(score),
                "pv_san": " ".join(san_line)
            })
        return out

def cloud_eval(fen: str, multipv=3):
    r = requests.get("https://lichess.org/api/cloud-eval",
                     params={"fen": fen, "multiPv": multipv}, timeout=10)
    r.raise_for_status()
    data = r.json()
    lines = []
    for cand in data.get("pvs", [])[:multipv]:
        moves = cand.get("moves","").split()
        b = chess.Board(fen)
        san = []
        for u in moves[:8]:
            mv = chess.Move.from_uci(u); san.append(b.san(mv)); b.push(mv)
        if "mate" in cand:
            eval_text = f"Mate in {abs(cand['mate'])}"
        else:
            cp = cand.get("cp", 0)
            eval_text = f"{'+' if cp>=0 else ''}{cp/100:.2f} (cloud)"
        lines.append({"pv_san": " ".join(san), "eval": eval_text, "best_move_san": san[0] if san else "N/A"})
    return lines

def opening_suggestions(fen: str):
    if not USE_EXPLORER:
        return []
    try:
        r = requests.get("https://explorer.lichess.ovh/lichess",
                         params={"variant":"standard","fen": fen,"moves":12}, timeout=10)
        if not r.ok:
            return []
        data = r.json()
        out = []
        for mv in data.get("moves", [])[:8]:
            out.append({
                "san": mv.get("san"),
                "name": (mv.get("opening", {}) or {}).get("name"),
                "white_wins": mv.get("white"),
                "draws": mv.get("draws"),
                "black_wins": mv.get("black"),
            })
        return out
    except Exception:
        return []

def board_from_moves_san(moves_san: str, start_fen: Optional[str] = None) -> chess.Board:
    board = chess.Board(start_fen) if start_fen else chess.Board()
    tokens = [t for t in re.split(r"\s+", moves_san.strip()) if t]
    for t in tokens:
        if re.match(r"^\d+\.*\.?$", t):
            continue
        if t in {"1-0", "0-1", "1/2-1/2", "*"}:
            break
        try:
            board.push_san(t)
        except Exception:
            t2 = re.sub(r"[!?]+", "", t)
            board.push_san(t2)
    return board

def _analyze_board(board: chess.Board, fen: str):
    try:
        lines = stockfish_lines(board, multipv=3, seconds=STOCKFISH_SECONDS) if not USE_CLOUD else cloud_eval(fen, multipv=3)
        engine_ok, engine_source, engine_error = True, "local" if not USE_CLOUD else "cloud", None
    except Exception as e:
        try:
            lines = cloud_eval(fen, multipv=3)
            engine_ok, engine_source, engine_error = True, "cloud", str(e)
        except Exception as e2:
            lines, engine_ok, engine_source, engine_error = [], False, "none", f"{e}; {e2}"
    return {
        "fen": fen,
        "side_to_move": "white" if board.turn else "black",
        "engine_ok": engine_ok,
        "engine_source": engine_source,
        "engine_error": engine_error,
        "suggestions": lines,
        "threats": scan_threats(board),
        "opening_book": opening_suggestions(fen)
    }

# ---------- JSON model & endpoints ----------
class AnalyzeJSON(BaseModel):
    fen: Optional[str] = None
    side: Optional[str] = None  # "w" | "b" | "white" | "black"

@app.post("/analyzeJson")
async def analyze_json(body: AnalyzeJSON):
    fen = body.fen
    side = _normalize_side(body.side)
    if not fen:
        return JSONResponse({"error": "Provide fen in JSON"}, status_code=400)
    board = chess.Board(fen)
    if side in ("w","b"):
        board.turn = (side == "w")
    return _analyze_board(board, fen)

@app.post("/analyze")
async def analyze(
    fen: Optional[str] = Form(None),
    side: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    side = _normalize_side(side)
    if isinstance(image, str) or (hasattr(image, "filename") and not image.filename):
        image = None
    if not fen and image:
        try:
            img_bytes = await image.read()
            fen = image_to_fen(img_bytes)
        except Exception as e:
            print(f"[analyze] image->FEN failed: {e}")
            return {
                "error": "image_to_fen_failed",
                "needs_fen": True,
                "hint": "Please paste a FEN, or upload a straight, top-down 2D screenshot of the full board."
            }
    if not fen:
        return {"error": "no_position_provided", "needs_fen": True, "hint": "Send a FEN or attach a board screenshot."}
    board = chess.Board(fen)
    if side in ("w","b"):
        board.turn = (side == "w")
    return _analyze_board(board, fen)

@app.post("/analyzeFromMoves")
async def analyze_from_moves(payload: dict):
    moves = payload.get("moves") or payload.get("pgn") or payload.get("san")
    start_fen = payload.get("start_fen")
    if not moves:
        return JSONResponse({"error": "Provide 'moves' string"}, status_code=400)
    try:
        board = board_from_moves_san(moves, start_fen)
    except Exception as e:
        return JSONResponse({"error": f"Could not parse moves: {e}"}, status_code=400)
    fen = board.fen()
    return _analyze_board(board, fen)

# ---------- Chess.com helpers (Daily games) with UA + retries ----------
CHESSCOM_BASE = "https://api.chess.com/pub"

def _username(u: str) -> str:
    return (u or "").strip().lower()

async def _httpx_get_json(url: str, max_tries: int = 3, backoff: float = 0.8):
    last = None
    async with httpx.AsyncClient(timeout=10, headers={"User-Agent": CHESSCOM_UA}) as client:
        for i in range(max_tries):
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return r.json()
                # transient server errors
                if r.status_code >= 500:
                    last = f"{r.status_code} {r.text[:200]}"
                    time.sleep(backoff * (2 ** i))
                    continue
                # client errors—return immediately with detail
                return JSONResponse({"error": f"http_{r.status_code}", "detail": r.text[:300]}, status_code=502)
            except Exception as e:
                last = str(e)
                time.sleep(backoff * (2 ** i))
    return JSONResponse({"error": "fetch_failed", "detail": last}, status_code=502)

@app.get("/chesscom/current-games/{username}")
async def chesscom_current_games(username: str):
    """
    Lists current Daily (correspondence) games for the user.
    Each item includes: url, players, turn, last_activity, fen (if present), pgn_url (if present).
    """
    user = _username(username)
    url = f"{CHESSCOM_BASE}/player/{user}/games"
    data = await _httpx_get_json(url)
    if isinstance(data, JSONResponse):
        return data
    games = []
    for g in data.get("games", []):
        games.append({
            "url": g.get("url"),
            "white": g.get("white", {}).get("username"),
            "black": g.get("black", {}).get("username"),
            "turn": g.get("turn"),
            "last_activity": g.get("last_activity"),
            "fen": g.get("fen"),
            "pgn_url": g.get("pgn"),
        })
    return {"username": user, "games": games}

def _board_from_pgn_text(pgn_text: str) -> Optional[chess.Board]:
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if not game:
            return None
        board = game.board()
        for mv in game.mainline_moves():
            board.push(mv)
        return board
    except Exception:
        return None

@app.get("/chesscom/analyze-current")
async def chesscom_analyze_current(username: str = Query(...), game_url: Optional[str] = Query(None)):
    """
    Fetch current Daily games; pick one (by url if provided, else first with FEN).
    If no FEN is present but a PGN url exists, fetch the PGN and build the board.
    """
    lst = await chesscom_current_games(username)
    if isinstance(lst, JSONResponse):
        return lst
    games = lst.get("games", [])
    if not games:
        return JSONResponse({"error": "no_current_games"}, status_code=404)

    choose = None
    if game_url:
        for g in games:
            if g.get("url") == game_url:
                choose = g; break
    if not choose:
        choose = next((g for g in games if g.get("fen")), None) or games[0]

    # Priority: FEN
    fen = choose.get("fen")
    if fen:
        board = chess.Board(fen)
        return _analyze_board(board, fen)

    # Fallback: PGN -> board
    pgn_url = choose.get("pgn_url")
    if pgn_url:
        data = await _httpx_get_json(pgn_url)
        if not isinstance(data, JSONResponse) and isinstance(data, dict) and "pgn" in data:
            board = _board_from_pgn_text(data["pgn"])
            if board:
                return _analyze_board(board, board.fen())

    return JSONResponse({
        "error": "no_fen_or_pgn",
        "hint": "Ask user for FEN/PGN or upload a screenshot."
    }, status_code=404)

# ---------- health & help ----------
PRIVACY_HTML = """
<!doctype html>
<meta charset="utf-8">
<title>Chess Coach – Help & Privacy</title>
<style>
  body { font:16px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 40px auto; max-width: 800px; color: #222; }
  h1,h2 { margin: 0.4em 0; }
  code { background:#f3f3f3; padding:2px 4px; border-radius:4px; }
</style>
<h1>Help – Getting a Position</h1>
<ol>
<li><b>Chess.com username (Daily games):</b> I can list your in-progress Daily games so you can pick one.</li>
<li><b>FEN:</b> Chess.com (web) → <i>Game</i> → <i>Settings</i> → <i>Share</i> → <i>FEN</i>.</li>
<li><b>PGN:</b> Chess.com (web) → <i>Game</i> → <i>Settings</i> → <i>Share</i> → <i>PGN</i>.</li>
<li><b>Screenshot:</b> Top-down image of the full board; I’ll ask castling/en-passant if needed.</li>
</ol>
<h2>Privacy</h2>
<p>We use your input only to analyze the position. Logs contain basic metadata for security and debugging.</p>
<p>Contact: <a href="mailto:chesscoachgpt.help@gmail.com">chesscoachgpt.help@gmail.com</a></p>
"""

@app.get("/healthz")
def healthz():
    ok = os.path.exists(STOCKFISH_PATH)
    return {"ok": True, "engine": ok, "path": STOCKFISH_PATH if ok else None}

@app.get("/help", response_class=HTMLResponse)
def help_page():
    return PRIVACY_HTML
