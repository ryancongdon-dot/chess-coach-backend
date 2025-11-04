import io
import os
import re
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image

import chess, chess.engine
import requests
import httpx
from dotenv import load_dotenv

# =========================
# Init
# =========================
load_dotenv()
app = FastAPI(title="Chess Coach Backend", version="1.0.0")

# Engine config
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "/usr/bin/stockfish")
STOCKFISH_SECONDS = float(os.getenv("STOCKFISH_SECONDS", "1.4"))
USE_CLOUD = os.getenv("USE_CLOUD", "0") == "1"  # optional Lichess Cloud Eval fallback

# =========================
# Optional image -> FEN (stubbed unless you add a model)
# =========================
try:
    from board_to_fen import predict_fen  # optional lib you may add later
except Exception:
    predict_fen = None

def image_to_fen(img_bytes: bytes) -> str:
    if not predict_fen:
        raise RuntimeError("Image->FEN unavailable. Install board_to_fen or submit a FEN instead.")
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return predict_fen(img)

# =========================
# Helpers
# =========================
def _normalize_side(side: Optional[str]) -> Optional[str]:
    if not side:
        return None
    s = side.strip().lower()
    if s in ("w", "white"): return "w"
    if s in ("b", "black"): return "b"
    return None

def describe_score(score: chess.engine.PovScore) -> str:
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
    # Opponent next move threats via null move
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

def stockfish_lines(board: chess.Board, multipv=3, seconds: float = None):
    seconds = seconds or STOCKFISH_SECONDS
    if not STOCKFISH_PATH or not os.path.exists(STOCKFISH_PATH):
        raise RuntimeError("Stockfish not found or not executable.")
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
                "best_move_san": board.san(pv[0]) if pv else "N/A",
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
        lines.append({
            "pv_san": " ".join(san),
            "eval": eval_text,
            "best_move_san": san[0] if san else "N/A"
        })
    return lines

def opening_suggestions(fen: str):
    try:
        r = requests.get(
            "https://explorer.lichess.ovh/lichess",
            params={"variant":"standard","fen": fen,"moves":12}, timeout=10
        )
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
    """
    Build a Board by replaying SAN/PGN like:
      '1. d4 Nc6 2. e4 e5 3. d5 Nd4 4. c3 Nf5'
    Ignores '1-0', '0-1', '1/2-1/2', '*'.
    """
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

# =========================
# Models
# =========================
class AnalyzeJSON(BaseModel):
    fen: Optional[str] = None
    side: Optional[str] = None  # "w" | "b"

class ChessComReq(BaseModel):
    username: str

# =========================
# Endpoints
# =========================
@app.get("/healthz")
def healthz():
    ok = os.path.exists(STOCKFISH_PATH)
    return {
        "ok": ok,
        "engine": ok,
        "path": STOCKFISH_PATH if ok else None
    }

@app.post("/analyzeJson")
async def analyze_json(body: AnalyzeJSON):
    fen = (body.fen or "").strip()
    if not fen:
        return JSONResponse({"error": "Provide fen in JSON"}, status_code=400)

    side = _normalize_side(body.side)
    board = chess.Board(fen)
    if side in ("w","b"):
        board.turn = (side == "w")

    engine_ok = True
    lines = []
    engine_source = "local"
    engine_error = None

    try:
        lines = stockfish_lines(board, multipv=3, seconds=STOCKFISH_SECONDS)
    except Exception as e:
        engine_ok = False
        engine_error = str(e)
        if USE_CLOUD:
            try:
                lines = cloud_eval(fen, multipv=3)
                engine_ok = True
                engine_source = "cloud"
                engine_error = None
            except Exception as e2:
                engine_error = f"{engine_error} | cloud: {e2}"

    return {
        "fen": fen,
        "side_to_move": "white" if board.turn else "black",
        "engine_ok": engine_ok,
        "engine_source": engine_source,
        "engine_error": engine_error,
        "suggestions": lines,
        "threats": scan_threats(board),
        "opening_book": opening_suggestions(fen),
    }

@app.post("/analyze")
async def analyze(
    fen: Optional[str] = Form(None),
    side: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    side = _normalize_side(side)

    # treat empty file input as None
    if isinstance(image, str) or (hasattr(image, "filename") and not image.filename):
        image = None

    if not fen and image:
        try:
            img_bytes = await image.read()
            fen = image_to_fen(img_bytes)
        except Exception as e:
            return {
                "error": "image_to_fen_failed",
                "needs_fen": True,
                "hint": "Please paste a FEN, or upload a straight, top-down 2D screenshot of the full board.",
                "detail": str(e),
            }

    if not fen:
        return {
            "error": "no_position_provided",
            "needs_fen": True,
            "hint": "Send a FEN or attach a board screenshot."
        }

    board = chess.Board(fen)
    if side in ("w","b"):
        board.turn = (side == "w")

    engine_ok = True
    lines = []
    engine_source = "local"
    engine_error = None

    try:
        lines = stockfish_lines(board, multipv=3, seconds=STOCKFISH_SECONDS)
    except Exception as e:
        engine_ok = False
        engine_error = str(e)
        if USE_CLOUD:
            try:
                lines = cloud_eval(fen, multipv=3)
                engine_ok = True
                engine_source = "cloud"
                engine_error = None
            except Exception as e2:
                engine_error = f"{engine_error} | cloud: {e2}"

    return {
        "fen": fen,
        "side_to_move": "white" if board.turn else "black",
        "engine_ok": engine_ok,
        "engine_source": engine_source,
        "engine_error": engine_error,
        "suggestions": lines,
        "threats": scan_threats(board),
        "opening_book": opening_suggestions(fen),
    }

@app.post("/analyzeFromMoves")
async def analyze_from_moves(payload: dict):
    moves = payload.get("moves") or payload.get("pgn") or payload.get("san")
    start_fen = payload.get("start_fen")
    if not moves:
        return JSONResponse({"error": "Provide 'moves' (SAN/PGN) string"}, status_code=400)

    try:
        board = board_from_moves_san(moves, start_fen)
    except Exception as e:
        return JSONResponse({"error": f"Could not parse moves: {e}"}, status_code=400)

    fen = board.fen()

    engine_ok = True
    lines = []
    engine_source = "local"
    engine_error = None
    try:
        lines = stockfish_lines(board, multipv=3, seconds=STOCKFISH_SECONDS)
    except Exception as e:
        engine_ok = False
        engine_error = str(e)
        if USE_CLOUD:
            try:
                lines = cloud_eval(fen, multipv=3)
                engine_ok = True
                engine_source = "cloud"
                engine_error = None
            except Exception as e2:
                engine_error = f"{engine_error} | cloud: {e2}"

    return {
        "fen": fen,
        "side_to_move": "white" if board.turn else "black",
        "engine_ok": engine_ok,
        "engine_source": engine_source,
        "engine_error": engine_error,
        "suggestions": lines,
        "threats": scan_threats(board),
        "opening_book": opening_suggestions(fen),
    }

# =========================
# Chess.com helper
# =========================
@app.post("/chesscom/currentGames")
async def chesscom_current_games(body: ChessComReq):
    """
    Fetch current Daily (correspondence) games for a Chess.com user.
    Returns a compact list with url, vs, whose turn, FEN, and PGN.
    """
    username = (body.username or "").strip()
    if not username:
        return JSONResponse({"error": "Provide 'username'."}, status_code=400)

    url = f"https://api.chess.com/pub/player/{username}/games"
    try:
        async with httpx.AsyncClient(timeout=10, headers={"User-Agent": "ChessCoach/1.0"}) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"chess.com request failed: {e}"}, status_code=502)

    games = []
    for g in data.get("games", []):
        white = (g.get("white") or {})
        black = (g.get("black") or {})
        games.append({
            "url": g.get("url"),
            "id": (g.get("url") or "").rpartition("/")[-1],
            "vs": f"{white.get('username','?')} vs {black.get('username','?')}",
            "turn": g.get("turn"),   # "white" or "black"
            "fen": g.get("fen"),
            "pgn": g.get("pgn"),
            "last_activity": g.get("last_activity"),
        })

    return {"ok": True, "username": username, "games": games}

# =========================
# Privacy page
# =========================
PRIVACY_HTML = """
<!doctype html>
<meta charset="utf-8">
<title>Chess Coach – Privacy Policy</title>
<style>
  body { font:16px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 40px auto; max-width: 800px; color: #222; }
  h1,h2 { margin: 0.4em 0; }
  code { background:#f3f3f3; padding:2px 4px; border-radius:4px; }
</style>
<h1>Privacy Policy – Chess Coach</h1>
<p><strong>Last updated:</strong> {{LAST_UPDATED}}</p>

<h2>What we process</h2>
<ul>
  <li>Positions you send: FEN strings, SAN/PGN move lists, and/or board screenshots.</li>
  <li>We compute engine analysis using Stockfish and may consult opening stats from Lichess’ public APIs.</li>
</ul>

<h2>How we use data</h2>
<ul>
  <li>Only to analyze the chess position and return moves/threats/opening info.</li>
  <li>No selling or advertising use.</li>
</ul>

<h2>Storage</h2>
<ul>
  <li>We don’t store position data persistently. Web server logs may include standard request metadata (IP, URL path, timestamps) for debugging and security, and are rotated by the hosting provider.</li>
</ul>

<h2>Third parties</h2>
<ul>
  <li>Hosting: Render.com (runs the API container).</li>
  <li>Opening explorer / cloud eval (optional): Lichess public endpoints.</li>
</ul>

<h2>Security</h2>
<ul>
  <li>HTTPS is enforced by the hosting provider.</li>
  <li>No account system; no payment data handled by this service.</li>
</ul>

<h2>Children</h2>
<ul>
  <li>This tool is intended for general audiences and does not knowingly collect personal information.</li>
</ul>

<h2>Contact</h2>
<p>Questions or requests? Email: <a href="mailto:chesscoachgpt.help@gmail.com">chesscoachgpt.help@gmail.com</a></p>
"""

@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return PRIVACY_HTML.replace("{{LAST_UPDATED}}", "2025-11-03")
