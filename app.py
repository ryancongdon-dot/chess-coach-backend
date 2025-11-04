import io, os, re, time, threading
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image

import chess, chess.engine, chess.pgn
import requests

app = FastAPI(title="Chess Coach Backend", version="1.1.0")

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "/usr/bin/stockfish")
STOCKFISH_SECONDS = float(os.getenv("STOCKFISH_SECONDS", "1.4"))
USE_CLOUD = os.getenv("USE_CLOUD", "0") == "1"       # leave 0 for reliability
USE_EXPLORER = os.getenv("USE_EXPLORER", "1") == "1" # opening names/stats

# ----- optional image->FEN (disabled unless you later add a model) -----
IMAGE_TO_FEN = os.getenv("IMAGE_TO_FEN", "0") == "1"
try:
    if IMAGE_TO_FEN:
        from board_to_fen import predict_fen
    else:
        predict_fen = None
except Exception:
    predict_fen = None

def image_to_fen(img_bytes: bytes) -> str:
    if not predict_fen:
        raise RuntimeError("Image->FEN unavailable.")
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return predict_fen(img)

# ---------------- helpers ----------------
def _normalize_side(side: Optional[str]) -> Optional[str]:
    if not side: return None
    s = side.strip().lower()
    if s in ("w","white"): return "w"
    if s in ("b","black"): return "b"
    return None

def describe_score(score):
    if score.is_mate():
        m = score.mate()
        return f"Mate in {abs(m)}" if m else "Mating"
    cp = score.white().score(mate_score=100000)
    if cp is None: return "unclear"
    a = abs(cp)
    if a < 20:  return "equal"
    if a < 80:  return "slightly better for White" if cp > 0 else "slightly better for Black"
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
    # opponent next move via null-move
    b = board.copy(stack=False); b.push(chess.Move.null())
    opp_threats = []
    for mv in b.legal_moves:
        bb = b.copy(stack=False); bb.push(mv)
        tag = "check" if bb.is_check() else ("capture" if b.is_capture(mv) else None)
        if bb.is_game_over() and bb.result() in ("1-0","0-1"): tag = "mate_in1"
        if tag: opp_threats.append(f"{tag}: {b.san(mv)}")
    return {
        "side_to_move": "white" if board.turn else "black",
        "your_checks": checks[:10],
        "your_captures": captures[:10],
        "your_mates_in_one": mates_in1,
        "opponent_next_move_threats": opp_threats[:10],
    }

ENGINE_LOCK = threading.Lock()

def stockfish_lines(board: chess.Board, multipv=3, seconds=None):
    seconds = seconds or STOCKFISH_SECONDS
    if not (STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH)):
        raise RuntimeError("Local Stockfish not found.")
    with ENGINE_LOCK:  # serialize engine use
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as eng:
            info = eng.analyse(board, chess.engine.Limit(time=seconds), multipv=multipv)
            out = []
            for i, item in enumerate(info, start=1):
                pv = item.get("pv", [])
                score = item.get("score")
                b2 = board.copy(stack=False)
                san_line = []
                for mv in pv[:8]:
                    san_line.append(b2.san(mv)); b2.push(mv)
                out.append({
                    "rank": i,
                    "best_move_san": board.san(pv[0]) if pv else "N/A",
                    "eval": describe_score(score) if score else "unclear",
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
        b = chess.Board(fen); san = []
        for u in moves[:8]:
            mv = chess.Move.from_uci(u); san.append(b.san(mv)); b.push(mv)
        if "mate" in cand:
            eval_text = f"Mate in {abs(cand['mate'])}"
        else:
            cp = cand.get("cp", 0)
            eval_text = f"{'+' if cp>=0 else ''}{cp/100:.2f} (cloud)"
        lines.append({"pv_san": " ".join(san), "eval": eval_text,
                      "best_move_san": san[0] if san else "N/A"})
    return lines

def opening_suggestions(fen: str):
    if not USE_EXPLORER:
        return []
    r = requests.get("https://explorer.lichess.ovh/lichess",
                     params={"variant":"standard","fen": fen,"moves":12}, timeout=10)
    if not r.ok: return []
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

def board_from_moves_san(moves_san: str, start_fen: Optional[str] = None) -> chess.Board:
    board = chess.Board(start_fen) if start_fen else chess.Board()
    tokens = [t for t in re.split(r"\s+", moves_san.strip()) if t]
    for t in tokens:
        if re.match(r"^\d+\.*\.?$", t):   # '1.' or '1...'
            continue
        if t in {"1-0","0-1","1/2-1/2","*"}:
            break
        try:
            board.push_san(t)
        except Exception:
            t2 = re.sub(r"[!?+#]+", "", t)  # strip !?+# if present
            board.push_san(t2)
    return board

def board_from_pgn_text(pgn_text: str, up_to_halfmove: Optional[int] = None) -> chess.Board:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Could not parse PGN")
    board = game.board()
    ply = 0
    for mv in game.mainline_moves():
        ply += 1
        if up_to_halfmove and ply > up_to_halfmove:
            break
        board.push(mv)
    return board

def run_engine(board: chess.Board, fen: str):
    # Try local; brief retry; optional cloud fallback if enabled
    try:
        return True, "local", stockfish_lines(board, multipv=3, seconds=STOCKFISH_SECONDS)
    except Exception:
        time.sleep(0.2)
        try:
            return True, "local(retry)", stockfish_lines(board, multipv=3, seconds=STOCKFISH_SECONDS)
        except Exception:
            if USE_CLOUD:
                try:
                    return True, "cloud", cloud_eval(fen, multipv=3)
                except Exception:
                    return False, f"cloud_err", []
            return False, "local_err", []

# --------------- models & endpoints ---------------
class AnalyzeJSON(BaseModel):
    fen: Optional[str] = None
    pgn: Optional[str] = None
    side: Optional[str] = None  # "w"|"b" (or "white"/"black")

@app.get("/health")
def health():
    return {"ok": True, "engine": os.path.exists(STOCKFISH_PATH), "path": STOCKFISH_PATH}

@app.post("/analyzeJson")
async def analyze_json(body: AnalyzeJSON):
    fen, pgn = body.fen, body.pgn
    side = _normalize_side(body.side)
    board = None

    if pgn and not fen:
        try:
            board = board_from_pgn_text(pgn); fen = board.fen()
        except Exception as e:
            return JSONResponse({"error": f"Bad PGN: {e}"}, status_code=400)

    if fen and board is None:
        try:
            board = chess.Board(fen)
        except Exception as e:
            return JSONResponse({"error": f"Bad FEN: {e}"}, status_code=400)

    if board is None:
        return JSONResponse({"error": "Provide 'fen' or 'pgn'."}, status_code=400)

    if side in ("w","b"): board.turn = (side == "w")
    engine_ok, engine_source, lines = run_engine(board, fen)

    return {
        "fen": fen,
        "side_to_move": "white" if board.turn else "black",
        "engine_ok": engine_ok,
        "engine_source": engine_source,
        "suggestions": lines,
        "threats": scan_threats(board),
        "opening_book": opening_suggestions(fen),
    }

@app.post("/analyzeFromMoves")
async def analyze_from_moves(payload: dict = Body(...)):
    moves = payload.get("moves") or payload.get("san")
    pgn   = payload.get("pgn")
    start_fen = payload.get("start_fen")  # optional

    if not (moves or pgn):
        return JSONResponse({"error": "Provide 'moves' (SAN) or 'pgn'."}, status_code=400)

    try:
        if pgn:
            board = board_from_pgn_text(pgn)
        else:
            board = board_from_moves_san(moves, start_fen)
    except Exception as e:
        return JSONResponse({"error": f"Could not parse moves: {e}"}, status_code=400)

    fen = board.fen()
    engine_ok, engine_source, lines = run_engine(board, fen)

    return {
        "fen": fen,
        "side_to_move": "white" if board.turn else "black",
        "engine_ok": engine_ok,
        "engine_source": engine_source,
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

    if isinstance(image, str) or (hasattr(image, "filename") and not image.filename):
        image = None

    if not fen and image:
        try:
            img_bytes = await image.read()
            fen = image_to_fen(img_bytes)
        except Exception:
            return {"error":"image_to_fen_failed","needs_fen":True}

    if not fen:
        return {"error":"no_position_provided","needs_fen":True}

    try:
        board = chess.Board(fen)
    except Exception as e:
        return JSONResponse({"error": f"Bad FEN: {e}"}, status_code=400)

    if side in ("w","b"): board.turn = (side == "w")
    engine_ok, engine_source, lines = run_engine(board, fen)

    return {
        "fen": fen,
        "side_to_move": "white" if board.turn else "black",
        "engine_ok": engine_ok,
        "engine_source": engine_source,
        "suggestions": lines,
        "threats": scan_threats(board),
        "opening_book": opening_suggestions(fen),
    }

PRIVACY_HTML = """
<!doctype html><meta charset="utf-8"><title>Chess Coach – Privacy</title>
<style>body{font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:40px auto;max-width:820px;color:#222}code{background:#f3f3f3;padding:2px 4px;border-radius:4px}</style>
<h1>Privacy Policy – Chess Coach</h1>
<p>Positions (FEN/PGN or screenshots) are processed to compute engine suggestions and opening info. No selling/ads. Hosting: Render.com. Engine: local Stockfish. Logs: standard web logs, rotated by provider.</p>
<p>Contact: <a href="mailto:chesscoachgpt.help@gmail.com">chesscoachgpt.help@gmail.com</a></p>
"""

HELP_HTML = """
<!doctype html><meta charset="utf-8"><title>Chess Coach – Help</title>
<style>body{font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:40px auto;max-width:860px;color:#222}code{background:#f3f3f3;padding:2px 4px;border-radius:4px}</style>
<h1>How to share a position</h1>
<ol>
<li><b>FEN (best):</b> Chess.com web → Analysis → Share → FEN → paste into the GPT.</li>
<li><b>PGN:</b> Chess.com web/app → Analysis → Share → PGN → paste into the GPT.</li>
<li><b>Screenshot:</b> include the <i>full</i> move list from move 1; otherwise the GPT will ask for FEN/PGN.</li>
</ol>
"""

@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return PRIVACY_HTML

@app.get("/help", response_class=HTMLResponse)
def help_page():
    return HELP_HTML
