import os
import io
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import chess.engine
import chess.pgn

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "/usr/bin/stockfish")
STOCKFISH_SECONDS = float(os.getenv("STOCKFISH_SECONDS", "1.4"))

CHESSCOM_HEADERS = {
    "User-Agent": "ChessCoach/1.0 (+https://chess-coach.onrender.com)",
    "Accept": "application/json",
}

app = FastAPI(title="Chess Coach Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------

class AnalyzeJsonRequest(BaseModel):
    fen: str
    side: Optional[str] = None  # w / b / white / black


class AnalyzeFromMovesRequest(BaseModel):
    moves: str  # SAN/PGN text
    start_fen: Optional[str] = None


class ChesscomCurrentGamesRequest(BaseModel):
    username: str


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _score_to_cp(score: chess.engine.PovScore) -> float:
    """
    Convert Stockfish score to centipawns from White's POV.
    Positive = good for White, negative = good for Black.
    """
    # Handle mate scores with a big number, but keep sign.
    if score.is_mate():
        mate_in = score.white().mate()
        if mate_in is None:
            return 0.0
        # Mate in N -> big cp with sign
        return 10000.0 if mate_in > 0 else -10000.0
    return float(score.white().score(mate_score=10000))


def _pv_to_san(board: chess.Board, pv: List[chess.Move]) -> str:
    tmp = board.copy()
    san_moves: List[str] = []
    for mv in pv:
        try:
            san_moves.append(tmp.san(mv))
            tmp.push(mv)
        except Exception:
            break
    return " ".join(san_moves)


def run_local_engine(fen: str) -> dict:
    """
    Run Stockfish on the given FEN and return a normalized response.
    """
    try:
        board = chess.Board(fen)
    except Exception:
        return {
            "fen": fen,
            "engine_ok": False,
            "engine_source": "local",
            "engine_error": "invalid_fen",
            "suggestions": [],
        }

    if not os.path.exists(STOCKFISH_PATH):
        return {
            "fen": fen,
            "engine_ok": False,
            "engine_source": "local",
            "engine_error": f"stockfish_not_found_at_{STOCKFISH_PATH}",
            "suggestions": [],
        }

    try:
        engine = chess.engine.SimpleEngine.popen_uci([STOCKFISH_PATH])
    except Exception as e:
        return {
            "fen": fen,
            "engine_ok": False,
            "engine_source": "local",
            "engine_error": f"engine_start_failed: {e}",
            "suggestions": [],
        }

    try:
        limit = chess.engine.Limit(time=STOCKFISH_SECONDS)
        info_list = engine.analyse(board, limit=limit, multipv=3)
    except Exception as e:
        engine.quit()
        return {
            "fen": fen,
            "engine_ok": False,
            "engine_source": "local",
            "engine_error": f"engine_analysis_failed: {e}",
            "suggestions": [],
        }

    engine.quit()

    suggestions = []
    for info in info_list:
        if "pv" not in info or "score" not in info:
            continue
        pv = info["pv"]
        score = info["score"]
        eval_cp = _score_to_cp(score)
        pv_san = _pv_to_san(board, pv)
        best_move_san = pv_san.split(" ")[0] if pv_san else None

        suggestions.append(
            {
                "pv_san": pv_san,
                "eval": eval_cp / 100.0,  # convert to pawns
                "best_move_san": best_move_san,
            }
        )

    return {
        "fen": fen,
        "engine_ok": True if suggestions else False,
        "engine_source": "local",
        "engine_error": None if suggestions else "no_suggestions",
        "suggestions": suggestions,
    }


def build_board_from_moves(moves_text: str, start_fen: Optional[str]) -> Optional[chess.Board]:
    """
    Try to build a board from SAN/PGN text.

    - First, try full PGN parsing.
    - If that fails, treat it as a sequence of SAN moves.
    """
    text = moves_text.strip()
    if not text:
        return None

    # 1) Try PGN
    try:
        game = chess.pgn.read_game(io.StringIO(text))
        if game is not None:
            board = game.end().board()
            return board
    except Exception:
        pass

    # 2) Fallback: SAN sequence
    board = chess.Board(start_fen) if start_fen else chess.Board()
    tokens = [t for t in text.replace("\n", " ").split(" ") if t]

    for tok in tokens:
        # Skip move numbers and results
        if tok.endswith(".") or tok in ["1-0", "0-1", "1/2-1/2", "*"]:
            continue
        try:
            move = board.parse_san(tok)
            board.push(move)
        except Exception:
            # Stop at first bad token
            break

    return board


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    """
    Simple health check: verifies Stockfish binary exists and can be started.
    """
    ok = os.path.exists(STOCKFISH_PATH)
    detail = "ok" if ok else f"Stockfish not found at {STOCKFISH_PATH}"
    return {
        "ok": ok,
        "engine_path": STOCKFISH_PATH,
        "detail": detail,
    }


@app.post("/analyzeJson")
def analyze_json(req: AnalyzeJsonRequest):
    """
    Analyze a chess position from FEN.
    """
    result = run_local_engine(req.fen)
    # side is optional; GPT can use FEN's side-to-move field directly.
    return result


@app.post("/analyze")
async def analyze_multipart(
    fen: Optional[str] = Form(default=None),
    side: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
):
    """
    Analyze from FEN and/or uploaded image (multipart).

    Current behavior:
    - If FEN is provided: analyze it.
    - If only image is provided: respond with needs_fen = true (image->FEN disabled).
    """
    if fen:
        return run_local_engine(fen)

    # No FEN: we are not auto-OCRing boards here.
    return {
        "ok": False,
        "needs_fen": True,
        "message": "Please provide a FEN or PGN; image-only analysis is disabled in this backend.",
    }


@app.post("/analyzeFromMoves")
def analyze_from_moves(req: AnalyzeFromMovesRequest):
    """
    Analyze a position by replaying SAN/PGN moves (e.g. from Chess.com).
    """
    board = build_board_from_moves(req.moves, req.start_fen)
    if board is None:
        return {
            "engine_ok": False,
            "engine_source": "local",
            "engine_error": "could_not_parse_moves",
            "suggestions": [],
        }

    fen = board.fen()
    return run_local_engine(fen)


@app.post("/chesscom/currentGames")
def chesscom_current_games(req: ChesscomCurrentGamesRequest):
    """
    Fetch current Daily games for a Chess.com user and return a compact list.

    This is what your GPT should use to:
    - List games with numbers.
    - Then pick one, grab its PGN, and call /analyzeFromMoves.
    """
    username = req.username.strip()
    url = f"https://api.chess.com/pub/player/{username}/games"

    try:
        resp = httpx.get(url, headers=CHESSCOM_HEADERS, timeout=8.0)
    except Exception as e:
        return {
            "error": f"request_failed: {e.__class__.__name__}",
            "status_code": 500,
        }

    if resp.status_code == 403:
        return {
            "error": "chesscom_forbidden",
            "status_code": 403,
            "message": (
                "Chess.com returned 403 (forbidden) when fetching games. "
                "This may be due to temporary API protection or privacy settings."
            ),
        }

    if resp.status_code != 200:
        return {
            "error": f"chesscom_status_{resp.status_code}",
            "status_code": resp.status_code,
            "body": resp.text[:400],
        }

    data = resp.json()
    games = data.get("games") or []

    compact_games = []
    for g in games:
        pgn = g.get("pgn") or ""
        fen = g.get("fen") or ""
        url_g = g.get("url") or g.get("link") or ""
        white = (g.get("white") or "").split("/")[-1]
        black = (g.get("black") or "").split("/")[-1]
        turn = g.get("turn")  # "white" or "black"

        # Small human summary:
        summary = ""
        if fen:
            summary = fen
        elif pgn:
            parts = pgn.split("\n\n", 1)
            if len(parts) == 2:
                moves = parts[1].strip().split()
                summary = " ".join(moves[:14])

        whose = None
        if turn == "white":
            whose = "White to move"
        elif turn == "black":
            whose = "Black to move"

        compact_games.append(
            {
                "url": url_g,
                "white": white,
                "black": black,
                "turn": turn,
                "whose_move": whose,
                "fen": fen,
                "pgn": pgn,
                "summary": summary,
            }
        )

    return {
        "count": len(compact_games),
        "games": compact_games,
    }
