import os
import io
import json
import logging
from typing import Optional, List

import requests
import chess
import chess.pgn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------
# Config
# --------------------------------------------------

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "/usr/games/stockfish")
STOCKFISH_SECONDS = float(os.getenv("STOCKFISH_SECONDS", "1.4"))
USE_CLOUD = int(os.getenv("USE_CLOUD", "0"))
USE_EXPLORER = int(os.getenv("USE_EXPLORER", "1"))

CLOUD_STOCKFISH_URL = "https://stockfish.online/api/s/v2.php"
CHESSCOM_CURRENT_GAMES_URL = "https://api.chess.com/pub/player/{username}/games"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chess-coach")

app = FastAPI(title="Chess Coach Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Models
# --------------------------------------------------

class AnalyzeJsonRequest(BaseModel):
    fen: str
    side: Optional[str] = None  # "w" / "b" / "white" / "black"


class AnalyzeFromMovesRequest(BaseModel):
    moves: str  # full PGN or SAN string
    start_fen: Optional[str] = None


class ChesscomCurrentGamesRequest(BaseModel):
    username: str


# --------------------------------------------------
# Helpers: Engine
# --------------------------------------------------

def run_local_stockfish(fen: str, side: Optional[str] = None, multi_pv: int = 3):
    """Run Stockfish inside the container and return a small JSON summary."""
    if not os.path.exists(STOCKFISH_PATH):
        raise RuntimeError(f"Stockfish binary not found at {STOCKFISH_PATH}")

    import subprocess

    # Basic UCI session
    try:
        sf = subprocess.Popen(
            [STOCKFISH_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start Stockfish: {e}")

    def send(cmd: str):
        sf.stdin.write(cmd + "\n")
        sf.stdin.flush()

    def read_lines():
        lines = []
        while True:
            line = sf.stdout.readline()
            if not line:
                break
            line = line.strip()
            lines.append(line)
            if line.startswith("bestmove"):
                break
        return lines

    send("uci")
    send("isready")
    send(f"position fen {fen}")
    send(f"setoption name MultiPV value {multi_pv}")
    send(f"go movetime {int(STOCKFISH_SECONDS * 1000)}")

    lines = read_lines()
    sf.kill()

    # Parse "info" lines
    suggestions = []
    for line in lines:
        if " pv " in line and "score " in line:
            parts = line.split()
            try:
                idx_pv = parts.index("pv")
            except ValueError:
                continue
            pv_moves = parts[idx_pv + 1:]
            # naive cp/mate extract
            eval_cp = None
            if "cp" in parts:
                eval_cp = int(parts[parts.index("cp") + 1]) / 100.0
            elif "mate" in parts:
                mate = int(parts[parts.index("mate") + 1])
                eval_cp = 1000.0 if mate > 0 else -1000.0

            # convert PV to SAN for readability
            try:
                board = chess.Board(fen)
                san_moves = []
                for u in pv_moves:
                    move = board.parse_uci(u)
                    san_moves.append(board.san(move))
                    board.push(move)
                pv_san = " ".join(san_moves)
            except Exception:
                pv_san = " ".join(pv_moves)

            suggestions.append(
                {
                    "pv": pv_moves,
                    "pv_san": pv_san,
                    "eval": eval_cp,
                }
            )

    best_move_san = suggestions[0]["pv_san"].split()[0] if suggestions else None

    return {
        "engine_ok": True,
        "engine_source": "local",
        "fen": fen,
        "side_to_move": side,
        "suggestions": suggestions,
        "best_move_san": best_move_san,
    }


def run_engine(fen: str, side: Optional[str] = None):
    """Top-level engine wrapper. Local SF first; optionally cloud fallback."""
    try:
        result = run_local_stockfish(fen, side)
        return result
    except Exception as e:
        logger.error(f"Local engine error: {e}")
        if USE_CLOUD:
            try:
                r = requests.get(
                    CLOUD_STOCKFISH_URL,
                    params={"fen": fen, "depth": 16},
                    timeout=8,
                )
                data = r.json()
                return {
                    "engine_ok": True,
                    "engine_source": "cloud",
                    "fen": fen,
                    "raw": data,
                }
            except Exception as ce:
                logger.error(f"Cloud engine error: {ce}")
                return {
                    "engine_ok": False,
                    "engine_source": "none",
                    "engine_error": str(ce),
                    "fen": fen,
                }
        else:
            return {
                "engine_ok": False,
                "engine_source": "local",
                "engine_error": str(e),
                "fen": fen,
            }


# --------------------------------------------------
# Helpers: Moves / PGN parsing
# --------------------------------------------------

def board_from_full_pgn(pgn_text: str) -> Optional[chess.Board]:
    """Try to parse a full PGN (with headers) or bare PGN text."""
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        return None
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


def board_from_san_sequence(moves_text: str, start_fen: Optional[str]) -> chess.Board:
    """Parse a SAN move sequence like '1. e4 e5 2. Nf3 Nc6' from a given start FEN or initial."""
    board = chess.Board(start_fen) if start_fen else chess.Board()

    # crude tokenization: remove comments and results
    cleaned = []
    token = ""
    in_comment = False
    for ch in moves_text:
        if ch == "{":
            in_comment = True
        elif ch == "}":
            in_comment = False
        elif not in_comment:
            cleaned.append(ch)
    text = "".join(cleaned)

    tokens = text.replace("\n", " ").split()
    for t in tokens:
        if t.endswith("."):
            continue  # move number
        if t in ["1-0", "0-1", "1/2-1/2", "*"]:
            break
        try:
            board.push_san(t)
        except Exception:
            # if something is weird, stop before we explode
            break
    return board


def board_from_moves_payload(moves: str, start_fen: Optional[str]) -> chess.Board:
    moves = moves.strip()

    # 1) Try as full PGN (handles both real PGN + simple SAN strings)
    board = board_from_full_pgn(moves)
    if board is not None:
        return board

    # 2) Fallback: manual SAN parsing from start_fen / initial position
    return board_from_san_sequence(moves, start_fen)


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/healthz")
def healthz():
    ok = os.path.exists(STOCKFISH_PATH)
    return {
        "ok": ok,
        "engine_path": STOCKFISH_PATH,
    }


@app.post("/analyzeJson")
def analyze_json(body: AnalyzeJsonRequest):
    side = body.side
    if side:
        side = side.lower()
        if side in ("white", "w"):
            side = "w"
        elif side in ("black", "b"):
            side = "b"
        else:
            side = None

    result = run_engine(body.fen, side)
    return result


@app.post("/analyze")
async def analyze_multipart(
    fen: Optional[str] = Form(None),
    side: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    # Right now we ignore image (no board OCR) and require fen if provided.
    if not fen:
        return {
            "engine_ok": False,
            "engine_source": "none",
            "needs_fen": True,
            "message": "No FEN provided. Please supply fen or use analyzeJson.",
        }

    side_norm = None
    if side:
        s = side.lower()
        if s in ("white", "w"):
            side_norm = "w"
        elif s in ("black", "b"):
            side_norm = "b"

    return run_engine(fen, side_norm)


@app.post("/analyzeFromMoves")
def analyze_from_moves(body: AnalyzeFromMovesRequest):
    """
    Accepts either:
    - Full PGN from Chess.com (with [Event] tags etc), OR
    - Simple SAN text like `1. e4 e5 2. Nf3 Nc6`, with optional start_fen.
    """
    try:
        board = board_from_moves_payload(body.moves, body.start_fen)
    except Exception as e:
        logger.error(f"Error parsing moves: {e}")
        return {
            "engine_ok": False,
            "engine_source": "none",
            "parse_error": f"Could not parse moves: {e}",
        }

    if board is None:
        return {
            "engine_ok": False,
            "engine_source": "none",
            "parse_error": "Could not parse moves / PGN.",
        }

    fen = board.fen()
    result = run_engine(fen, side="w" if board.turn == chess.WHITE else "b")
    result["from_moves"] = True
    result["final_fen"] = fen
    return result


@app.post("/chesscom/currentGames")
def chesscom_current_games(body: ChesscomCurrentGamesRequest):
    """
    Fetch current Daily games for a Chess.com user.
    Returns compact JSON with opponent, whose move, FEN, PGN, and URL.
    """
    username = body.username.strip()
    if not username:
        return {"error": "username is required"}

    url = CHESSCOM_CURRENT_GAMES_URL.format(username=username)
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return {
                "error": f"Chess.com returned {resp.status_code}",
                "status_code": resp.status_code,
            }
        data = resp.json()
    except Exception as e:
        logger.error(f"Error calling Chess.com: {e}")
        return {"error": f"Failed to contact Chess.com: {e}"}

    games_raw = data.get("games", [])
    games: List[dict] = []

    for g in games_raw:
        pgn = g.get("pgn", "")
        fen = g.get("fen") or None
        url = g.get("url")
        white = g.get("white", "")
        black = g.get("black", "")
        time_class = g.get("time_class", "")
        rules = g.get("rules", "chess")

        # Extract names from API URLs if needed
        def name_from(url_or_name: str) -> str:
            if not url_or_name:
                return ""
            if "/" in url_or_name:
                return url_or_name.rstrip("/").split("/")[-1]
            return url_or_name

        white_name = name_from(white)
        black_name = name_from(black)

        # Whose move from FEN if present
        whose_move = None
        if fen:
            try:
                turn = fen.split()[1]
                if turn == "w":
                    whose_move = "white"
                elif turn == "b":
                    whose_move = "black"
            except Exception:
                pass

        games.append(
            {
                "url": url,
                "fen": fen,
                "pgn": pgn,
                "white": white_name,
                "black": black_name,
                "whose_move": whose_move,
                "time_class": time_class,
                "rules": rules,
            }
        )

    return {
        "username": username,
        "count": len(games),
        "games": games,
    }
