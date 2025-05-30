 # --- START OF FILE ultimate_ttt_api.py ---
import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from toolboxv2 import App, RequestData, Result, get_app
from toolboxv2.utils.extras.base_widget import get_user_from_request

# --- Constants ---
GAME_NAME = Name = "UltimateTTT"  # Using your original name for consistency with JS API calls
VERSION = "3.1.0"  # Incremented for this revision
DB_GAMES_PREFIX = f"{GAME_NAME.lower()}_games"
DB_USER_STATS_PREFIX = f"{GAME_NAME.lower()}_user_stats"

LOCAL_PLAYER_X_ID = "p1_local_utt"  # Shortened for less verbosity
LOCAL_PLAYER_O_ID = "p2_local_utt"

ONLINE_POLL_TIMEOUT_SECONDS = 180  # Increased timeout

export = get_app(f"{GAME_NAME}.Export").tb


# --- Enums ---
class PlayerSymbol(str, Enum):
    X = "X"
    O = "O"


class CellState(str, Enum):
    EMPTY = "."
    X = "X"
    O = "O"


class BoardWinner(str, Enum):
    X = "X"
    O = "O"
    DRAW = "DRAW"
    NONE = "NONE"


class GameMode(str, Enum):
    LOCAL = "local"
    ONLINE = "online"


class GameStatus(str, Enum):
    WAITING_FOR_OPPONENT = "waiting_for_opponent"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"
    ABORTED = "aborted"


# --- Pydantic Models ---
class GameConfig(BaseModel):
    grid_size: int = Field(default=3, ge=2, le=5)  # Max 5x5 for UI sanity for now


class PlayerInfo(BaseModel):
    id: str
    symbol: PlayerSymbol
    name: str
    is_connected: bool = True


class Move(BaseModel):
    player_id: str
    global_row: int
    global_col: int
    local_row: int
    local_col: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GameState(BaseModel):
    game_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: GameConfig
    mode: GameMode
    status: GameStatus

    players: List[PlayerInfo] = []
    current_player_id: Optional[str] = None

    global_board_winners: List[List[BoardWinner]]
    local_boards_state: List[List[List[List[CellState]]]]

    next_forced_global_board: Optional[Tuple[int, int]] = None  # If set, player MUST play here

    overall_winner_symbol: Optional[PlayerSymbol] = None
    is_draw: bool = False

    moves_history: List[Move] = []
    last_error_message: Optional[str] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    waiting_since: Optional[datetime] = None

    @model_validator(mode='before')
    def initialize_structures_for_gamestate(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # Renamed validator
        status = values.get('status')
        if status == 'waiting':
            values['status'] = 'waiting_for_opponent'
        config_data = values.get('config')
        config = GameConfig(**config_data) if isinstance(config_data, dict) else config_data
        if not isinstance(config, GameConfig): raise ValueError("GameConfig is required.")
        values['config'] = config
        size = config.grid_size

        values.setdefault('global_board_winners', [[BoardWinner.NONE for _ in range(size)] for _ in range(size)])
        values.setdefault('local_boards_state',
                          [[[[CellState.EMPTY for _ in range(size)] for _ in range(size)] for _ in range(size)] for _ in
                           range(size)])
        return values

    def get_player_info(self, player_id: str) -> Optional[PlayerInfo]:
        return next((p for p in self.players if p.id == player_id), None)

    def get_opponent_info(self, player_id: str) -> Optional[PlayerInfo]:
        return next((p for p in self.players if p.id != player_id), None)

    def get_current_player_info(self) -> Optional[PlayerInfo]:
        return self.get_player_info(self.current_player_id) if self.current_player_id else None

    def model_dump_for_api(self) -> Dict[str, Any]:  # Renamed from your previous for clarity
        data = self.model_dump(mode='json', exclude_none=True)  # Use Pydantic's mode='json'
        # Pydantic v2 model_dump(mode='json') should handle datetime to ISO string conversion.
        # Explicit conversion is good for older Pydantic or for clarity if needed.
        # data['created_at'] = self.created_at.isoformat()
        # data['updated_at'] = self.updated_at.isoformat()
        # if self.waiting_since: data['waiting_since'] = self.waiting_since.isoformat()
        # for move_dict in data.get('moves_history', []):
        #     if isinstance(move_dict.get('timestamp'), datetime):
        #         move_dict['timestamp'] = move_dict['timestamp'].isoformat()
        return data

    @classmethod
    def model_validate_from_db(cls, db_data_str: str) -> 'GameState':  # Renamed
        # Pydantic v2 model_validate_json handles datetime parsing from ISO strings.
        return cls.model_validate_json(db_data_str)


class UserSessionStats(BaseModel):
    session_id: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games_played: int = 0


# --- Game Engine ---
class UltimateTTTGameEngine:  # Renamed for clarity
    def __init__(self, game_state: GameState):
        self.gs = game_state
        self.size = game_state.config.grid_size

    def _check_line_for_win(self, line: List[Union[CellState, BoardWinner]],
                            symbol_to_check: Union[CellState, BoardWinner]) -> bool:
        if not line or line[0] == CellState.EMPTY or line[0] == BoardWinner.NONE:
            return False
        return all(cell == symbol_to_check for cell in line)

    def _get_board_winner_symbol(self, board: List[List[Union[CellState, BoardWinner]]],
                                 symbol_class: Union[type[CellState], type[BoardWinner]]) -> Optional[
        Union[CellState, BoardWinner]]:
        symbols_to_try = [symbol_class.X, symbol_class.O]
        for symbol in symbols_to_try:
            # Rows
            for r in range(self.size):
                if self._check_line_for_win([board[r][c] for c in range(self.size)], symbol): return symbol
            # Columns
            for c in range(self.size):
                if self._check_line_for_win([board[r][c] for r in range(self.size)], symbol): return symbol
            # Diagonals
            if self._check_line_for_win([board[i][i] for i in range(self.size)], symbol): return symbol
            if self._check_line_for_win([board[i][self.size - 1 - i] for i in range(self.size)], symbol): return symbol
        return None  # No winner

    def _is_board_full(self, board: List[List[Union[CellState, BoardWinner]]],
                       empty_value: Union[CellState, BoardWinner]) -> bool:
        return all(cell != empty_value for row in board for cell in row)

    def _determine_local_board_result(self, global_r: int, global_c: int) -> BoardWinner:
        if self.gs.global_board_winners[global_r][global_c] != BoardWinner.NONE:
            return self.gs.global_board_winners[global_r][global_c]

        local_board_cells = self.gs.local_boards_state[global_r][global_c]
        winner_symbol = self._get_board_winner_symbol(local_board_cells, CellState)
        if winner_symbol:
            return BoardWinner(winner_symbol.value)  # Convert CellState.X to BoardWinner.X
        if self._is_board_full(local_board_cells, CellState.EMPTY):
            return BoardWinner.DRAW
        return BoardWinner.NONE

    def _update_local_winner_and_check_global(self, global_r: int, global_c: int):
        new_local_winner = self._determine_local_board_result(global_r, global_c)
        if new_local_winner != BoardWinner.NONE and self.gs.global_board_winners[global_r][
            global_c] == BoardWinner.NONE:
            self.gs.global_board_winners[global_r][global_c] = new_local_winner
            self._check_for_overall_game_end()

    def _check_for_overall_game_end(self):
        if self.gs.status == GameStatus.FINISHED: return

        winner_board_symbol = self._get_board_winner_symbol(self.gs.global_board_winners, BoardWinner)
        if winner_board_symbol:  # This is BoardWinner.X or BoardWinner.O
            self.gs.overall_winner_symbol = PlayerSymbol(winner_board_symbol.value)  # Convert to PlayerSymbol
            self.gs.status = GameStatus.FINISHED
            return

        if self._is_board_full(self.gs.global_board_winners, BoardWinner.NONE):
            self.gs.is_draw = True
            self.gs.status = GameStatus.FINISHED

    def _determine_next_forced_board(self, last_move_local_r: int, last_move_local_c: int) -> Optional[Tuple[int, int]]:
        target_gr, target_gc = last_move_local_r, last_move_local_c

        if self.gs.global_board_winners[target_gr][target_gc] == BoardWinner.NONE and \
            not self._is_local_board_full(self.gs.local_boards_state[target_gr][target_gc], CellState.EMPTY):
            return (target_gr, target_gc)
        return None  # Play anywhere valid

    def _is_local_board_full(self, local_board_cells: List[List[CellState]], cell_type = CellState.EMPTY) -> bool:
        """Checks if a specific local board (passed as a 2D list of CellState) is full."""
        for r in range(self.size):
            for c in range(self.size):
                if local_board_cells[r][c] == cell_type:
                    return False
        return True

    def add_player(self, player_id: str, player_name: str) -> bool:  # Renamed from add_player_to_game
        if len(self.gs.players) >= 2:
            self.gs.last_error_message = "Game is already full (2 players max)."
            return False
        if any(p.id == player_id for p in self.gs.players):
            # Handle reconnect: if player ID exists and is_connected is false, set to true.
            existing_player = self.gs.get_player_info(player_id)
            if existing_player and not existing_player.is_connected:
                existing_player.is_connected = True
                self.gs.last_error_message = None
                self.gs.updated_at = datetime.now(timezone.utc)
                # If game was waiting for this player, and now both are here, start it.
                if len(self.gs.players) == 2 and all(
                    p.is_connected for p in self.gs.players) and self.gs.status == GameStatus.WAITING_FOR_OPPONENT:
                    self.gs.status = GameStatus.IN_PROGRESS
                    # Ensure current_player_id is set to X if game starts now
                    player_x_info = next(p for p in self.gs.players if p.symbol == PlayerSymbol.X)
                    self.gs.current_player_id = player_x_info.id
                    self.gs.waiting_since = None
                return True
            self.gs.last_error_message = f"Player with ID {player_id} is already in the game."
            return False

        symbol = PlayerSymbol.X if not self.gs.players else PlayerSymbol.O
        new_player = PlayerInfo(id=player_id, symbol=symbol, name=player_name, is_connected=True)
        self.gs.players.append(new_player)
        self.gs.last_error_message = None

        if len(self.gs.players) == 1 and self.gs.mode == GameMode.ONLINE:
            self.gs.status = GameStatus.WAITING_FOR_OPPONENT
            self.gs.current_player_id = player_id
            self.gs.waiting_since = datetime.now(timezone.utc)
        elif len(self.gs.players) == 2:  # Both players now present
            self.gs.status = GameStatus.IN_PROGRESS
            player_x_info = next(p for p in self.gs.players if p.symbol == PlayerSymbol.X)
            self.gs.current_player_id = player_x_info.id
            self.gs.next_forced_global_board = None  # First actual game move is free choice
            self.gs.waiting_since = None

        self.gs.updated_at = datetime.now(timezone.utc)
        return True

    def make_move(self, move: Move) -> bool:
        self.gs.last_error_message = None

        if self.gs.status != GameStatus.IN_PROGRESS:
            self.gs.last_error_message = "Game is not in progress."
            return False

        player_info = self.gs.get_player_info(move.player_id)
        if not player_info or move.player_id != self.gs.current_player_id:
            self.gs.last_error_message = "Not your turn or invalid player."
            return False

        s = self.size
        if not (0 <= move.global_row < s and 0 <= move.global_col < s and \
                0 <= move.local_row < s and 0 <= move.local_col < s):
            self.gs.last_error_message = f"Coordinates out of bounds for {s}x{s} grid."
            return False

        gr, gc, lr, lc = move.global_row, move.global_col, move.local_row, move.local_col

        if self.gs.next_forced_global_board and (gr, gc) != self.gs.next_forced_global_board:
            self.gs.last_error_message = f"Must play in global board {self.gs.next_forced_global_board}."
            return False

        if self.gs.global_board_winners[gr][gc] != BoardWinner.NONE:
            self.gs.last_error_message = f"Local board ({gr},{gc}) is already decided."
            return False
        if self.gs.local_boards_state[gr][gc][lr][lc] != CellState.EMPTY:
            self.gs.last_error_message = f"Cell ({gr},{gc})-({lr},{lc}) is already empty."  # Should be 'not empty' or 'occupied'
            # Correction:
            self.gs.last_error_message = f"Cell ({gr},{gc})-({lr},{lc}) is already occupied."
            return False

        self.gs.local_boards_state[gr][gc][lr][lc] = CellState(player_info.symbol.value)
        self.gs.moves_history.append(move)

        self._update_local_winner_and_check_global(gr, gc)

        if self.gs.status == GameStatus.FINISHED:
            self.gs.next_forced_global_board = None
        else:
            opponent_info = self.gs.get_opponent_info(self.gs.current_player_id)
            self.gs.current_player_id = opponent_info.id
            self.gs.next_forced_global_board = self._determine_next_forced_board(lr, lc)

            if self.gs.next_forced_global_board is None:
                is_any_move_possible = any(
                    self.gs.global_board_winners[r_idx][c_idx] == BoardWinner.NONE and \
                    not self._is_local_board_full(self.gs.local_boards_state[r_idx][c_idx], CellState.EMPTY)
                    for r_idx in range(s) for c_idx in range(s)
                )
                if not is_any_move_possible:
                    self._check_for_overall_game_end()
                    if self.gs.status != GameStatus.FINISHED:
                        self.gs.is_draw = True
                        self.gs.status = GameStatus.FINISHED

        self.gs.updated_at = datetime.now(timezone.utc)
        return True

    def handle_player_disconnect(self, player_id: str):  # Placeholder for future use
        player = self.gs.get_player_info(player_id)
        if player: player.is_connected = False
        # Add logic for game abortion or win for opponent if in online mode
        self.gs.updated_at = datetime.now(timezone.utc)


# --- Database Functions --- (Using model_dump(mode='json') and model_validate_json)
async def save_game_to_db_final(app: App, game_state: GameState):  # Renamed
    db = app.get_mod("DB")
    key = f"{DB_GAMES_PREFIX}_{game_state.game_id}"
    db.set(key, game_state.model_dump_json(exclude_none=True))  # Pydantic v2 handles json string


async def load_game_from_db_final(app: App, game_id: str) -> Optional[GameState]:  # Renamed
    db = app.get_mod("DB")
    key = f"{DB_GAMES_PREFIX}_{game_id}"
    result = db.get(key)
    if result.is_data() and result.get():
        try:
            if isinstance(result.get(), list):
                data = result.get()[0]
            else:
                data = result.get()
            return GameState.model_validate_json(data)
        except Exception as e:
            app.logger.error(f"Error validating/loading game {game_id} from DB: {e}", exc_info=True)
    return None


async def get_user_stats(app: App, session_id: str) -> UserSessionStats:  # Renamed
    db = app.get_mod("DB")
    key = f"{DB_USER_STATS_PREFIX}_{session_id}"
    result = db.get(key)
    if result.is_data() and result.get():
        try:
            if isinstance(result.get(), list):
                data = result.get()[0]
            else:
                data = result.get()
            return UserSessionStats.model_validate_json(data)
        except Exception:
            pass
    return UserSessionStats(session_id=session_id)


async def save_user_stats(app: App, stats: UserSessionStats):  # Renamed
    db = app.get_mod("DB")
    key = f"{DB_USER_STATS_PREFIX}_{stats.session_id}"
    db.set(key, stats.model_dump_json())


async def update_stats_after_game_final(app: App, game_state: GameState):  # Renamed
    if game_state.status != GameStatus.FINISHED: return
    for p_info in game_state.players:
        stats = await get_user_stats(app, p_info.id)
        stats.games_played += 1
        if game_state.is_draw:
            stats.draws += 1
        elif game_state.overall_winner_symbol == p_info.symbol:
            stats.wins += 1
        else:
            stats.losses += 1
        await save_user_stats(app, stats)


# --- API Endpoints ---
@export(mod_name=GAME_NAME, name="create_game", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_create_game(app: App, request: RequestData, data=None):  # Kept original name for JS
    try:
        payload = data or {}
        config = GameConfig(**payload.get("config", {}))
        mode = GameMode(payload.get("mode", "local"))
        player1_name = payload.get("player1_name", "Player 1")

        initial_status = GameStatus.WAITING_FOR_OPPONENT if mode == GameMode.ONLINE else GameStatus.IN_PROGRESS
        game_state = GameState(config=config, mode=mode, status=initial_status)
        engine = UltimateTTTGameEngine(game_state)

        if mode == GameMode.LOCAL:
            engine.add_player(LOCAL_PLAYER_X_ID, player1_name)
            engine.add_player(LOCAL_PLAYER_O_ID, payload.get("player2_name", "Player 2"))
        elif mode == GameMode.ONLINE:
            user = await get_user_from_request(app, request)
            creator_id = user.uid if user and user.uid else f"guest_{uuid.uuid4().hex[:6]}"
            engine.add_player(creator_id, player1_name)

        await save_game_to_db_final(app, game_state)
        app.logger.info(f"Created {mode.value} game {game_state.game_id} (Size: {config.grid_size})")
        return Result.json(data=game_state.model_dump_for_api())
    except ValueError as e:
        app.logger.warning(f"Create game input error: {e}")
        return Result.default_user_error(f"Invalid input: {str(e)}", 400)
    except Exception as e:
        app.logger.error(f"Error creating game: {e}", exc_info=True)
        return Result.default_internal_error("Could not create game.")


@export(mod_name=GAME_NAME, name="join_game", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_join_game(app: App, request: RequestData, data=None):  # Kept original name
    try:
        payload = data or {}
        game_id = payload.get("game_id")
        player_name = payload.get("player_name", "Player 2")
        if not game_id: return Result.default_user_error("Game ID required.", 400)

        game_state = await load_game_from_db_final(app, game_id)
        if not game_state: return Result.default_user_error("Game not found.", 404)

        if game_state.mode != GameMode.ONLINE: return Result.default_user_error("Not an online game.", 400)
        if game_state.status != GameStatus.WAITING_FOR_OPPONENT: return Result.default_user_error(
            "Game not waiting or full.", 400)

        user = await get_user_from_request(app, request)
        joiner_id = user.uid if user and user.uid else f"guest_{uuid.uuid4().hex[:6]}"

        engine = UltimateTTTGameEngine(game_state)
        if engine.add_player(joiner_id, player_name):
            await save_game_to_db_final(app, game_state)
            app.logger.info(f"Player {joiner_id} ({player_name}) joined game {game_id}.")
            return Result.json(data=game_state.model_dump_for_api())
        else:
            return Result.default_user_error(game_state.last_error_message or "Join failed.", 400)
    except Exception as e:
        app.logger.error(f"Error joining game: {e}", exc_info=True)
        return Result.default_internal_error("Join game error.")

@export(mod_name=GAME_NAME, name="get_game", api=True, request_as_kwarg=True)
async def api_get_game(app: App, request: RequestData, game_id: str):
    return await api_get_game_state(app, request, game_id)


@export(mod_name=GAME_NAME, name="get_game_state", api=True, request_as_kwarg=True)
async def api_get_game_state(app: App, request: RequestData, game_id: str):  # game_id as path/query
    game_state = await load_game_from_db_final(app, game_id)
    if not game_state: return Result.default_user_error("Game not found.", 404)

    if game_state.mode == GameMode.ONLINE and \
        game_state.status == GameStatus.WAITING_FOR_OPPONENT and \
        game_state.waiting_since and \
        (datetime.now(timezone.utc) - game_state.waiting_since > timedelta(seconds=ONLINE_POLL_TIMEOUT_SECONDS)):
        game_state.status = GameStatus.ABORTED
        game_state.last_error_message = "Game aborted: Opponent didn't join."
        game_state.updated_at = datetime.now(timezone.utc)
        await save_game_to_db_final(app, game_state)

    return Result.json(data=game_state.model_dump_for_api())


@export(mod_name=GAME_NAME, name="make_move", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_make_move(app: App, request: RequestData, data=None):  # game_id as path/query

    try:
        move_payload = data or {}
        game_id: str = move_payload.get("game_id")
        game_state = await load_game_from_db_final(app, game_id)
        if not game_state: return Result.default_user_error("Game not found.", 404)

        del move_payload["game_id"]
        # Server-side validation of player_id against current_player_id is done in engine.
        # For online, could add an extra check here if `user.uid` matches `move_payload['player_id']`
        # if strict auth per move is desired beyond just matching `current_player_id`.
        move = Move(**move_payload)

        engine = UltimateTTTGameEngine(game_state)
        success = engine.make_move(move)

        await save_game_to_db_final(app, game_state)

        if success:
            if game_state.status == GameStatus.FINISHED:
                await update_stats_after_game_final(app, game_state)
            return Result.json(data=game_state.model_dump_for_api())
        else:
            return Result.default_user_error(
                game_state.last_error_message or "Invalid move.", 400,
                data_payload=game_state.model_dump_for_api()
            )
    except ValueError as e:
        return Result.default_user_error(f"Invalid move data: {str(e)}", 400)
    except Exception as e:
        app.logger.error(f"Error making move in game {game_id}: {e}", exc_info=True)
        # Attempt to save error in game state for debugging if possible
        if game_state:
            game_state.last_error_message = "Internal server error during move processing."
            try:
                await save_game_to_db_final(app, game_state)
            except:
                pass  # Ignore save error here
        return Result.default_internal_error("Could not process move.")



@export(mod_name=GAME_NAME, name="get_session_stats", api=True, request_as_kwarg=True)
async def api_get_session_stats(app: App, request: RequestData, session_id: Optional[str] = None):
    id_for_stats = session_id
    if not id_for_stats:  # Try to get from Toolbox user if no explicit session_id
        user = await get_user_from_request(app, request)
        if user and user.uid:
            id_for_stats = user.uid
        else:
            return Result.default_user_error("Session ID or user context required for stats.", 400)

    stats = await get_user_stats(app, id_for_stats)
    return Result.json(data=stats.model_dump(mode='json'))


# --- UI Initialization ---
@export(mod_name=GAME_NAME, name="init_config", initial=True)  # Kept original name
def init_ultimate_ttt_module(app: App):
    app.run_any(("CloudM", "add_ui"),
                name=GAME_NAME,
                title="Ultimate Tic-Tac-Toe",  # Simpler title
                path=f"/api/{GAME_NAME}/ui",
                description="Strategic Tic-Tac-Toe with nested grids."
                )
    app.logger.info(f"{GAME_NAME} module (v{VERSION}) initialized.")


# --- UI Endpoint ---
@get_app().tb(mod_name=GAME_NAME, version=VERSION, level=0, api=True, name="ui", state=False)
def ultimate_ttt_ui_page(app_ref: Optional[App] = None):
    app_instance = app_ref if app_ref else get_app(GAME_NAME)
    # Full HTML, CSS, and JS will be provided in the next message block
    html_and_js_content = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultimate Tic-Tac-Toe</title>
    <style>
        :root {
            --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";

            /* Light Theme */
            --bg-main-light: #f8f9fa; --bg-card-light: #ffffff; --text-primary-light: #212529;
            --text-secondary-light: #495057; --border-light: #dee2e6; --shadow-light: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
            --primary-light: #007bff; --primary-hover-light: #0056b3; --secondary-light: #6c757d;
            --danger-light: #dc3545; --success-light: #198754; --warning-light: #ffc107; --info-light: #0dcaf0;
            --text-on-primary-light: #fff; --cell-bg-light: #fff; --cell-hover-light: #e9ecef;
            --cell-border-light: #ced4da; --player-x-color-light: #dc3545; --player-o-color-light: #007bff;
            --local-won-x-bg-light: rgba(220, 53, 69, 0.1); --local-won-o-bg-light: rgba(0, 123, 255, 0.1);
            --local-draw-bg-light: rgba(108, 117, 125, 0.1);
            --forced-target-border-light: gold; --forced-target-shadow-light: 0 0 0 3px gold;
            --playable-anywhere-border-light: var(--primary-light);

            /* Dark Theme */
            --bg-main-dark: #121212; --bg-card-dark: #1e1e1e; --text-primary-dark: #e0e0e0;
            --text-secondary-dark: #a0a0a0; --border-dark: #3a3a3a; --shadow-dark: 0 0.125rem 0.25rem rgba(0,0,0,0.3);
            --primary-dark: #3b82f6; --primary-hover-dark: #2563eb; --secondary-dark: #868e96;
            --danger-dark: #f87171; --success-dark: #4ade80; --warning-dark: #facc15; --info-dark: #2dd4bf;
            --text-on-primary-dark: #fff; --cell-bg-dark: #2a2a2a; --cell-hover-dark: #383838;
            --cell-border-dark: #4f4f4f; --player-x-color-dark: #f87171; --player-o-color-dark: #60a5fa;
            --local-won-x-bg-dark: rgba(248, 113, 113, 0.15); --local-won-o-bg-dark: rgba(96, 165, 250, 0.15);
            --local-draw-bg-dark: rgba(134, 142, 150, 0.15);
            --forced-target-border-dark: #ffd700; --forced-target-shadow-dark: 0 0 0 3px #ffd700;
            --playable-anywhere-border-dark: var(--primary-dark);
        }

        html[data-theme="light"] {
            --bg-main: var(--bg-main-light); --bg-card: var(--bg-card-light); --text-primary: var(--text-primary-light);
            --text-secondary: var(--text-secondary-light); --border-color: var(--border-light);
            --shadow-color: var(--shadow-light); --primary-color: var(--primary-light);
            --primary-hover: var(--primary-hover-light); --secondary-color: var(--secondary-light);
            --danger-color: var(--danger-light); --success-color: var(--success-light);
            --warning-color: var(--warning-light); --info-color: var(--info-light);
            --text-on-primary: var(--text-on-primary-light); --cell-bg: var(--cell-bg-light);
            --cell-hover: var(--cell-hover-light); --cell-border: var(--cell-border-light);
            --player-x-color: var(--player-x-color-light); --player-o-color: var(--player-o-color-light);
            --local-won-x-bg: var(--local-won-x-bg-light); --local-won-o-bg: var(--local-won-o-bg-light);
            --local-draw-bg: var(--local-draw-bg-light); --forced-target-border: var(--forced-target-border-light);
            --forced-target-shadow: var(--forced-target-shadow-light);
            --playable-anywhere-border: var(--playable-anywhere-border-light);
        }

        html[data-theme="dark"] {
            --bg-main: var(--bg-main-dark); --bg-card: var(--bg-card-dark); --text-primary: var(--text-primary-dark);
            --text-secondary: var(--text-secondary-dark); --border-color: var(--border-dark);
            --shadow-color: var(--shadow-dark); --primary-color: var(--primary-dark);
            --primary-hover: var(--primary-hover-dark); --secondary-color: var(--secondary-dark);
            --danger-color: var(--danger-dark); --success-color: var(--success-dark);
            --warning-color: var(--warning-dark); --info-color: var(--info-dark);
            --text-on-primary: var(--text-on-primary-dark); --cell-bg: var(--cell-bg-dark);
            --cell-hover: var(--cell-hover-dark); --cell-border: var(--cell-border-dark);
            --player-x-color: var(--player-x-color-dark); --player-o-color: var(--player-o-color-dark);
            --local-won-x-bg: var(--local-won-x-bg-dark); --local-won-o-bg: var(--local-won-o-bg-dark);
            --local-draw-bg: var(--local-draw-bg-dark); --forced-target-border: var(--forced-target-border-dark);
            --forced-target-shadow: var(--forced-target-shadow-dark);
            --playable-anywhere-border: var(--playable-anywhere-border-dark);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-sans); background-color: var(--bg-main); color: var(--text-primary);
            display: flex; flex-direction: column; align-items: center; padding: 1rem; min-height: 100vh;
        }
        .app-header {
            width: 100%; max-width: 900px; display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 1.5rem; padding: 0.5rem 0;
        }
        .app-title { font-size: clamp(1.8em, 4vw, 2.2em); color: var(--primary-color); margin: 0; font-weight: 600;}
        .theme-switcher {
            padding: 0.5rem 1rem; background-color: var(--bg-card); color: var(--text-primary);
            border: 1px solid var(--border-color); border-radius: 0.375rem; cursor: pointer;
            font-size: 0.9em; box-shadow: var(--shadow-color);
        }
        .theme-switcher:hover { opacity: 0.8; }

        .main-content-wrapper { display: flex; flex-direction: column; align-items: center; width: 100%; max-width: 900px; }
        .section-card {
            background-color: var(--bg-card); padding: clamp(1rem, 3vw, 1.5rem); border-radius: 0.5rem;
            box-shadow: var(--shadow-color); margin-bottom: 1.5rem; width: 100%; max-width: 500px;
            border: 1px solid var(--border-color);
        }
        .section-card h2, .section-card h3 {
            font-size: clamp(1.2em, 3vw, 1.5em); margin-bottom: 1rem; color: var(--primary-color);
            padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color);
        }
        .form-group { margin-bottom: 1rem; }
        .form-group label { display: block; font-weight: 500; margin-bottom: 0.5rem; color: var(--text-secondary); font-size: 0.9em;}
        .form-input, .form-select {
            width: 100%; padding: 0.6rem 0.75rem; border-radius: 0.375rem;
            border: 1px solid var(--border-color); background-color: var(--cell-bg); color: var(--text-primary);
            font-size: 0.95em;
        }
        .form-input:focus, .form-select:focus { outline: none; border-color: var(--primary-color); box-shadow: 0 0 0 0.15rem rgba(var(--primary-colorRGB, 0,123,255), 0.25); }

        .button-row { display: flex; gap: 0.75rem; margin-top: 1rem; flex-wrap: wrap; }
        .button {
            padding: 0.6rem 1.2rem; border-radius: 0.375rem; border: none; cursor: pointer;
            font-weight: 500; font-size: 0.9em; transition: background-color 0.2s, transform 0.1s;
            text-align: center; display: inline-flex; align-items: center; justify-content: center;
            background-color: var(--primary-color); color: var(--text-on-primary);
        }
        .button:hover { background-color: var(--primary-hover); }
        .button:active { transform: translateY(1px); }
        .button.secondary { background-color: var(--secondary-color); }
        .button.secondary:hover { filter: brightness(0.9); }
        .button.danger { background-color: var(--danger-color); }
        .button.danger:hover { filter: brightness(0.9); }
        .button:disabled { opacity: 0.6; cursor: not-allowed; }

        .status-bar {
            font-size: clamp(0.95em, 2.5vw, 1.1em); font-weight: 500; text-align: center;
            padding: 0.75rem 1rem; border-radius: 0.375rem; min-height: 48px;
            display: flex; align-items: center; justify-content: center;
            border: 1px solid transparent; margin-bottom: 1rem;
        }
        .status-bar.info { background-color: var(--info-color); color: var(--text-on-primary); border-color: var(--info-color); opacity:0.9;}
        .status-bar.error { background-color: var(--danger-color); color: var(--text-on-primary); border-color: var(--danger-color); }
        .status-bar.success { background-color: var(--success-color); color: var(--text-on-primary); border-color: var(--success-color); }

        .game-board-area { width: 100%; display: flex; flex-direction: column; align-items: center; }
        .global-grid-display {
            display: grid; border: 3px solid var(--border-color-strong);
            background-color: var(--bg-card); box-shadow: var(--shadow-color);
            border-radius: 0.5rem; padding: clamp(3px, 1vw, 5px); gap: clamp(3px, 1vw, 5px);
        }
        .local-board-container {
            border: 2px solid var(--border-color); display: flex; align-items: center; justify-content: center;
            transition: box-shadow 0.2s, border-color 0.2s, background-color 0.2s; position: relative;
        }
        .local-board-container.forced-target {
            box-shadow: var(--forced-target-shadow); border-color: var(--forced-target-border) !important;
        }
        .local-board-container.playable-anywhere { border-color: var(--playable-anywhere-border) !important; opacity: 0.9; }

        .local-board-container.won-X { background-color: var(--local-won-x-bg); }
        .local-board-container.won-O { background-color: var(--local-won-o-bg); }
        .local-board-container.won-DRAW { background-color: var(--local-draw-bg); }

        .local-board-container .winner-overlay {
            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; pointer-events: none; opacity: 0;
        }
        .local-board-container.won-X .winner-overlay.player-X,
        .local-board-container.won-O .winner-overlay.player-O,
        .local-board-container.won-DRAW .winner-overlay.draw { opacity: 1; }
        .winner-overlay.player-X { color: var(--player-x-color); }
        .winner-overlay.player-O { color: var(--player-o-color); }
        .winner-overlay.draw { color: var(--text-secondary); }

        .local-grid { display: grid; width: 100%; height: 100%; gap: clamp(1px, 0.5vw, 2px); }
        .cell {
            border: 1px solid var(--cell-border); display: flex; align-items: center; justify-content: center;
            font-weight: bold; background-color: var(--cell-bg);
            transition: background-color 0.1s; cursor: default; aspect-ratio: 1/1;
        }
        .cell.playable { cursor: pointer; }
        .cell.playable:hover { background-color: var(--cell-hover); }
        .cell.player-X { color: var(--player-x-color); }
        .cell.player-O { color: var(--player-o-color); }

        .local-board-container.won-X .local-grid,
        .local-board-container.won-O .local-grid,
        .local-board-container.won-DRAW .local-grid { opacity: 0.5; }
        .local-board-container.won-X .cell, .local-board-container.won-O .cell, .local-board-container.won-DRAW .cell {
            cursor: not-allowed !important; background-color: transparent !important;
        }

        .game-controls-ingame { margin-top: 1.5rem; display: flex; gap: 0.75rem; justify-content: center; }
        .stats-area { text-align: center; }
        .stats-area p { margin: 0.2rem 0; font-size: 0.9em; color: var(--text-secondary); }
        .stats-area strong { color: var(--text-primary); font-weight: 600; }
        .hidden { display: none !important; }

        .local-board-container.inactive-target {
            opacity: 0.6; /* Dim boards that are not the current target */
            /* You might want to visually differentiate more, e.g., a slightly different border */
            /* border-color: var(--text-secondary) !important; */ /* Example */
        }

        .local-board-container.inactive-target .cell.playable {
            /* This should ideally not happen - if a board is inactive, its cells shouldn't be playable.
               But as a safeguard for styling if .playable somehow gets added. */
            cursor: not-allowed;
            background-color: var(--disabled-cell-bg) !important; /* From your theme */
        }
        .local-board-container.inactive-target .cell:not(.player-X):not(.player-O):hover {
            background-color: var(--cell-bg); /* Prevent hover effect on inactive boards' empty cells */
        }

        .modal-overlay {
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(0,0,0,0.6); display: flex;
            align-items: center; justify-content: center; z-index: 1050;
        }
        .modal-content {
            background-color: var(--bg-card); padding: 1.5rem 2rem; border-radius: 0.5rem;
            box-shadow: var(--shadow-color); max-width: 420px; width: 90%; text-align: center;
            border: 1px solid var(--border-color);
        }
        .modal-title { font-size: 1.3em; color: var(--primary-color); margin-bottom: 1rem; font-weight: 600;}
        .modal-message { margin-bottom: 1.5rem; font-size: 1em; color: var(--text-secondary); line-height: 1.5; }
        .modal-buttons { display: flex; gap: 0.75rem; justify-content: flex-end; }

        @media (max-width: 600px) {
            .app-header { flex-direction: column; gap: 0.75rem; }
            .button-row { flex-direction: column; }
            .button { width: 100%; }
            .button:not(:last-child) { margin-bottom: 0.5rem; }
        }
    </style>
</head>
<body>
    <header class="app-header">
        <h1 class="app-title">Ultimate TTT</h1>
        <button id="themeToggleBtn" class="theme-switcher">Dark Mode</button>
    </header>

    <main class="main-content-wrapper">
        <section id="gameSetupSection" class="section-card">
            <h2>New Game</h2>
            <div class="form-group">
                <label for="gridSizeSelect">Grid Size (N x N):</label>
                <select id="gridSizeSelect" class="form-select">
                    <option value="2">2x2</option>
                    <option value="3" selected>3x3 (Classic)</option>
                    <option value="4">4x4</option>
                    <option value="5">5x5</option>
                </select>
            </div>
            <div class="form-group">
                <label for="player1NameInput">Player 1 (X) Name:</label>
                <input type="text" id="player1NameInput" class="form-input" value="Player X">
            </div>
            <div id="localP2NameGroup" class="form-group">
                <label for="player2NameInput">Player 2 (O) Name:</label>
                <input type="text" id="player2NameInput" class="form-input" value="Player O">
            </div>

            <div class="button-row">
                <button id="startLocalGameBtn" class="button">Play Local Game</button>
                <button id="startOnlineGameBtn" class="button secondary">Create Online Game</button>
            </div>
            <div class="form-group" style="margin-top: 1.5rem;">
                <label for="joinGameIdInput">Join Online Game by ID:</label>
                <div style="display: flex; gap: 0.5rem;">
                    <input type="text" id="joinGameIdInput" class="form-input" placeholder="Enter Game ID">
                    <button id="joinOnlineGameBtn" class="button secondary">Join</button>
                </div>
            </div>
        </section>

        <section id="onlineWaitSection" class="section-card hidden">
            <h2>Online Game Lobby</h2>
            <p>Share this Game ID with your opponent:</p>
            <div id="gameIdShare" class="form-input" style="font-weight: bold; text-align: center; margin: 0.75rem 0; cursor: pointer; background-color: var(--cell-hover);" title="Click to copy Game ID">GAME_ID_HERE</div>
            <p id="waitingStatus" style="margin-top: 0.5rem; font-style: italic;">Waiting for opponent to join...</p>
            <div class="button-row" style="margin-top: 1.5rem;">
                 <button id="cancelOnlineWaitBtn" class="button danger">Cancel Game</button>
            </div>
        </section>

        <section id="statsSection" class="section-card stats-area">
            <h3>Session Stats (<span id="statsSessionId"></span>)</h3>
            <p>Played: <strong id="statsGamesPlayed">0</strong></p>
            <p>Wins: <strong id="statsWins">0</strong></p>
            <p>Losses: <strong id="statsLosses">0</strong></p>
            <p>Draws: <strong id="statsDraws">0</strong></p>
        </section>

        <section id="gameSection" class="game-board-area hidden">
            <div id="statusBar" class="status-bar info">Loading game...</div>
            <div id="globalGridDisplay" class="global-grid-display"></div>
            <div class="game-controls-ingame section-card button-row" style="max-width: none; margin-top: 1rem;">
                <button id="resetGameBtn" class="button danger">Reset Game</button>
                <button id="backToMenuBtn" class="button secondary">Back to Menu</button>
            </div>
        </section>
    </main>

    <div id="modalOverlay" class="modal-overlay hidden">
        <div class="modal-content">
            <h3 id="modalTitle" class="modal-title">Modal Title</h3>
            <p id="modalMessage" class="modal-message">Modal message.</p>
            <div class="modal-buttons">
                <button id="modalCancelBtn" class="button secondary">Cancel</button>
                <button id="modalConfirmBtn" class="button">Confirm</button>
            </div>
        </div>
    </div>

    <script unsave="true">
    (function() {
        "use strict";

        let gameSetupSection, statsSection, onlineWaitSection, gameArea, statusBar, globalGridDisplay;
        let gridSizeSelect, player1NameInput, player2NameInput, localP2NameGroup;
        let startLocalGameBtn, startOnlineGameBtn, joinGameIdInput, joinOnlineGameBtn, gameIdShareEl, waitingStatusEl, cancelOnlineWaitBtn;
        let resetGameBtn, backToMenuBtn, themeToggleBtn;
        let statsGamesPlayedEl, statsWinsEl, statsLossesEl, statsDrawsEl, statsSessionIdEl;
        let modalOverlay, modalTitle, modalMessage, modalCancelBtn, modalConfirmBtn;
        let modalConfirmCallback = null;

        let currentSessionId = null;
        let currentGameId = null;
        let currentGameState = null;
        let clientPlayerInfo = null;
        let localPlayerActiveSymbol = 'X';
        let onlineGamePollInterval = null;
        const POLLING_RATE_MS = 2500; // Increased polling rate slightly
        const API_MODULE_NAME = "UltimateTTT";

        const LOCAL_P1_ID = "p1_local_utt";
        const LOCAL_P2_ID = "p2_local_utt";

        function initApp() {
            console.log("UTTT Initializing...");
            gameSetupSection = document.getElementById('gameSetupSection');
            statsSection = document.getElementById('statsSection');
            onlineWaitSection = document.getElementById('onlineWaitSection');
            gameArea = document.getElementById('gameSection'); // Corrected ID
            statusBar = document.getElementById('statusBar');
            globalGridDisplay = document.getElementById('globalGridDisplay');

            gridSizeSelect = document.getElementById('gridSizeSelect');
            player1NameInput = document.getElementById('player1NameInput');
            player2NameInput = document.getElementById('player2NameInput');
            localP2NameGroup = document.getElementById('localP2NameGroup');

            startLocalGameBtn = document.getElementById('startLocalGameBtn');
            startOnlineGameBtn = document.getElementById('startOnlineGameBtn');
            joinGameIdInput = document.getElementById('joinGameIdInput');
            joinOnlineGameBtn = document.getElementById('joinOnlineGameBtn');
            gameIdShareEl = document.getElementById('gameIdShare');
            waitingStatusEl = document.getElementById('waitingStatus');
            cancelOnlineWaitBtn = document.getElementById('cancelOnlineWaitBtn');

            resetGameBtn = document.getElementById('resetGameBtn');
            backToMenuBtn = document.getElementById('backToMenuBtn');
            themeToggleBtn = document.getElementById('themeToggleBtn');

            statsGamesPlayedEl = document.getElementById('statsGamesPlayed');
            statsWinsEl = document.getElementById('statsWins');
            statsLossesEl = document.getElementById('statsLosses');
            statsDrawsEl = document.getElementById('statsDraws');
            statsSessionIdEl = document.getElementById('statsSessionId');

            modalOverlay = document.getElementById('modalOverlay');
            modalTitle = document.getElementById('modalTitle');
            modalMessage = document.getElementById('modalMessage');
            modalCancelBtn = document.getElementById('modalCancelBtn');
            modalConfirmBtn = document.getElementById('modalConfirmBtn');

            setupEventListeners();
            initializeTheme();
            determineSessionId();
            loadSessionStats();
            checkUrlForJoin();
            showScreen('gameSetup');
        }

        function setupEventListeners() {
            startLocalGameBtn.addEventListener('click', () => createNewGame('local'));
            startOnlineGameBtn.addEventListener('click', () => createNewGame('online'));
            joinOnlineGameBtn.addEventListener('click', joinOnlineGame);
            gameIdShareEl.addEventListener('click', copyGameIdToClipboard);
            cancelOnlineWaitBtn.addEventListener('click', () => {
                // TODO: API call to abort game if P1 created it and cancels
                showScreen('gameSetup');
                stopOnlinePolling();
            });

            resetGameBtn.addEventListener('click', confirmResetGame);
            backToMenuBtn.addEventListener('click', confirmBackToMenu);
            themeToggleBtn.addEventListener('click', toggleTheme);
            globalGridDisplay.addEventListener('click', onBoardClickDelegation);

            modalCancelBtn.addEventListener('click', hideModal);
            modalConfirmBtn.addEventListener('click', () => {
                if (modalConfirmCallback) modalConfirmCallback();
                hideModal();
            });
        }

        // Event listener (already in your full JS, ensure it's correctly placed)
        function onBoardClickDelegation(event) {
            const cell = event.target.closest('.cell.playable'); // Only react to clicks on playable cells
            if (cell && currentGameState && currentGameState.status === 'in_progress') {
                const gr = parseInt(cell.dataset.gr);
                const gc = parseInt(cell.dataset.gc);
                const lr = parseInt(cell.dataset.lr);
                const lc = parseInt(cell.dataset.lc);

                // Additional check: ensure the board containing this cell is indeed an active target
                // This is a safeguard, as .playable should only be on cells in active boards.
                const parentBoardContainer = cell.closest('.local-board-container');
                if (parentBoardContainer &&
                    (parentBoardContainer.classList.contains('forced-target') || parentBoardContainer.classList.contains('playable-anywhere'))) {
                    console.log(`CLICK_DELEGATION - Clicked playable cell: Global (${gr},${gc}), Local (${lr},${lc})`);
                    makePlayerMove(gr, gc, lr, lc);
                } else {
                    console.warn("CLICK_DELEGATION - Clicked a .playable cell not in an active target board. This shouldn't happen.", cell);
                }
            } else if (cell && (!currentGameState || currentGameState.status !== 'in_progress')) {
                console.log("CLICK_DELEGATION - Clicked a cell, but game not in progress or no game state.");
            }
}

        function determineSessionId() {
            if (window.TB?.user?.getUid && typeof window.TB.user.getUid === 'function') {
                const uid = window.TB.user.getUid();
                if (uid) currentSessionId = uid;
            }
            if (!currentSessionId) {
                currentSessionId = localStorage.getItem('uttt_guest_session_id');
                if (!currentSessionId) {
                    currentSessionId = 'guest_uttt_' + Date.now().toString(36) + Math.random().toString(36).substring(2, 7);
                    localStorage.setItem('uttt_guest_session_id', currentSessionId);
                }
            }
            if(statsSessionIdEl) statsSessionIdEl.textContent = currentSessionId.substring(0, 12) + "...";
            console.log("Session ID for stats/online:", currentSessionId);
        }

        function checkUrlForJoin() {
            const urlParams = new URLSearchParams(window.location.search);
            const gameIdToJoin = urlParams.get('join_game_id');
            if (gameIdToJoin) {
                joinGameIdInput.value = gameIdToJoin;
                if (window.TB?.ui?.Toast) TB.ui.Toast.showInfo(`Attempting to join game ${gameIdToJoin}...`, { duration: 2000 });
                joinOnlineGame();
                window.history.replaceState({}, document.title, window.location.pathname);
            }
        }

        function showScreen(screenName) {
            console.log("Showing screen:", screenName);
            ['gameSetup', 'onlineWait', 'game'].forEach(name => {
                document.getElementById(name + 'Section')?.classList.add('hidden');
            });
            const targetScreen = document.getElementById(screenName + 'Section');
            if (targetScreen) {
                targetScreen.classList.remove('hidden');
            }

            if (screenName === 'gameSetup') {
                statsSection?.classList.remove('hidden');
                localP2NameGroup?.classList.remove('hidden');
                stopOnlinePolling();
                currentGameId = null; currentGameState = null; clientPlayerInfo = null;
            } else {
                statsSection?.classList.add('hidden');
                if (screenName === 'onlineWait' && localP2NameGroup) {
                    localP2NameGroup.classList.add('hidden');
                }
            }
        }

        function initializeTheme() { /* Same as previous */
            const savedTheme = localStorage.getItem('uttt_theme') ||
                               (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
            document.documentElement.setAttribute('data-theme', savedTheme);
            updateThemeButtonText(savedTheme);
        }
        function toggleTheme() { /* Same as previous */
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('uttt_theme', newTheme);
            updateThemeButtonText(newTheme);
        }
        function updateThemeButtonText(theme) { /* Same as previous */
             themeToggleBtn.textContent = theme === 'light' ? '🌙 Dark Mode' : '☀️ Light Mode';
        }

        async function apiRequest(endpoint, payload = null, method = 'GET', queryParams = {}) {
            let url = `/api/${API_MODULE_NAME}/${endpoint}`; // Ensure module name is correct
            if (method === 'GET' && payload) { // For GET, payload becomes queryParams
                 queryParams = {...queryParams, ...payload};
                 payload = null;
            }

            if ((endpoint.startsWith('get_game_state') || endpoint.startsWith('make_move')) && queryParams.game_id) {
                 url = `/api/${API_MODULE_NAME}/${endpoint.split('/')[0]}/${queryParams.game_id}`; // Construct path
                 delete queryParams.game_id; // Remove from general query if used in path
            }


            if (!window.TB?.api?.request) {
                console.error("TB.api.request not available.");
                showModal("API Error", "Framework error: Cannot communicate with server.", null, "OK", "");
                return { error: true, message: "API_UNAVAILABLE" };
            }
            if(window.TB?.ui?.Loader) TB.ui.Loader.show({text: "Processing...", hideMainContent:false, playAnimation: "Y2+41:R2+61"});
            try {
                // Toolbox `request` function: moduleName, toolName, data, method, options ({queryParams})
                const response = await window.TB.api.request(API_MODULE_NAME, endpoint, payload, method, {queryParams});
                if(window.TB?.ui?.Loader) TB.ui.Loader.hide();

                if (response.error !== window.TB.ToolBoxError.none) {
                    const errorMsg = response.info?.help_text || response.data?.message || `API Error (${response.error})`;
                    console.error(`API Error [${endpoint}]:`, errorMsg, response);
                    if(window.TB?.ui?.Toast) TB.ui.Toast.showError(errorMsg.substring(0,150), {duration: 4000});
                    return { error: true, message: errorMsg, data: response.get() }; // response.get() might be null or error object
                }
                return { error: false, data: response.get() };
            } catch (err) {
                if(window.TB?.ui?.Loader) TB.ui.Loader.hide();
                console.error(`Network/JS Error [${endpoint}]:`, err);
                if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Network or application error.", {duration: 4000});
                return { error: true, message: "NETWORK_ERROR" };
            }
        }

         async function createNewGame(mode) {
            const size = parseInt(gridSizeSelect.value);
            const config = { grid_size: size };
            const p1Name = player1NameInput.value.trim() || (mode === 'local' ? "Player X" : "Me");

            const payload = { config, mode, player1_name: p1Name };
            if (mode === 'local') {
                payload.player2_name = player2NameInput.value.trim() || "Player O";
            }

            const response = await apiRequest('create_game', payload, 'POST', {hideMainContentWhileLoading: true});

            if (!response.error && response.data?.game_id) {
                currentGameState = response.data;
                currentGameId = currentGameState.game_id;

                if (mode === 'local') {
                    // clientPlayerInfo is not used for local turn logic, localPlayerActiveSymbol handles display
                    processGameStateUpdate(currentGameState); // Sets localPlayerActiveSymbol
                    showScreen('game');
                } else if (mode === 'online') {
                    // Creator is P1. Their clientPlayerInfo is based on currentSessionId (which should match player.id from server)
                    clientPlayerInfo = currentGameState.players.find(p => p.id === currentSessionId);
                    if (!clientPlayerInfo && currentGameState.players.length > 0) {
                        clientPlayerInfo = currentGameState.players[0]; // Fallback if ID mismatch
                        console.warn("Online game created: Forcing clientPlayerInfo to P1 from server. SessionID:", currentSessionId, "P1 ID:", clientPlayerInfo.id);
                    }
                    gameIdShareEl.textContent = currentGameId;
                    waitingStatusEl.textContent = `Waiting for opponent... Game ID: ${currentGameId}`;
                    showScreen('onlineWait');
                    startOnlinePolling();
                }
            }
        }

        async function joinOnlineGame() {
            const gameIdToJoin = joinGameIdInput.value.trim();
            const playerName = player1NameInput.value.trim() || "Challenger";
            if (!gameIdToJoin) {
                if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Please enter a Game ID."); return;
            }

            const response = await apiRequest('join_game', { game_id: gameIdToJoin, player_name: playerName }, 'POST', {hideMainContentWhileLoading: true});

            if (!response.error && response.data?.game_id) {
                currentGameState = response.data;
                currentGameId = currentGameState.game_id;
                // The joiner becomes a player. Their clientPlayerInfo is based on currentSessionId.
                clientPlayerInfo = currentGameState.players.find(p => p.id === currentSessionId);
                if (!clientPlayerInfo && currentGameState.players.length === 2) {
                    // If currentSessionId didn't match (e.g. guest ID regeneration, or different browser used link)
                    // The server assigned a new guest_id to this joiner, or used their Toolbox UID.
                    // We need to find which of the two players is "me" (the one not already identified as P1 if P1 is known)
                    // Or simply the one whose ID is our currentSessionId.
                    // The backend's add_player now handles making the joiner the second player.
                     console.warn("Online game joined: My session ID:", currentSessionId, "Players in game:", currentGameState.players);
                     if(!clientPlayerInfo) clientPlayerInfo = currentGameState.players[1]; // Assume I am P2 if my session ID didn't match for some reason.
                } else if (!clientPlayerInfo) {
                     console.error("Could not identify client player after joining.");
                }

                processGameStateUpdate(currentGameState);
                // processGameStateUpdate will handle screen transition if game becomes IN_PROGRESS
                if (currentGameState.status === 'in_progress') {
                     showScreen('game'); // Explicitly show game if already in progress
                     startOnlinePolling();
                } else if (currentGameState.status === 'waiting_for_opponent'){
                    // This means I joined but the other player (P1) might have disconnected or something unusual
                    waitingStatusEl.textContent = `Joined. Still waiting for P1 or game to start... Game ID: ${currentGameId}`;
                    showScreen('onlineWait');
                    startOnlinePolling(); // Poll for game start
                } else {
                     // Game might be full, finished, or aborted
                     if(window.TB?.ui?.Toast) TB.ui.Toast.showError(currentGameState.last_error_message || "Could not join game (unexpected status).");
                     showScreen('gameSetup');
                }
            }
        }

        function processGameStateUpdate(newGameState) {
            console.log("PROCESS_GAME_STATE_UPDATE - Received:", newGameState);
            const oldStatus = currentGameState ? currentGameState.status : null;
            const previousPlayerId = currentGameState ? currentGameState.current_player_id : null;
            currentGameState = newGameState;
            if (!currentGameState || !currentGameState.game_id) {
                console.error("Invalid newGameState received!");
                if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Error: Corrupted game update.");
                return;
            }

        if (currentGameState.mode === 'local') {
            const currentPlayer = currentGameState.players.find(p => p.id === currentGameState.current_player_id);
            localPlayerActiveSymbol = currentPlayer ? currentPlayer.symbol : '?';
        } else if (currentGameState.mode === 'online') {
            if (!clientPlayerInfo || !currentGameState.players.find(p => p.id === clientPlayerInfo.id)) {
                 clientPlayerInfo = currentGameState.players.find(p => p.id === currentSessionId);
                 console.log("PROCESS_GAME_STATE_UPDATE (Online) - ClientPlayerInfo (re)set:", clientPlayerInfo);
            }

            // If this client was on the "onlineWaitScreen" and game is now "in_progress"
            const onlineWaitScreenActive = !document.getElementById('onlineWaitSection').classList.contains('hidden');
            if (onlineWaitScreenActive && currentGameState.status === 'in_progress') {
                if(window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("Opponent connected! Game starting.", {duration: 2000});
                showScreen('game');
            }
            if (currentGameState.status === 'aborted') {
                 showModal("Game Aborted", currentGameState.last_error_message || "Opponent did not join or left.", () => showScreen('gameSetup'));
                 stopOnlinePolling();
            }
        }

        renderBoard();
        updateStatusBar();

        if (currentGameState.status === 'finished') {
            stopOnlinePolling();
            showGameOverModal();
            loadSessionStats();
        } else if (currentGameState.mode === 'online' && currentGameState.status === 'in_progress') {
            const isMyTurnOnline = clientPlayerInfo && currentGameState.current_player_id === clientPlayerInfo.id;
            if (!isMyTurnOnline) {
                startOnlinePolling(); // Opponent's turn, keep polling
            } else {
                stopOnlinePolling(); // My turn, stop polling
                if (previousPlayerId !== currentGameState.current_player_id && oldStatus === 'in_progress') {
                    // If turn just switched to me
                    if(window.TB?.ui?.Toast) TB.ui.Toast.showInfo("It's your turn!", {duration: 2000});
                }
            }
        }
    }

    async function makePlayerMove(globalR, globalC, localR, localC) {
        if (!currentGameState || !currentGameId || currentGameState.status !== 'in_progress') return;

        let playerIdForMove;
        if (currentGameState.mode === 'local') {
            playerIdForMove = currentGameState.current_player_id;
        } else {
            if (!clientPlayerInfo || currentGameState.current_player_id !== clientPlayerInfo.id) {
                if(window.TB?.ui?.Toast) TB.ui.Toast.showInfo("Not your turn."); return;
            }
            playerIdForMove = clientPlayerInfo.id;
        }

            const movePayload = {
                player_id: playerIdForMove, global_row: globalR, global_col: globalC,
                local_row: localR, local_col: localC,game_id:currentGameId
            };

            if (globalGridDisplay) globalGridDisplay.style.pointerEvents = 'none';
            // Pass game_id as path parameter for make_move
            const response = await apiRequest(`make_move`, movePayload, 'POST');
  if (globalGridDisplay) globalGridDisplay.style.pointerEvents = 'auto';

        if (!response.error && response.data) {
            processGameStateUpdate(response.data);
        } else if (response.data?.game_id) {
            processGameStateUpdate(response.data);
        }
    }

    function startOnlinePolling() {
        stopOnlinePolling();
        if (currentGameState && currentGameState.mode === 'online' &&
            (currentGameState.status === 'waiting_for_opponent' ||
             (currentGameState.status === 'in_progress' && (!clientPlayerInfo || currentGameState.current_player_id !== clientPlayerInfo.id))
            )
           ) {
            console.log("Starting online polling for game:", currentGameId, "Status:", currentGameState.status);
            onlineGamePollInterval = setInterval(fetchCurrentGameState, POLLING_RATE_MS);
        } else {
            // console.log("Not starting polling. Conditions not met.");
        }
    }

        function confirmResetGame() { /* Same as before */
            if (!currentGameState) return;
            showModal('Reset Game?', 'Start a new game with current settings?',
                async () => {
                    if (currentGameState.mode === 'local') {
                        await createNewGame('local'); // Uses current form values
                    } else {
                        if(window.TB?.ui?.Toast) TB.ui.Toast.showInfo("For online, please start a new game from the menu to reset.");
                    }
                }
            );
        }
        function confirmBackToMenu() { /* Same as before */
            if (currentGameState && currentGameState.status !== 'finished') {
                showModal('Back to Menu?', 'Progress will be lost. Sure?', () => showScreen('gameSetup'));
            } else {
                showScreen('gameSetup');
            }
        }

        function startOnlinePolling() {
            stopOnlinePolling(); // Clear existing before starting a new one
            if (currentGameState && currentGameState.mode === 'online' &&
                (currentGameState.status === 'waiting_for_opponent' ||
                 (currentGameState.status === 'in_progress' && (!clientPlayerInfo || currentGameState.current_player_id !== clientPlayerInfo.id))
                )
               ) {
                console.log("Starting online polling for game:", currentGameId, "Status:", currentGameState.status);
                onlineGamePollInterval = setInterval(fetchCurrentGameState, POLLING_RATE_MS);
            } else {
                console.log("Not starting online polling. Mode:", currentGameState?.mode, "Status:", currentGameState?.status, "Is my turn (if online):", clientPlayerInfo && currentGameState?.current_player_id === clientPlayerInfo?.id);
            }
        }
        function stopOnlinePolling() { /* Same as before */
            if (onlineGamePollInterval) clearInterval(onlineGamePollInterval);
            onlineGamePollInterval = null;
        }
        async function fetchCurrentGameState(forceUpdate=true) {
              if (!currentGameId || !currentGameState || currentGameState.mode !== 'online') {
                stopOnlinePolling(); return;
                }

                const isMyTurnOnline = clientPlayerInfo && currentGameState.current_player_id === clientPlayerInfo.id;
                if (currentGameState.status === 'in_progress' && isMyTurnOnline && !forceUpdate) {
                     return;
                }
            // Pass game_id as path parameter for get_game_state
            const response = await apiRequest(`get_game_state?game_id=${currentGameId}`, null, 'GET');

                if (!response.error && response.data) {
                    if (forceUpdate || !currentGameState || response.data.updated_at !== currentGameState.updated_at || response.data.status !== currentGameState.status) {
                        processGameStateUpdate(response.data);
                    }
                } else if (response.error && response.message !== "API_UNAVAILABLE" && response.message !== "NETWORK_ERROR") {
                    stopOnlinePolling();
                    if(window.TB?.ui?.Toast && currentGameState.status !== 'finished' && currentGameState.status !== 'aborted') {
                        TB.ui.Toast.showError("Comms error. Polling stopped.", {duration: 2000});
                    }
                }
            }


        async function loadSessionStats() { /* Same as before */
            const response = await apiRequest('get_session_stats', {session_id: currentSessionId}, 'GET');
            if (!response.error && response.data) updateStatsDisplay(response.data);
            else updateStatsDisplay({ games_played:0, wins:0, losses:0, draws:0 });
        }
        function updateStatsDisplay(stats) { /* Same as before */
            statsGamesPlayedEl.textContent = stats.games_played ?? 0;
            statsWinsEl.textContent = stats.wins ?? 0;
            statsLossesEl.textContent = stats.losses ?? 0;
            statsDrawsEl.textContent = stats.draws ?? 0;
        }

        function renderBoard() {
            if (!currentGameState || !globalGridDisplay) return;
            const N = currentGameState.config.grid_size;
            dynamicallySetGridStyles(N); // Ensure styles are set for current N
            globalGridDisplay.innerHTML = '';

            console.log("RENDER_BOARD - Forced target from state:", currentGameState.next_forced_global_board);

            for (let gr = 0; gr < N; gr++) {
                for (let gc = 0; gc < N; gc++) {
                    const localBoardContainer = document.createElement('div');
                    localBoardContainer.className = 'local-board-container';
                    localBoardContainer.dataset.gr = gr; localBoardContainer.dataset.gc = gc;

                    const localWinner = currentGameState.global_board_winners[gr][gc];
                    if (localWinner !== 'NONE') {
                        localBoardContainer.classList.add('won-' + localWinner);
                        localBoardContainer.classList.add('won-' + localWinner);
                        const overlay = document.createElement('div');
                        overlay.className = 'winner-overlay player-' + (localWinner === 'DRAW' ? 'draw' : localWinner);
                        overlay.textContent = localWinner === 'DRAW' ? 'D' : localWinner;
                        localBoardContainer.appendChild(overlay);
                    }

                    let isThisBoardTheActiveTarget = false;
                    if (currentGameState.status === 'in_progress' && localWinner === 'NONE') {
                        const forcedTarget = currentGameState.next_forced_global_board;
                        if (forcedTarget) { // A specific board is forced
                            if (forcedTarget[0] === gr && forcedTarget[1] === gc) {
                                localBoardContainer.classList.add('forced-target');
                                isThisBoardTheActiveTarget = true;
                                console.log(`RENDER_BOARD - Board (${gr},${gc}) is FORCED target.`);
                            } else {
                                // Not the forced target, so visually dim or mark as inactive
                                localBoardContainer.classList.add('inactive-target'); // Add a new CSS class for this
                            }
                        } else { // No specific board forced - play anywhere valid
                            localBoardContainer.classList.add('playable-anywhere');
                            // isThisBoardTheActiveTarget remains true for all non-won boards if no forced target
                            isThisBoardTheActiveTarget = true;
                            console.log(`RENDER_BOARD - Board (${gr},${gc}) is playable (anywhere rule).`);
                        }
                    } else {
                         localBoardContainer.classList.add('inactive-target'); // Board is won or game over
                    }
                    // --- END: UI Helper for Guided Placement ---

                    const localGrid = document.createElement('div');
                    localGrid.className = 'local-grid';
                    const localCells = currentGameState.local_boards_state[gr][gc];

                    for (let lr = 0; lr < N; lr++) {
                        for (let lc = 0; lc < N; lc++) {
                            const cell = document.createElement('div');
                            cell.className = 'cell';
                            cell.dataset.gr = gr; cell.dataset.gc = gc;
                            cell.dataset.lr = lr; cell.dataset.lc = lc;

                            const cellState = localCells[lr][lc];
                            if (cellState !== '.') { // '.' is CellState.EMPTY
                                cell.textContent = cellState;
                                cell.classList.add('player-' + cellState);
                            }

                            // --- Determine if this specific cell is playable ---
                            let isCellCurrentlyPlayable = false;
                            if (currentGameState.status === 'in_progress' && localWinner === 'NONE' && cellState === '.') {
                                // Check if it's this client's turn to act
                                let isThisClientsTurnToAct = (currentGameState.mode === 'local') ||
                                                             (clientPlayerInfo && currentGameState.current_player_id === clientPlayerInfo.id);

                                if (isThisClientsTurnToAct) {
                                    if (isThisBoardTheActiveTarget) { // If this board is highlighted as active/forced
                                        isCellCurrentlyPlayable = true;
                                    }
                                }
                            }

                            if (isCellCurrentlyPlayable) {
                                cell.classList.add('playable');
                            } else {
                                // Could add a generic .disabled-cell class if not already implied by lack of .playable
                                // and the .inactive-target on the parent board.
                            }
                            localGrid.appendChild(cell);
                        }
                    }
                    localBoardContainer.appendChild(localGrid);
                    globalGridDisplay.appendChild(localBoardContainer);
                }
            }
            // No separate highlightGuidance() function is needed if renderBoard fully handles it.
        }

        function updateStatusBar() { /* Mostly same, ensure player names are used */
            if (!statusBar || !currentGameState) return;
            let message = ""; let msgType = "info";

            if (currentGameState.status === 'waiting_for_opponent') {
                message = "Waiting for opponent...";
            } else if (currentGameState.status === 'in_progress') {
                const currentPlayer = currentGameState.players.find(p => p.id === currentGameState.current_player_id);
                const pName = currentPlayer ? currentPlayer.name : "Player";
                const pSymbol = currentPlayer ? currentPlayer.symbol : "?";

                if (currentGameState.mode === 'local') {
                    message = `${pName} (${pSymbol})'s Turn.`;
                } else {
                    message = (clientPlayerInfo && clientPlayerInfo.id === currentGameState.current_player_id) ?
                              `Your Turn (${clientPlayerInfo.symbol})` :
                              `Waiting for ${pName} (${pSymbol})...`;
                }

                if (currentGameState.next_forced_global_board) {
                    const [gr, gc] = currentGameState.next_forced_global_board;
                    message += ` Play in highlighted board (${gr+1},${gc+1}).`;
                } else {
                    message += " Play in any valid highlighted board.";
                }
                if (currentGameState.last_error_message) {
                    message = `Error: ${currentGameState.last_error_message}`; msgType = "error";
                }
            } else if (currentGameState.status === 'finished') {
                msgType = "success";
                if (currentGameState.is_draw) message = "Game Over: It's a DRAW!";
                else {
                    const winner = currentGameState.players.find(p => p.symbol === currentGameState.overall_winner_symbol);
                    message = `Game Over: ${winner ? winner.name : 'Player'} (${currentGameState.overall_winner_symbol}) WINS!`;
                }
            } else if (currentGameState.status === 'aborted') {
                 message = currentGameState.last_error_message || "Game Aborted."; msgType = "error";
            }
            statusBar.textContent = message;
            statusBar.className = `status-bar ${msgType}`;
        }

        function showGameOverModal() { /* Same as before, ensure names used */
            let title = "Game Over!"; let content = "";
            if (currentGameState.is_draw) content = "The game ended in a DRAW!";
            else {
                const winner = currentGameState.players.find(p => p.symbol === currentGameState.overall_winner_symbol);
                content = `${winner ? winner.name : 'Player'} (${currentGameState.overall_winner_symbol}) is victorious!`;
            }
            showModal(title, content, () => createNewGame(currentGameState.mode), "Play Again", "Menu");
        }

        function copyGameIdToClipboard() { /* Same as before */
            const gameId = gameIdShareEl.textContent;
            if (navigator.clipboard && gameId) {
                navigator.clipboard.writeText(gameId)
                    .then(() => { if(window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("ID copied!", {duration:1500}); })
                    .catch(err => { if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Copy failed."); });
            } else if(window.TB?.ui?.Toast) TB.ui.Toast.showWarning("No ID to copy.");
        }

        function showModal(title, message, onConfirm = null, confirmText = "OK", cancelText = "Cancel") { /* Same */
            modalTitle.textContent = title; modalMessage.textContent = message;
            modalConfirmBtn.textContent = confirmText; modalCancelBtn.textContent = cancelText;
            modalConfirmCallback = onConfirm; modalOverlay.classList.remove('hidden');
            if (!onConfirm) modalConfirmBtn.classList.add('hidden');
            else modalConfirmBtn.classList.remove('hidden');
        }
        function hideModal() { /* Same */ modalOverlay.classList.add('hidden'); modalConfirmCallback = null; }

        function dynamicallySetGridStyles(N) { /* Same as before, check calculations carefully */
            if (!globalGridDisplay) return;
            const mainWrap = document.querySelector('.main-content-wrapper');
            const availableWidth = mainWrap ? mainWrap.offsetWidth - 20 : window.innerWidth - 40;

            let boardPixelSize = Math.min(availableWidth, window.innerHeight * 0.65, N * 100 + (N-1)*5);
            boardPixelSize = Math.max(N * 45 + (N-1)*2, boardPixelSize); // Min
            boardPixelSize = Math.min(boardPixelSize, 650); // Max

            globalGridDisplay.style.width = `${boardPixelSize}px`;
            globalGridDisplay.style.height = `${boardPixelSize}px`;

            const globalGap = Math.max(2, Math.floor(boardPixelSize / (N * 30)));
            globalGridDisplay.style.gap = `${globalGap}px`;
            globalGridDisplay.style.padding = `${globalGap}px`;

            const localBoardOuterSize = (boardPixelSize - (N - 1) * globalGap - 2 * globalGap) / N;
            const localBoardInnerSize = Math.max(10, localBoardOuterSize - (2*2)); // 2px border * 2

            const localCellGap = Math.max(1, Math.floor(localBoardInnerSize / (N * 40)));
            const estimatedCellSize = (localBoardInnerSize - (N - 1) * localCellGap) / N;

            const cellFontSize = Math.max(8, estimatedCellSize * 0.42 / Math.sqrt(N/2.5) );
            const winnerOverlayFontSize = Math.max(15, localBoardInnerSize * 0.55 / Math.sqrt(N/2.5) );
            const winnerOverlayDrawFontSize = Math.max(12, localBoardInnerSize * 0.35 / Math.sqrt(N/2.5) );

            let dynamicStyleSheet = document.getElementById('dynamicGameStylesUTTT');
            if (!dynamicStyleSheet) {
                dynamicStyleSheet = document.createElement('style');
                dynamicStyleSheet.id = 'dynamicGameStylesUTTT';
                document.head.appendChild(dynamicStyleSheet);
            }
            dynamicStyleSheet.innerHTML = `
                .global-grid-display { grid-template-columns: repeat(${N}, 1fr); grid-template-rows: repeat(${N}, 1fr); }
                .local-grid { grid-template-columns: repeat(${N}, 1fr); grid-template-rows: repeat(${N}, 1fr); gap: ${localCellGap}px; }
                .cell { font-size: ${cellFontSize}px !important; }
                .local-board-container .winner-overlay { font-size: ${winnerOverlayFontSize}px !important; }
                .local-board-container .winner-overlay.draw { font-size: ${winnerOverlayDrawFontSize}px !important; }
            `;
        }

        if (window.TB?.events) {
            if (window.TB.config?.get('appRootId') || window.TB._isInitialized === true) initApp();
            else window.TB.events.on('tbjs:initialized', initApp, { once: true });
        } else {
             console.warn("Toolbox not fully loaded, attempting init on DOMContentLoaded.");
             document.addEventListener('DOMContentLoaded', () => {
                if (window.TB?.events?.on) window.TB.events.on('tbjs:initialized', initApp, { once: true });
                else if (window.TB?._isInitialized) initApp();
                else console.error("CRITICAL: TB not available after DOMContentLoaded.");
             });
        }

    })();
    </script>
</body>
</html>"""
    return Result.html(app_instance.web_context() + html_and_js_content)

# --- END OF FILE ultimate_ttt_api.py ---
