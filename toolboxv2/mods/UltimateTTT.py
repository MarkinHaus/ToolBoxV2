# --- START OF FILE ultimate_ttt_api.py ---
import asyncio
import os
import uuid
from collections.abc import AsyncGenerator
# py > 3.11
try:
    from datetime import UTC, datetime, timedelta
except ImportError:
    from datetime import datetime, timedelta, timezone
    UTC = timezone.utc
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from toolboxv2 import App, RequestData, Result, get_app
from toolboxv2.utils.extras.base_widget import get_user_from_request

# --- Constants ---
GAME_NAME = Name = "UltimateTTT"  # Using your original name for consistency with JS API calls
VERSION = "3.1.0"  # Incremented for this revision
DB_GAMES_PREFIX = f"{GAME_NAME.lower()}_games"
DB_USER_STATS_PREFIX = f"{GAME_NAME.lower()}_user_stats"

LOCAL_PLAYER_X_ID = "p1_local_utt"  # Shortened for less verbosity
LOCAL_PLAYER_O_ID = "p2_local_utt"

ONLINE_POLL_TIMEOUT_SECONDS = 180  # For initial opponent join
PAUSED_GAME_RESUME_WINDOW_SECONDS = 24 * 60 * 60  # 24 hours for a paused game


export = get_app(f"{GAME_NAME}.Export").tb
MINIMAX_SEARCH_DEPTH = 4 # Adjust this for strength vs. speed (e.g., 4, 5, or 6)


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


class NPCDifficulty(str, Enum):
    NONE = "none"  # Indicates a human player
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    # INSANE = "insane" # Add if you implement it

NPC_PLAYER_ID_PREFIX = "npc_utt_"
NPC_EASY_ID = f"{NPC_PLAYER_ID_PREFIX}{NPCDifficulty.EASY.value}"
NPC_MEDIUM_ID = f"{NPC_PLAYER_ID_PREFIX}{NPCDifficulty.MEDIUM.value}"
NPC_HARD_ID = f"{NPC_PLAYER_ID_PREFIX}{NPCDifficulty.HARD.value}"

# --- Pydantic Models ---
class GameConfig(BaseModel):
    grid_size: int = Field(default=3, ge=2, le=5)  # Max 5x5 for UI sanity for now


class PlayerInfo(BaseModel):
    id: str
    symbol: PlayerSymbol
    name: str
    is_connected: bool = True
    is_npc: bool = False  # New field
    npc_difficulty: NPCDifficulty | None = None


class Move(BaseModel):
    player_id: str
    global_row: int
    global_col: int
    local_row: int
    local_col: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GameState(BaseModel):
    game_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: GameConfig
    mode: GameMode
    status: GameStatus

    players: list[PlayerInfo] = []
    current_player_id: str | None = None

    global_board_winners: list[list[BoardWinner]]
    local_boards_state: list[list[list[list[CellState]]]]

    last_made_move_coords: tuple[int, int, int, int] | None = None
    next_forced_global_board: tuple[int, int] | None = None  # If set, player MUST play here

    overall_winner_symbol: PlayerSymbol | None = None
    is_draw: bool = False

    moves_history: list[Move] = []
    last_error_message: str | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    waiting_since: datetime | None = None

    @model_validator(mode='before')
    def initialize_structures_for_gamestate(cls, values: dict[str, Any]) -> dict[str, Any]:  # Renamed validator
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

    def get_player_info(self, player_id: str) -> PlayerInfo | None:
        return next((p for p in self.players if p.id == player_id), None)

    def get_opponent_info(self, player_id: str) -> PlayerInfo | None:
        return next((p for p in self.players if p.id != player_id), None)

    def get_current_player_info(self) -> PlayerInfo | None:
        return self.get_player_info(self.current_player_id) if self.current_player_id else None

    def model_dump_for_api(self) -> dict[str, Any]:  # Renamed from your previous for clarity
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

    def _check_line_for_win(self, line: list[CellState | BoardWinner],
                            symbol_to_check: CellState | BoardWinner) -> bool:
        if not line or line[0] == CellState.EMPTY or line[0] == BoardWinner.NONE:
            return False
        return all(cell == symbol_to_check for cell in line)

    def _get_board_winner_symbol(self, board: list[list[CellState | BoardWinner]],
                                 symbol_class: type[CellState] | type[BoardWinner]) -> CellState | BoardWinner | None:
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

    def _is_board_full(self, board: list[list[CellState | BoardWinner]],
                       empty_value: CellState | BoardWinner) -> bool:
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

    def _determine_next_forced_board(self, last_move_local_r: int, last_move_local_c: int) -> tuple[int, int] | None:
        target_gr, target_gc = last_move_local_r, last_move_local_c

        if self.gs.global_board_winners[target_gr][target_gc] == BoardWinner.NONE and \
            not self._is_local_board_full(self.gs.local_boards_state[target_gr][target_gc], CellState.EMPTY):
            return (target_gr, target_gc)
        return None  # Play anywhere valid

    def _is_local_board_full(self, local_board_cells: list[list[CellState]], cell_type=CellState.EMPTY) -> bool:
        """Checks if a specific local board (passed as a 2D list of CellState) is full."""
        for r in range(self.size):
            for c in range(self.size):
                if local_board_cells[r][c] == cell_type:
                    return False
        return True

    def add_player(self, player_id: str, player_name: str,
                   is_npc: bool = False, npc_difficulty: NPCDifficulty | None = None) -> bool:
        if len(self.gs.players) >= 2:
            self.gs.last_error_message = "Game is already full (2 players max)."
            return False

        # Reconnect logic for existing player (human or NPC if that makes sense)
        existing_player = self.gs.get_player_info(player_id)
        if existing_player:
            if not existing_player.is_connected:
                existing_player.is_connected = True
                # If NPC "reconnects", ensure its properties are correct (though unlikely scenario for NPC)
                if is_npc:
                    existing_player.is_npc = True
                    existing_player.npc_difficulty = npc_difficulty
                    existing_player.name = player_name  # Update name if it changed for NPC

                self.gs.last_error_message = None
                self.gs.updated_at = datetime.now(UTC)

                if len(self.gs.players) == 2 and all(p.is_connected for p in self.gs.players) and \
                    self.gs.status == GameStatus.WAITING_FOR_OPPONENT:  # Should not be waiting if NPC is P2
                    self.gs.status = GameStatus.IN_PROGRESS
                    player_x_info = next(p for p in self.gs.players if p.symbol == PlayerSymbol.X)
                    self.gs.current_player_id = player_x_info.id
                    self.gs.waiting_since = None
                return True
            else:  # Player ID exists and is already connected
                self.gs.last_error_message = f"Player with ID {player_id} is already in the game and connected."
                return False

        # Adding a new player
        symbol = PlayerSymbol.X if not self.gs.players else PlayerSymbol.O

        # Construct PlayerInfo with NPC details if applicable
        player_info_data = {
            "id": player_id,
            "symbol": symbol,
            "name": player_name,
            "is_connected": True,  # NPCs are always "connected"
            "is_npc": is_npc
        }
        if is_npc and npc_difficulty:
            player_info_data["npc_difficulty"] = npc_difficulty

        new_player = PlayerInfo(**player_info_data)
        self.gs.players.append(new_player)
        self.gs.last_error_message = None

        if len(self.gs.players) == 1:  # First player added
            if self.gs.mode == GameMode.ONLINE:
                self.gs.status = GameStatus.WAITING_FOR_OPPONENT
                self.gs.current_player_id = player_id
                self.gs.waiting_since = datetime.now(UTC)
            # For local mode with P1, we wait for P2 (human or NPC) to be added
            # No status change yet, current_player_id not set until P2 joins

        elif len(self.gs.players) == 2:  # Both players now present
            self.gs.status = GameStatus.IN_PROGRESS
            player_x_info = next(p for p in self.gs.players if p.symbol == PlayerSymbol.X)
            self.gs.current_player_id = player_x_info.id  # X always starts
            self.gs.next_forced_global_board = None
            self.gs.waiting_since = None

            # If the second player added is an NPC and it's their turn (e.g. P1 is human, P2 is NPC, P1 made a move)
            # This specific logic is more for when make_move hands over to an NPC.
            # Here, we just set up the game. X (P1) will make the first move.

        self.gs.updated_at = datetime.now(UTC)
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

        self.gs.updated_at = datetime.now(UTC)
        self.gs.last_made_move_coords = (move.global_row, move.global_col, move.local_row, move.local_col)

        return True

    def handle_player_disconnect(self, player_id: str):
        player = self.gs.get_player_info(player_id)
        app = get_app(GAME_NAME)  # Hol dir die App-Instanz
        if player:
            if not player.is_connected:  # Already marked as disconnected
                app.logger.info(f"Player {player_id} was already marked as disconnected from game {self.gs.game_id}.")
                return

            player.is_connected = False
            self.gs.updated_at = datetime.now(UTC)
            app.logger.info(f"Player {player_id} disconnected from game {self.gs.game_id}. Name: {player.name}")

            if self.gs.mode == GameMode.ONLINE:
                if self.gs.status == GameStatus.IN_PROGRESS:
                    opponent = self.gs.get_opponent_info(player_id)
                    if opponent and opponent.is_connected:
                        self.gs.status = GameStatus.ABORTED  # Use ABORTED as "paused"
                        self.gs.player_who_paused = player_id  # Store who disconnected
                        # This message is for the game state, will be seen by the other player via SSE
                        self.gs.last_error_message = f"Player {player.name} disconnected. Waiting for them to rejoin."
                        app.logger.info(
                            f"Game {self.gs.game_id} PAUSED, waiting for {player.name} ({player_id}) to reconnect.")
                    else:
                        # Opponent also disconnected or was already gone
                        self.gs.status = GameStatus.ABORTED
                        self.gs.last_error_message = "Both players disconnected. Game aborted."
                        self.gs.player_who_paused = None  # No specific player to wait for
                        app.logger.info(
                            f"Game {self.gs.game_id} ABORTED, both players (or last active player) disconnected.")
                elif self.gs.status == GameStatus.WAITING_FOR_OPPONENT:
                    # If the creator (P1) disconnects while waiting for P2
                    if len(self.gs.players) == 1 and self.gs.players[0].id == player_id:
                        self.gs.status = GameStatus.ABORTED
                        self.gs.last_error_message = "Game creator disconnected before opponent joined. Game aborted."
                        self.gs.player_who_paused = None
                        app.logger.info(
                            f"Game {self.gs.game_id} ABORTED, creator {player.name} ({player_id}) disconnected while WAITING_FOR_OPPONENT.")
                elif self.gs.status == GameStatus.ABORTED and self.gs.player_who_paused:
                    # Game was already paused (e.g. P1 disconnected), and now P2 (the waiting one) disconnects
                    if self.gs.player_who_paused != player_id:  # Ensure it's the other player
                        self.gs.last_error_message = "Other player also disconnected during pause. Game aborted."
                        self.gs.player_who_paused = None  # No one specific to wait for now
                        app.logger.info(
                            f"Game {self.gs.game_id} ABORTED, waiting player {player.name} ({player_id}) disconnected.")

    def handle_player_reconnect(self, player_id: str) -> bool:
        player = self.gs.get_player_info(player_id)
        app = get_app(GAME_NAME)
        if not player:
            app.logger.warning(f"Reconnect attempt for unknown player {player_id} in game {self.gs.game_id}.")
            return False

        if player.is_connected:
            app.logger.info(
                f"Player {player.name} ({player_id}) attempted reconnect but was already marked as connected to game {self.gs.game_id}.")
            if self.gs.status == GameStatus.ABORTED and self.gs.player_who_paused == player_id:
                opponent = self.gs.get_opponent_info(player_id)
                if opponent and opponent.is_connected:
                    self.gs.status = GameStatus.IN_PROGRESS
                    self.gs.last_error_message = f"Connection for {player.name} re-established. Game resumed."
                    self.gs.player_who_paused = None
                    self.gs.updated_at = datetime.now(UTC)
                    app.logger.info(
                        f"Game {self.gs.game_id} resumed as already-connected pauser {player.name} re-interacted.")
                else:
                    self.gs.last_error_message = f"Welcome back, {player.name}! Your opponent is still not connected."
            return True

        player.is_connected = True
        self.gs.updated_at = datetime.now(UTC)
        app.logger.info(
            f"Player {player.name} ({player_id}) reconnected to game {self.gs.game_id}. Previous status: {self.gs.status}, Paused by: {self.gs.player_who_paused}")

        if self.gs.status == GameStatus.ABORTED:
            if self.gs.player_who_paused == player_id:  # The player who caused the pause has reconnected
                opponent = self.gs.get_opponent_info(player_id)
                if opponent and opponent.is_connected:
                    self.gs.status = GameStatus.IN_PROGRESS
                    self.gs.last_error_message = f"Player {player.name} reconnected. Game resumed!"
                    self.gs.player_who_paused = None
                    app.logger.info(
                        f"Game {self.gs.game_id} RESUMED. Pauser {player.name} reconnected, opponent {opponent.name} is present.")
                else:  # Pauser reconnected, opponent (still) gone or never joined (if P1 disconnected from WAITING)
                    if not opponent and len(
                        self.gs.players) == 1:  # P1 reconnected to a game they created but no P2 yet
                        self.gs.status = GameStatus.WAITING_FOR_OPPONENT
                        self.gs.player_who_paused = None
                        self.gs.current_player_id = player_id
                        self.gs.last_error_message = f"Creator {player.name} reconnected. Waiting for opponent."
                        self.gs.waiting_since = datetime.now(UTC)  # Reset waiting timer
                    elif opponent:  # Opponent was there but is now disconnected
                        self.gs.player_who_paused = opponent.id  # Now waiting for the other person
                        self.gs.last_error_message = f"Welcome back, {player.name}! Your opponent ({opponent.name}) is not connected. Game remains paused."
                        app.logger.info(
                            f"Game {self.gs.game_id} still PAUSED. {player.name} reconnected, but opponent {opponent.name} is NOT. Waiting for {opponent.name}.")
                    else:  # Should be rare: 2 players in list, but opponent object not found for P1
                        self.gs.last_error_message = f"Welcome back, {player.name}! Opponent details unclear. Game remains paused."


            elif self.gs.player_who_paused and self.gs.player_who_paused != player_id:
                # The *other* player reconnected, while game was paused for initial pauser.
                initial_pauser_info = self.gs.get_player_info(self.gs.player_who_paused)
                if initial_pauser_info and initial_pauser_info.is_connected:  # This implies both are now connected.
                    self.gs.status = GameStatus.IN_PROGRESS
                    self.gs.last_error_message = "Both players are now connected. Game resumed!"
                    self.gs.player_who_paused = None
                    app.logger.info(
                        f"Game {self.gs.game_id} RESUMED. Waiting player {player.name} reconnected, initial pauser {initial_pauser_info.name} also present.")
                else:
                    self.gs.last_error_message = f"Welcome back, {player.name}! Still waiting for {initial_pauser_info.name if initial_pauser_info else 'the other player'} to reconnect."
                    app.logger.info(
                        f"Game {self.gs.game_id} still PAUSED. Player {player.name} reconnected, but still waiting for original pauser {self.gs.player_who_paused}.")

            else:  # game is ABORTED but no specific player_who_paused (hard abort by timeout or both disconnected)
                if len(self.gs.players) == 2:  # Was a two-player game
                    opponent = self.gs.get_opponent_info(player_id)
                    if opponent:
                        # Revive the game to a paused state, waiting for the other player
                        self.gs.player_who_paused = opponent.id
                        self.gs.status = GameStatus.ABORTED  # Remains aborted, but now specifically for opponent
                        self.gs.last_error_message = f"Welcome back, {player.name}! Game was fully aborted. Now waiting for {opponent.name} to rejoin."
                        app.logger.info(
                            f"Game {self.gs.game_id} REVIVED from HARD ABORT by {player.name}. Now paused, waiting for {opponent.name} ({opponent.id}).")
                    else:  # Should not happen if two players were in game and player_id is one of them
                        self.gs.last_error_message = f"Player {player.name} reconnected, but game state is inconsistent (opponent not found)."
                        app.logger.warning(
                            f"Game {self.gs.game_id} HARD ABORT revival by {player.name} failed, opponent info missing.")
                elif len(self.gs.players) == 1 and self.gs.players[0].id == player_id:
                    # P1 created, P1 disconnected, game WAITING_FOR_OPPONENT timed out & hard aborted. P1 tries to rejoin.
                    self.gs.status = GameStatus.WAITING_FOR_OPPONENT
                    self.gs.player_who_paused = None
                    self.gs.current_player_id = player_id
                    self.gs.last_error_message = f"Creator {player.name} reconnected. Waiting for opponent."
                    self.gs.waiting_since = datetime.now(UTC)  # Reset waiting timer
                    app.logger.info(
                        f"Game {self.gs.game_id} (previously hard aborted while waiting) revived by creator {player.name}. Now WAITING_FOR_OPPONENT.")
                else:
                    self.gs.last_error_message = f"Player {player.name} reconnected, but the game was aborted and cannot be revived in its current player configuration."
                    app.logger.info(
                        f"Game {self.gs.game_id} HARD ABORTED. Player {player.name} reconnected, but game cannot resume in current configuration.")


        elif self.gs.status == GameStatus.IN_PROGRESS:
            opponent = self.gs.get_opponent_info(player_id)
            if not opponent or not opponent.is_connected:
                self.gs.status = GameStatus.ABORTED
                self.gs.player_who_paused = opponent.id if opponent else None
                self.gs.last_error_message = f"Welcome back, {player.name}! Your opponent disconnected while you were away. Waiting for them."
                app.logger.info(
                    f"Game {self.gs.game_id} transitions to PAUSED. {player.name} reconnected to IN_PROGRESS, but opponent {opponent.id if opponent else 'N/A'} is gone.")
            else:
                self.gs.last_error_message = f"Player {player.name} re-established connection during active game."
                app.logger.info(
                    f"Player {player.name} ({player_id}) re-established connection to IN_PROGRESS game {self.gs.game_id}.")

        elif self.gs.status == GameStatus.WAITING_FOR_OPPONENT:
            if len(self.gs.players) == 1 and self.gs.players[0].id == player_id:
                self.gs.last_error_message = f"Creator {player.name} reconnected. Still waiting for opponent."
                self.gs.current_player_id = player_id
                self.gs.waiting_since = datetime.now(UTC)  # Reset waiting timer
                app.logger.info(
                    f"Creator {player.name} ({player_id}) reconnected to WAITING_FOR_OPPONENT game {self.gs.game_id}.")
            else:
                app.logger.warning(
                    f"Non-creator {player.name} or unexpected player count for reconnect to WAITING_FOR_OPPONENT game {self.gs.game_id}.")

        return True


# -- NPC Agents ---
import random  # Add this import at the top of your file if not already there

# Assuming the following classes and enums are defined as in your provided code:
# from pydantic import BaseModel, Field, model_validator
# from enum import Enum
# from datetime import datetime, timezone
# import uuid
# class PlayerSymbol(str, Enum): ...
# class CellState(str, Enum): ...
# class BoardWinner(str, Enum): ...
# class NPCDifficulty(str, Enum): ...
# class GameStatus(str, Enum): ...
# class GameMode(str, Enum): ...
# class GameConfig(BaseModel): ...
# class PlayerInfo(BaseModel): ...
# class Move(BaseModel): ...
# class GameState(BaseModel): ...
# class UltimateTTTGameEngine: ...


def get_npc_move_easy(game_state: GameState, npc_player_info: PlayerInfo) -> Move | None:
    gs = game_state
    size = gs.config.grid_size
    npc_symbol = npc_player_info.symbol
    opponent_symbol = PlayerSymbol.O if npc_symbol == PlayerSymbol.X else PlayerSymbol.X

    engine = UltimateTTTGameEngine(gs)

    all_raw_possible_moves: list[dict[str, int]] = []

    is_forced_to_play_specific_board = False
    forced_gr, forced_gc = -1, -1

    if gs.next_forced_global_board:
        fgr, fgc = gs.next_forced_global_board
        if gs.global_board_winners[fgr][fgc] == BoardWinner.NONE and \
            not engine._is_local_board_full(gs.local_boards_state[fgr][fgc]):
            is_forced_to_play_specific_board = True
            forced_gr, forced_gc = fgr, fgc
            for lr in range(size):
                for lc in range(size):
                    if gs.local_boards_state[fgr][fgc][lr][lc] == CellState.EMPTY:
                        all_raw_possible_moves.append({'gr': fgr, 'gc': fgc, 'lr': lr, 'lc': lc})

    if not is_forced_to_play_specific_board:
        all_raw_possible_moves = []
        for gr_idx in range(size):
            for gc_idx in range(size):
                if gs.global_board_winners[gr_idx][gc_idx] == BoardWinner.NONE and \
                    not engine._is_local_board_full(gs.local_boards_state[gr_idx][gc_idx]):
                    for lr in range(size):
                        for lc in range(size):
                            if gs.local_boards_state[gr_idx][gc_idx][lr][lc] == CellState.EMPTY:
                                all_raw_possible_moves.append({'gr': gr_idx, 'gc': gc_idx, 'lr': lr, 'lc': lc})

    if not all_raw_possible_moves:
        gs.last_error_message = "NPC Error (Easy): No possible moves found."
        return None

    strategically_safer_moves: list[dict[str, int]] = []
    for move_dict in all_raw_possible_moves:
        opponent_target_gr, opponent_target_gc = move_dict['lr'], move_dict['lc']

        if gs.global_board_winners[opponent_target_gr][opponent_target_gc] != BoardWinner.NONE or \
            engine._is_local_board_full(gs.local_boards_state[opponent_target_gr][opponent_target_gc]):
            strategically_safer_moves.append(move_dict)
            continue

        opponent_can_win_target_board = False
        target_local_board_for_opponent = gs.local_boards_state[opponent_target_gr][opponent_target_gc]
        opponent_cell_state_val = CellState(opponent_symbol.value)
        for r_opp in range(size):
            for c_opp in range(size):
                if target_local_board_for_opponent[r_opp][c_opp] == CellState.EMPTY:
                    temp_board = [row[:] for row in target_local_board_for_opponent]
                    temp_board[r_opp][c_opp] = opponent_cell_state_val
                    if engine._get_board_winner_symbol(temp_board, CellState) == opponent_cell_state_val:
                        opponent_can_win_target_board = True
                        break
            if opponent_can_win_target_board:
                break

        if not opponent_can_win_target_board:
            strategically_safer_moves.append(move_dict)

    candidate_moves_for_decision = strategically_safer_moves if strategically_safer_moves else all_raw_possible_moves

    if not candidate_moves_for_decision:
        gs.last_error_message = "NPC Error (Easy): No candidate moves after filtering."
        # This case should be extremely rare if all_raw_possible_moves was not empty.
        # If it happens, it means all moves were deemed unsafe and strategically_safer_moves is empty,
        # so candidate_moves_for_decision becomes all_raw_possible_moves.
        # So, this specific error message path is unlikely unless all_raw_possible_moves was initially empty.
        if all_raw_possible_moves:  # Should always be true if we reach here unless logic error above.
            candidate_moves_for_decision = all_raw_possible_moves
        else:
            return None  # Truly no moves.

    chosen_move_dict: dict[str, int] | None = None

    if is_forced_to_play_specific_board:
        if not candidate_moves_for_decision:
            gs.last_error_message = "NPC Error (Easy): No candidate moves for forced board (logic error)."
            return None

        current_local_board_state = gs.local_boards_state[forced_gr][forced_gc]
        npc_cell_state_val = CellState(npc_symbol.value)
        opponent_cell_state_on_forced_board = CellState(opponent_symbol.value)

        for move_d in candidate_moves_for_decision:
            temp_board = [row[:] for row in current_local_board_state]
            temp_board[move_d['lr']][move_d['lc']] = npc_cell_state_val
            if engine._get_board_winner_symbol(temp_board, CellState) == npc_cell_state_val:
                chosen_move_dict = move_d
                break
        if chosen_move_dict:
            return Move(player_id=npc_player_info.id,
                        global_row=chosen_move_dict['gr'], global_col=chosen_move_dict['gc'],
                        local_row=chosen_move_dict['lr'], local_col=chosen_move_dict['lc'])

        for move_d in candidate_moves_for_decision:
            temp_board = [row[:] for row in current_local_board_state]
            temp_board[move_d['lr']][move_d['lc']] = opponent_cell_state_on_forced_board
            if engine._get_board_winner_symbol(temp_board, CellState) == opponent_cell_state_on_forced_board:
                chosen_move_dict = move_d
                break
        if chosen_move_dict:
            return Move(player_id=npc_player_info.id,
                        global_row=chosen_move_dict['gr'], global_col=chosen_move_dict['gc'],
                        local_row=chosen_move_dict['lr'], local_col=chosen_move_dict['lc'])

        if candidate_moves_for_decision:
            chosen_move_dict = random.choice(candidate_moves_for_decision)
            # Fallthrough to return statement at the end

    else:  # Play anywhere
        winning_moves_local_board = []
        blocking_moves_local_board = []
        send_opponent_to_finished_board_moves = []
        other_neutral_moves = []

        npc_cell_state_val = CellState(npc_symbol.value)
        opponent_cell_state_for_play_anywhere = CellState(opponent_symbol.value)

        for move_d in candidate_moves_for_decision:
            gr, gc, lr, lc = move_d['gr'], move_d['gc'], move_d['lr'], move_d['lc']
            board_to_play_on = gs.local_boards_state[gr][gc]

            temp_board_win = [row[:] for row in board_to_play_on]
            temp_board_win[lr][lc] = npc_cell_state_val
            if engine._get_board_winner_symbol(temp_board_win, CellState) == npc_cell_state_val:
                winning_moves_local_board.append(move_d)
                continue

            temp_board_block = [row[:] for row in board_to_play_on]
            temp_board_block[lr][lc] = opponent_cell_state_for_play_anywhere
            # Check if placing opponent's symbol in this cell would make them win (so NPC should block it by taking it)
            # This logic for blocking_moves_local_board is subtle: the NPC wants to *take* this cell
            # because if the *opponent* took it on *their* turn, they would win.
            # So, simulating the opponent's symbol is correct to identify the cell, but the NPC makes the move.
            if engine._get_board_winner_symbol(temp_board_block, CellState) == opponent_cell_state_for_play_anywhere:
                blocking_moves_local_board.append(move_d)
                continue

            opponent_target_gr_pa, opponent_target_gc_pa = lr, lc
            if gs.global_board_winners[opponent_target_gr_pa][opponent_target_gc_pa] != BoardWinner.NONE or \
                engine._is_local_board_full(gs.local_boards_state[opponent_target_gr_pa][opponent_target_gc_pa]):
                send_opponent_to_finished_board_moves.append(move_d)
                continue

            other_neutral_moves.append(move_d)

        if winning_moves_local_board:
            chosen_move_dict = random.choice(winning_moves_local_board)
        elif blocking_moves_local_board:
            chosen_move_dict = random.choice(blocking_moves_local_board)
        elif send_opponent_to_finished_board_moves:
            chosen_move_dict = random.choice(send_opponent_to_finished_board_moves)
        elif other_neutral_moves:  # This list must contain elements if candidate_moves_for_decision had elements
            chosen_move_dict = random.choice(other_neutral_moves)
        elif candidate_moves_for_decision:  # Fallback if all moves fell into categories that were empty
            chosen_move_dict = random.choice(candidate_moves_for_decision)

    if chosen_move_dict:
        return Move(player_id=npc_player_info.id,
                    global_row=chosen_move_dict['gr'], global_col=chosen_move_dict['gc'],
                    local_row=chosen_move_dict['lr'], local_col=chosen_move_dict['lc'])

    # This fallback should ideally not be reached if candidate_moves_for_decision always has something.
    # But as a final safeguard, if chosen_move_dict is still None, pick from all_raw_possible_moves
    # This implies some logic flaw above or an edge case where no categories matched.
    if all_raw_possible_moves:
        gs.last_error_message = "NPC Error (Easy): Critical Fallback to totally random move from raw list."
        safety_choice = random.choice(all_raw_possible_moves)
        return Move(player_id=npc_player_info.id,
                    global_row=safety_choice['gr'], global_col=safety_choice['gc'],
                    local_row=safety_choice['lr'], local_col=safety_choice['lc'])

    return None



# Assume your existing imports for GameState, PlayerInfo, Move, Enums, etc.
# from ... import GameState, PlayerInfo, Move, PlayerSymbol, CellState, BoardWinner, UltimateTTTGameEngine
# --- Helper functions for Zwickmühle detection (for 3x3 primarily) ---

def _check_line_for_potential_win(line: list[CellState], player_symbol_cell: CellState, size: int) -> bool:
    """Checks if a line has (size-1) of player's symbols and one empty cell."""
    if size <= 1: return False  # Not meaningful for 1x1 or smaller
    player_pieces = line.count(player_symbol_cell)
    empty_cells = line.count(CellState.EMPTY)
    return player_pieces == size - 1 and empty_cells == 1


def _count_potential_wins_on_board(board: list[list[CellState]], player_symbol_cell: CellState, size: int) -> int:
    """Counts how many lines on the board are potential wins for the player."""
    if not board or len(board) != size or not board[0] or len(board[0]) != size:
        # Invalid board structure for counting
        return 0

    potential_wins = 0
    # Rows
    for r in range(size):
        if _check_line_for_potential_win(board[r], player_symbol_cell, size):
            potential_wins += 1
    # Columns
    for c in range(size):
        col_line = [board[r][c] for r in range(size)]
        if _check_line_for_potential_win(col_line, player_symbol_cell, size):
            potential_wins += 1
    # Diagonal 1 (top-left to bottom-right)
    diag1_line = [board[i][i] for i in range(size)]
    if _check_line_for_potential_win(diag1_line, player_symbol_cell, size):
        potential_wins += 1
    # Diagonal 2 (top-right to bottom-left)
    diag2_line = [board[i][size - 1 - i] for i in range(size)]
    if _check_line_for_potential_win(diag2_line, player_symbol_cell, size):
        potential_wins += 1
    return potential_wins


# --- End Helper functions ---
# --- Helper functions from previous (ensure they are available) ---
# _check_line_for_potential_win(line: List[CellState], player_symbol_cell: CellState, size: int) -> bool
# _count_potential_wins_on_board(board: List[List[CellState]], player_symbol_cell: CellState, size: int) -> int

# --- New/Refined Helper functions for scoring/evaluation ---

def _get_npc_move_outcome_on_board(
    local_board_state: list[list[CellState]],
    move_lr: int, move_lc: int,
    npc_symbol_cell: CellState,
    opponent_symbol_cell: CellState,
    engine: UltimateTTTGameEngine,
    zwick_eval_size: int
) -> dict[str, bool | int]:
    """
    Evaluates what the NPC achieves by playing at (move_lr, move_lc) on local_board_state.
    Returns: {'wins': bool, 'creates_zwick': bool, 'blocks_opponent_win': bool, 'sets_up_line': bool}
    """
    outcome = {'wins': False, 'creates_zwick': False, 'blocks_opponent_win': False, 'sets_up_line': False}

    # Simulate NPC's move
    temp_board_npc_plays = [row[:] for row in local_board_state]
    temp_board_npc_plays[move_lr][move_lc] = npc_symbol_cell

    if engine._get_board_winner_symbol(temp_board_npc_plays, CellState) == npc_symbol_cell:
        outcome['wins'] = True

    # Zwickmühle implies setting up lines, so check Zwick first.
    num_potential_wins_after_npc_move = _count_potential_wins_on_board(temp_board_npc_plays, npc_symbol_cell,
                                                                       zwick_eval_size)
    if num_potential_wins_after_npc_move >= 2:
        outcome['creates_zwick'] = True
    elif num_potential_wins_after_npc_move == 1:  # and not already winning/zwicking
        outcome['sets_up_line'] = True

    # Check if this move blocks an opponent's immediate win
    # This means if the opponent played at (move_lr, move_lc), they would have won.
    temp_board_opponent_hypothetical_move = [row[:] for row in local_board_state]  # Fresh board
    temp_board_opponent_hypothetical_move[move_lr][move_lc] = opponent_symbol_cell
    if engine._get_board_winner_symbol(temp_board_opponent_hypothetical_move, CellState) == opponent_symbol_cell:
        outcome['blocks_opponent_win'] = True

    return outcome


def _get_opponent_threat_on_sent_board(
    target_local_board_state: list[list[CellState]],  # Board where opponent will be sent
    opponent_symbol_cell: CellState,
    engine: UltimateTTTGameEngine,
    zwick_eval_size: int,
    game_board_size: int
) -> dict[str, bool]:
    """
    Evaluates if the opponent, IF SENT TO target_local_board_state,
    can immediately win or create a Zwickmühle there.
    Returns: {'can_win': bool, 'can_zwick': bool}
    """
    threats = {'can_win': False, 'can_zwick': False}

    # Iterate through all empty cells on the target board for the opponent
    for r_opp in range(game_board_size):
        for c_opp in range(game_board_size):
            if target_local_board_state[r_opp][c_opp] == CellState.EMPTY:
                # Simulate opponent playing in this empty cell
                temp_board_opp_plays = [row[:] for row in target_local_board_state]
                temp_board_opp_plays[r_opp][c_opp] = opponent_symbol_cell

                if engine._get_board_winner_symbol(temp_board_opp_plays, CellState) == opponent_symbol_cell:
                    threats['can_win'] = True
                    return threats  # Max threat, no need to check further

                if not threats['can_zwick']:  # Only check Zwick if not already found for this board
                    if _count_potential_wins_on_board(temp_board_opp_plays, opponent_symbol_cell, zwick_eval_size) >= 2:
                        threats['can_zwick'] = True
                        # Continue checking other opponent moves; one might be an immediate win.
    return threats


# --- Main Medium NPC function with scoring ---

def get_npc_move_medium(game_state: GameState, npc_player_info: PlayerInfo) -> Move | None:
    gs = game_state
    game_board_size = gs.config.grid_size
    zwick_eval_size = 3  # Zwickmühle logic is primarily for 3x3 boards

    npc_symbol = npc_player_info.symbol
    npc_symbol_cellstate = CellState(npc_symbol.value)
    opponent_symbol = PlayerSymbol.O if npc_symbol == PlayerSymbol.X else PlayerSymbol.X
    opponent_symbol_cellstate = CellState(opponent_symbol.value)

    engine = UltimateTTTGameEngine(gs)

    # 1. Determine all raw possible moves
    all_raw_possible_moves: list[dict[str, int]] = []
    # This flag indicates if the NPC *must* play on a specific board vs. having free choice.
    # Free choice arises if gs.next_forced_global_board is None, OR if it points to a completed board.
    npc_has_free_choice = True
    forced_board_coords_for_npc: tuple[int, int] | None = None

    if gs.next_forced_global_board:
        fgr, fgc = gs.next_forced_global_board
        if gs.global_board_winners[fgr][fgc] == BoardWinner.NONE and \
            not engine._is_local_board_full(gs.local_boards_state[fgr][fgc]):
            npc_has_free_choice = False  # NPC is forced to this specific, active board
            forced_board_coords_for_npc = (fgr, fgc)
            # Populate moves only for this board
            for lr in range(game_board_size):
                for lc in range(game_board_size):
                    if gs.local_boards_state[fgr][fgc][lr][lc] == CellState.EMPTY:
                        all_raw_possible_moves.append({'gr': fgr, 'gc': fgc, 'lr': lr, 'lc': lc})

    if npc_has_free_choice:  # NPC can play anywhere (either initially or got a free move)
        all_raw_possible_moves = []  # Ensure it's empty before populating
        for gr_idx in range(game_board_size):
            for gc_idx in range(game_board_size):
                # NPC can only play on global boards that are NOT YET WON/DRAWN
                if gs.global_board_winners[gr_idx][gc_idx] == BoardWinner.NONE and \
                    not engine._is_local_board_full(gs.local_boards_state[gr_idx][gc_idx]):
                    for lr_idx in range(game_board_size):
                        for lc_idx in range(game_board_size):
                            if gs.local_boards_state[gr_idx][gc_idx][lr_idx][lc_idx] == CellState.EMPTY:
                                all_raw_possible_moves.append({'gr': gr_idx, 'gc': gc_idx, 'lr': lr_idx, 'lc': lc_idx})

    if not all_raw_possible_moves:
        gs.last_error_message = "NPC Error (Medium): No possible moves found (engine state might be stuck)."
        return None

    # 2. Score each possible move
    scored_moves: list[tuple[dict[str, int], float]] = []

    for move_dict in all_raw_possible_moves:
        npc_play_gr, npc_play_gc = move_dict['gr'], move_dict['gc']
        npc_play_lr, npc_play_lc = move_dict['lr'], move_dict['lc']

        # --- A. Evaluate NPC's gain on the board it's playing (npc_play_gr, npc_play_gc) ---
        current_board_for_npc_to_play_on = gs.local_boards_state[npc_play_gr][npc_play_gc]
        npc_gain_details = _get_npc_move_outcome_on_board(
            current_board_for_npc_to_play_on, npc_play_lr, npc_play_lc,
            npc_symbol_cellstate, opponent_symbol_cellstate,
            engine, zwick_eval_size
        )

        npc_gain_score = 0
        if npc_gain_details['wins']:
            npc_gain_score += 10000  # Highest priority: win the local board
        elif npc_gain_details['creates_zwick']:
            npc_gain_score += 5000  # Next: create a Zwickmühle
        elif npc_gain_details['blocks_opponent_win']:
            npc_gain_score += 2000  # Then: block opponent's win on this board
        elif npc_gain_details['sets_up_line']:
            npc_gain_score += 500  # Lower: set up a single line

        # Positional bonus for 3x3 (playing on the local board)
        if game_board_size == 3 and gs.global_board_winners[npc_play_gr][npc_play_gc] == BoardWinner.NONE:
            if (npc_play_lr, npc_play_lc) == (1, 1):
                npc_gain_score += 20  # Center
            elif (npc_play_lr, npc_play_lc) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                npc_gain_score += 10  # Corner

        # --- B. Evaluate consequence for opponent on the board they are SENT TO ---
        sends_opponent_to_gr, sends_opponent_to_gc = npc_play_lr, npc_play_lc
        opponent_target_board_state = gs.local_boards_state[sends_opponent_to_gr][sends_opponent_to_gc]

        opponent_consequence_penalty = 0
        gives_opponent_free_move_on_next_turn = False

        if gs.global_board_winners[sends_opponent_to_gr][sends_opponent_to_gc] != BoardWinner.NONE or \
            engine._is_local_board_full(opponent_target_board_state):
            gives_opponent_free_move_on_next_turn = True
            # If NPC is not achieving a win/Zwick with its own move, heavily penalize giving a free move.
            if npc_gain_score < 5000:
                opponent_consequence_penalty += 4000  # High penalty
            else:  # NPC wins/Zwicks - giving free move might be an acceptable trade-off
                opponent_consequence_penalty += 200  # Lower penalty
        else:
            # Board opponent is sent to is active, check for immediate threats
            opponent_threats_on_sent_board = _get_opponent_threat_on_sent_board(
                opponent_target_board_state, opponent_symbol_cellstate,
                engine, zwick_eval_size, game_board_size
            )
            if opponent_threats_on_sent_board['can_win']:
                opponent_consequence_penalty += 20000  # Extremely high penalty
            elif opponent_threats_on_sent_board['can_zwick']:
                opponent_consequence_penalty += 8000  # Very high penalty

        total_score = npc_gain_score - opponent_consequence_penalty
        scored_moves.append((move_dict, total_score))

    if not scored_moves:  # Should only happen if all_raw_possible_moves was empty
        gs.last_error_message = "NPC Error (Medium): No moves scored (implies no raw moves)."
        return None

    # Sort by score descending
    scored_moves.sort(key=lambda item: item[1], reverse=True)

    # For debugging, you might want to print the top few scored moves:
    # print(f"NPC {npc_player_info.name} Top Scored Moves:")
    # for i, (m, s) in enumerate(scored_moves[:5]):
    #     print(f"  {i+1}. Move: {m}, Score: {s}")

    top_score = scored_moves[0][1]
    best_options = [m_dict for m_dict, score in scored_moves if score == top_score]

    chosen_move_dict = random.choice(best_options)

    return Move(player_id=npc_player_info.id,
                global_row=chosen_move_dict['gr'], global_col=chosen_move_dict['gc'],
                local_row=chosen_move_dict['lr'], local_col=chosen_move_dict['lc'])


# --- Helper functions from previous (ensure they are available) ---
# _check_line_for_potential_win(line: List[CellState], player_symbol_cell: CellState, size: int) -> bool
# _count_potential_wins_on_board(board: List[List[CellState]], player_symbol_cell: CellState, size: int) -> int
# _get_npc_move_outcome_on_board(...) -> Dict[str, Union[bool, int]]
# _get_opponent_threat_on_sent_board(...) -> Dict[str, bool]

# --- New/Refined Helper functions for HARD NPC ---

def _simulate_and_evaluate_next_global_state(
    current_gs: GameState,
    npc_move_gr: int, npc_move_gc: int, npc_move_lr: int, npc_move_lc: int,
    npc_player_info: PlayerInfo,
    opponent_player_info: PlayerInfo,  # Need opponent info for simulation
    engine_class: type,  # Pass the UltimateTTTGameEngine class
    zwick_eval_size: int,
    game_board_size: int
) -> tuple[
    GameState | None, BoardWinner | None, bool]:  # (new_gs, local_winner_of_npc_move, game_over_after_npc_move)
    """
    Simulates the NPC making a move and returns the new game state.
    This is a shallow simulation, doesn't play out opponent's turn.
    Returns the game state *after* the NPC's move is made and local/global winners updated.
    """
    # Create a deep copy of the game state to simulate on
    sim_gs = current_gs.model_copy(deep=True)
    sim_engine = engine_class(sim_gs)

    move_to_make = Move(
        player_id=npc_player_info.id,
        global_row=npc_move_gr, global_col=npc_move_gc,
        local_row=npc_move_lr, local_col=npc_move_lc
    )

    if not sim_engine.make_move(move_to_make):
        # This should not happen if the move was valid from all_raw_possible_moves
        return None, None, True  # Indicate error / game over

    local_winner_of_this_move = sim_gs.global_board_winners[npc_move_gr][npc_move_gc]
    game_over = sim_gs.status == GameStatus.FINISHED

    return sim_gs, local_winner_of_this_move, game_over


def _count_global_potential_wins(global_board_winners: list[list[BoardWinner]], npc_board_winner_symbol: BoardWinner,
                                 size: int) -> int:
    """Counts potential global winning lines for the NPC."""
    potential_wins = 0
    # Rows
    for r in range(size):
        line = [global_board_winners[r][c] for c in range(size)]
        if line.count(npc_board_winner_symbol) == size - 1 and line.count(BoardWinner.NONE) == 1:
            potential_wins += 1
    # Columns
    for c in range(size):
        line = [global_board_winners[r][c] for r in range(size)]
        if line.count(npc_board_winner_symbol) == size - 1 and line.count(BoardWinner.NONE) == 1:
            potential_wins += 1
    # Diagonals
    diag1 = [global_board_winners[i][i] for i in range(size)]
    if diag1.count(npc_board_winner_symbol) == size - 1 and diag1.count(BoardWinner.NONE) == 1:
        potential_wins += 1
    diag2 = [global_board_winners[i][size - 1 - i] for i in range(size)]
    if diag2.count(npc_board_winner_symbol) == size - 1 and diag2.count(BoardWinner.NONE) == 1:
        potential_wins += 1
    return potential_wins


# --- Minimax with Alpha-Beta Pruning ---
import math  # For infinity

# We'll reuse the simulation and scoring logic from get_npc_move_hard
# as the heuristic evaluation function. Let's rename/adapt it slightly.

def heuristic_evaluate_game_state(
    current_gs: GameState,
    npc_player_info: PlayerInfo,
    opponent_player_info: PlayerInfo,
    engine_class: type,  # type of UltimateTTTGameEngine
    zwick_eval_size: int,  # Typically 3 for Zwickmühle logic
    game_board_size: int
) -> float:
    # Get integer mappings for players and the enum_to_int conversion map
    npc_int, opponent_int, enum_to_int_map = _get_int_symbol_maps(npc_player_info, opponent_player_info)

    # Check for terminal state first (no NumPy needed here, uses GameState status)
    if current_gs.status == GameStatus.FINISHED:
        if current_gs.overall_winner_symbol == npc_player_info.symbol:  # Compare PlayerSymbol directly
            return 1000000.0
        elif current_gs.overall_winner_symbol == opponent_player_info.symbol:  # Compare PlayerSymbol directly
            return -1000000.0
        else:  # Draw
            return 0.0

    score = 0.0

    # 1. Convert global_board_winners to NumPy array
    # Ensure all enum values are handled by enum_to_int_map; otherwise, map to a default like -1
    # or raise an error if an unknown enum is encountered.
    try:
        global_winners_np = np.array(
            [[enum_to_int_map[winner] for winner in row] for row in current_gs.global_board_winners],
            dtype=int
        )
    except KeyError as e:
        print(f"KeyError during global_winners_np conversion: {e}. Enum value not in map.")
        # Handle error appropriately, e.g., return a neutral score or raise
        return 0.0  # Or some other error indicator

    num_global_potential_wins_npc = _count_global_potential_wins_np(global_winners_np, npc_int, game_board_size)
    num_global_potential_wins_opp = _count_global_potential_wins_np(global_winners_np, opponent_int, game_board_size)

    if num_global_potential_wins_npc >= 2:
        score += 200000
    elif num_global_potential_wins_npc == 1:
        score += 75000
    if num_global_potential_wins_opp >= 2:
        score -= 180000
    elif num_global_potential_wins_opp == 1:
        score -= 70000

    npc_global_boards_won_count = np.sum(global_winners_np == npc_int)
    opp_global_boards_won_count = np.sum(global_winners_np == opponent_int)
    score += npc_global_boards_won_count * 50000
    score -= opp_global_boards_won_count * 45000

    # 2. Local board advantages
    current_engine_instance = engine_class(current_gs)  # For _is_local_board_full check

    for r_glob in range(game_board_size):
        for c_glob in range(game_board_size):
            if current_gs.global_board_winners[r_glob][c_glob] == BoardWinner.NONE and \
                not current_engine_instance._is_local_board_full(current_gs.local_boards_state[r_glob][c_glob]):

                local_board_list = current_gs.local_boards_state[r_glob][c_glob]
                try:
                    local_board_np = np.array(
                        [[enum_to_int_map[cell] for cell in row] for row in local_board_list],
                        dtype=int
                    )
                except KeyError as e:
                    print(
                        f"KeyError during local_board_np conversion (board {r_glob},{c_glob}): {e}. Enum value not in map.")
                    continue  # Skip this board or handle error

                # Use zwick_eval_size for these local board evaluations
                npc_local_pot_wins = _count_potential_wins_on_board_np(local_board_np, npc_int, zwick_eval_size)
                opp_local_pot_wins = _count_potential_wins_on_board_np(local_board_np, opponent_int, zwick_eval_size)

                if npc_local_pot_wins >= 2:
                    score += 1000
                elif npc_local_pot_wins == 1:
                    score += 300

                if opp_local_pot_wins >= 2:
                    score -= 900
                elif opp_local_pot_wins == 1:
                    score -= 270

                if zwick_eval_size == 3:  # Positional bonus only for 3x3 interpretation
                    if local_board_np[1, 1] == npc_int:
                        score += 20
                    elif local_board_np[1, 1] == opponent_int:
                        score -= 18

                    # Using direct indexing for corners for 3x3
                    corners_coords = [(0, 0), (0, 2), (2, 0), (2, 2)]
                    for r_corn, c_corn in corners_coords:
                        if local_board_np[r_corn, c_corn] == npc_int:
                            score += 10
                        elif local_board_np[r_corn, c_corn] == opponent_int:
                            score -= 9
    return score




def minimax(
    current_gs: GameState,
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing_player: bool,
    npc_player_info: PlayerInfo,
    opponent_player_info: PlayerInfo,
    engine_class: type,
    zwick_eval_size: int,
    game_board_size: int,
    initial_call: bool = False  # Default to False for recursive calls
) -> float | tuple[float, dict[str, int] | None]:  # Return type depends on initial_call

    if depth == 0 or current_gs.status == GameStatus.FINISHED:
        eval_score = heuristic_evaluate_game_state(
            current_gs, npc_player_info, opponent_player_info,
            engine_class, zwick_eval_size, game_board_size
        )
        # For the initial call, if it hits depth 0 or terminal state immediately,
        # it means no moves were possible from the root, or depth was 0.
        # It should still return a move (None in this case) if initial_call is True.
        if initial_call:
            return eval_score, None
        return eval_score  # For recursive calls, just return the score

    active_player_id_in_gs = current_gs.current_player_id
    if not active_player_id_in_gs:
        # Error case or unexpected state, return worst possible score for current player
        worst_score = -math.inf if is_maximizing_player else math.inf
        if initial_call: return worst_score, None
        return worst_score

    possible_moves_dicts: list[dict[str, int]] = []
    current_engine_instance = engine_class(current_gs)
    forced_board = current_gs.next_forced_global_board
    can_play_anywhere_active_player = True

    if forced_board:
        fgr, fgc = forced_board
        if current_gs.global_board_winners[fgr][fgc] == BoardWinner.NONE and \
            not current_engine_instance._is_local_board_full(current_gs.local_boards_state[fgr][fgc]):
            can_play_anywhere_active_player = False
            for lr in range(game_board_size):
                for lc in range(game_board_size):
                    if current_gs.local_boards_state[fgr][fgc][lr][lc] == CellState.EMPTY:
                        possible_moves_dicts.append({'gr': fgr, 'gc': fgc, 'lr': lr, 'lc': lc})

    if can_play_anywhere_active_player:
        possible_moves_dicts = []
        for gr_idx in range(game_board_size):
            for gc_idx in range(game_board_size):
                if current_gs.global_board_winners[gr_idx][gc_idx] == BoardWinner.NONE and \
                    not current_engine_instance._is_local_board_full(current_gs.local_boards_state[gr_idx][gc_idx]):
                    for lr_idx in range(game_board_size):
                        for lc_idx in range(game_board_size):
                            if current_gs.local_boards_state[gr_idx][gc_idx][lr_idx][lc_idx] == CellState.EMPTY:
                                possible_moves_dicts.append({'gr': gr_idx, 'gc': gc_idx, 'lr': lr_idx, 'lc': lc_idx})

    if not possible_moves_dicts:
        eval_score = heuristic_evaluate_game_state(
            current_gs, npc_player_info, opponent_player_info,
            engine_class, zwick_eval_size, game_board_size
        )
        if initial_call: return eval_score, None
        return eval_score

    best_move_for_this_level: dict[str, int] | None = None  # Only used if initial_call is True

    # Determine which player object to use for simulating the move
    # based on who is the active_player_id_in_gs for the *current recursion level*
    player_info_for_sim_move: PlayerInfo
    other_player_info_for_sim: PlayerInfo

    if active_player_id_in_gs == npc_player_info.id:
        player_info_for_sim_move = npc_player_info
        other_player_info_for_sim = opponent_player_info
    elif active_player_id_in_gs == opponent_player_info.id:
        player_info_for_sim_move = opponent_player_info
        other_player_info_for_sim = npc_player_info
    else:  # Should not happen
        worst_score = -math.inf if is_maximizing_player else math.inf
        if initial_call: return worst_score, None
        return worst_score

    if is_maximizing_player:  # Corresponds to the NPC's interest (Maximizer)
        max_eval = -math.inf
        random.shuffle(possible_moves_dicts)
        for move_dict in possible_moves_dicts:
            sim_gs_after_move, _, _ = _simulate_and_evaluate_next_global_state(
                current_gs, move_dict['gr'], move_dict['gc'], move_dict['lr'], move_dict['lc'],
                player_info_for_sim_move,  # Player whose turn it is in current_gs
                other_player_info_for_sim,  # The other player
                engine_class, zwick_eval_size, game_board_size
            )
            if sim_gs_after_move is None: continue

            # Recursive call: initial_call is False, is_maximizing_player flips
            eval_score_from_recursion = minimax(
                sim_gs_after_move, depth - 1, alpha, beta, False,  # Now Minimizer's turn
                npc_player_info, opponent_player_info, engine_class,
                zwick_eval_size, game_board_size, initial_call=False  # Explicitly False
            )
            # eval_score_from_recursion is guaranteed to be a float here

            if eval_score_from_recursion > max_eval:
                max_eval = eval_score_from_recursion
                if initial_call:  # Store the move that led to this max_eval
                    best_move_for_this_level = move_dict

            alpha = max(alpha, eval_score_from_recursion)
            if beta <= alpha:
                break

        if initial_call: return max_eval, best_move_for_this_level
        return max_eval

    else:  # Corresponds to the Opponent's interest (Minimizer)
        min_eval = math.inf
        random.shuffle(possible_moves_dicts)
        for move_dict in possible_moves_dicts:
            sim_gs_after_move, _, _ = _simulate_and_evaluate_next_global_state(
                current_gs, move_dict['gr'], move_dict['gc'], move_dict['lr'], move_dict['lc'],
                player_info_for_sim_move,  # Player whose turn it is in current_gs
                other_player_info_for_sim,  # The other player
                engine_class, zwick_eval_size, game_board_size
            )
            if sim_gs_after_move is None: continue

            # Recursive call: initial_call is False, is_maximizing_player flips
            eval_score_from_recursion = minimax(
                sim_gs_after_move, depth - 1, alpha, beta, True,  # Now Maximizer's turn
                npc_player_info, opponent_player_info, engine_class,
                zwick_eval_size, game_board_size, initial_call=False  # Explicitly False
            )
            # eval_score_from_recursion is guaranteed to be a float here

            if eval_score_from_recursion < min_eval:
                min_eval = eval_score_from_recursion
                if initial_call:  # This path is tricky for initial call, as NPC is maximizer.
                    # If by some chance initial_call was true and is_maximizing_player was false,
                    # it would mean we are asking for the opponent's best move.
                    best_move_for_this_level = move_dict

            beta = min(beta, eval_score_from_recursion)
            if beta <= alpha:
                break

        if initial_call: return min_eval, best_move_for_this_level  # Should not be the primary path for NPC move
        return min_eval


import numpy as np  # Add this at the top of your file

# --- Constants for NumPy representation ---
EMPTY_INT = 0
# We'll determine NPC_INT and OPPONENT_INT dynamically based on player symbols
# DRAW_INT for BoardWinner.DRAW
DRAW_INT = 3


# Helper to get integer representations for players and the enum-to-int map
def _get_int_symbol_maps(npc_player_info: PlayerInfo, opponent_player_info: PlayerInfo) -> tuple[
    int, int, dict[CellState | BoardWinner, int]]:
    """
    Determines integer mapping for NPC (1), Opponent (2) and creates a full enum-to-int map.
    """
    # Assign 1 to NPC, 2 to Opponent consistently.
    # The actual symbol (X or O) doesn't matter for the int value, only who is who.
    npc_int_val = 1
    opponent_int_val = 2

    enum_to_int_map = {
        CellState.EMPTY: EMPTY_INT,
        BoardWinner.NONE: EMPTY_INT,
        BoardWinner.DRAW: DRAW_INT,
    }
    # Map NPC's symbols
    enum_to_int_map[npc_player_info.symbol] = npc_int_val  # For overall_winner_symbol check
    enum_to_int_map[CellState(npc_player_info.symbol.value)] = npc_int_val
    enum_to_int_map[BoardWinner(npc_player_info.symbol.value)] = npc_int_val

    # Map Opponent's symbols
    enum_to_int_map[opponent_player_info.symbol] = opponent_int_val  # For overall_winner_symbol check
    enum_to_int_map[CellState(opponent_player_info.symbol.value)] = opponent_int_val
    enum_to_int_map[BoardWinner(opponent_player_info.symbol.value)] = opponent_int_val

    return npc_int_val, opponent_int_val, enum_to_int_map


def _count_lines_on_board_np(
    board_np: np.ndarray,  # 2D NumPy array
    player_int_symbol: int,
    size: int,
    check_type: str = "potential_win"  # "potential_win" or "actual_win"
) -> int:
    """
    Counts lines (potential or actual wins) on a NumPy board for the player.
    For "potential_win": player_int_symbol == size - 1 and EMPTY_INT == 1
    For "actual_win": player_int_symbol == size
    """
    if board_np.shape != (size, size):
        return 0

    count = 0

    lines_to_check = []
    # Rows
    for r in range(size): lines_to_check.append(board_np[r, :])
    # Columns
    for c in range(size): lines_to_check.append(board_np[:, c])
    # Diagonals
    lines_to_check.append(np.diag(board_np))
    lines_to_check.append(np.diag(np.fliplr(board_np)))  # Flipped left-right for other diagonal

    for line in lines_to_check:
        if check_type == "potential_win":
            if np.sum(line == player_int_symbol) == size - 1 and np.sum(line == EMPTY_INT) == 1:
                count += 1
        elif check_type == "actual_win":
            if np.all(line == player_int_symbol):  # If all elements in the line match player symbol
                # For actual_win, we usually just need to know if one exists, not count them.
                # But to fit the "count" structure, we'll count. If used for win check, >0 means win.
                count += 1
    return count


# Wrapper for clarity, though _count_lines_on_board_np can do both
def _count_potential_wins_on_board_np(board_np: np.ndarray, player_int_symbol: int, size: int) -> int:
    return _count_lines_on_board_np(board_np, player_int_symbol, size, "potential_win")


def _check_actual_win_on_board_np(board_np: np.ndarray, player_int_symbol: int, size: int) -> bool:
    return _count_lines_on_board_np(board_np, player_int_symbol, size, "actual_win") > 0


def _count_global_potential_wins_np(global_board_winners_np: np.ndarray, player_int_board_winner_symbol: int,
                                    size: int) -> int:
    return _count_potential_wins_on_board_np(global_board_winners_np, player_int_board_winner_symbol, size)

def get_npc_move_hard(game_state: GameState, npc_player_info: PlayerInfo) -> Move | None:
    gs = game_state
    game_board_size = gs.config.grid_size
    zwick_eval_size = 3

    opponent_info = gs.get_opponent_info(npc_player_info.id)
    if not opponent_info:
        gs.last_error_message = "NPC Error (Hard): Opponent info not found."
        # Fallback to a simpler NPC or random move if opponent info is missing
        # For simplicity, let's assume this means error or we can't proceed with complex logic
        print("Error: Opponent info missing for Hard NPC, cannot use Minimax.")
        # Attempt to use Medium as a fallback
        medium_move = get_npc_move_medium(game_state, npc_player_info)
        if medium_move: return medium_move
        # If medium also fails, or as a very basic fallback:
        # Find any valid random move (simplified from minimax's move generation)
        # This fallback part needs robust valid move generation if used.
        # For now, main path assumes opponent_info is present.
        return None


    EngineClass = UltimateTTTGameEngine

    # Initial call to minimax
    # The NPC is the maximizing player, and it's their turn effectively for this decision.
    # current_gs.current_player_id should be npc_player_info.id at the point of calling this function
    if gs.current_player_id != npc_player_info.id:
        gs.last_error_message = "NPC Error (Hard): Called when not NPC's turn."
        print(f"Error: Hard NPC called when not its turn. Current: {gs.current_player_id}, NPC: {npc_player_info.id}")
        return None # Cannot make a move if it's not its turn

    # Note: The `is_maximizing_player` in the minimax call refers to the player whose perspective
    # we are maximizing *at that level of the tree*.
    # The first call to minimax for the NPC's move decision should consider the NPC as the maximizer for that initial state.
    # The minimax function itself will then alternate is_maximizing_player for subsequent recursive calls.
    #total_num_bord = game_board_size ** 2
    #total_num_on_board = sum(1 for row in gs.local_boards_state for board in row for cell in board if cell != CellState.EMPTY)
    # if over 60% full increc MINIMAX_SEARCH_DEPTH
    #used_depth = min(MINIMAX_SEARCH_DEPTH, int(5 * (total_num_bord / total_num_on_board) + 1))
    #print(f"NPC USED DEPTH: {used_depth}", int(5 * (total_num_bord / total_num_on_board) + 1))
    best_score, best_move_dict = minimax(
        current_gs=gs,
        depth=MINIMAX_SEARCH_DEPTH,
        alpha=-math.inf,
        beta=math.inf,
        is_maximizing_player=True, # NPC is the maximizer for its own turn
        npc_player_info=npc_player_info,
        opponent_player_info=opponent_info,
        engine_class=EngineClass,
        zwick_eval_size=zwick_eval_size,
        game_board_size=game_board_size,
        initial_call=True # This is the top-level call to get the best move
    )

    if best_move_dict:
        # print(f"HARD NPC MINIMAX best_score: {best_score}, best_move: {best_move_dict}")
        return Move(
            player_id=npc_player_info.id,
            global_row=best_move_dict['gr'],
            global_col=best_move_dict['gc'],
            local_row=best_move_dict['lr'],
            local_col=best_move_dict['lc']
        )
    else:
        # This means no move was found (e.g., game already over or no possible moves from initial state)
        # or an error occurred in minimax.
        gs.last_error_message = "NPC Error (Hard): Minimax did not return a valid move."
        print(f"Hard NPC Minimax failed to find a move. Score: {best_score}. Falling back if possible.")
        # Fallback to medium or simpler logic if Minimax fails (e.g., unexpected state)
        return get_npc_move_medium(gs, npc_player_info)

NPC_DISPATCHER = {
    NPCDifficulty.EASY: get_npc_move_easy,
    NPCDifficulty.MEDIUM: get_npc_move_medium,
    NPCDifficulty.HARD: get_npc_move_hard,
    # NPCDifficulty.INSANE: get_npc_move_insane,
}


# --- Database Functions --- (Using model_dump(mode='json') and model_validate_json)
init = [False]

def get_db(app: App, name="Main"):
    if not init[0]:
        app.save_load("DB", spec=f"{Name}_DB")
        db = app.get_mod("DB", spec=f"{Name}_DB")
        db.edit_cli("LD")
        init.append(db)
        init[0] = True
        print("DB initialized")
    return init[-1]


async def save_game_to_db_final(app: App, game_state: GameState):  # Renamed
    db = get_db(app, name="save_game_to_db_final")
    key = f"{DB_GAMES_PREFIX}_{game_state.game_id}"
    db.set(key, game_state.model_dump_json(exclude_none=True))  # Pydantic v2 handles json string


async def load_game_from_db_final(app: App, game_id: str) -> GameState | None:  # Renamed
    db = get_db(app, name="load_game_from_db_final")
    key = f"{DB_GAMES_PREFIX}_{game_id}"
    result = db.get(key)
    if result.is_data() and result.get():
        try:
            data = result.get()[0] if isinstance(result.get(), list) else result.get()
            return GameState.model_validate_json(data)
        except Exception as e:
            app.logger.error(f"Error validating/loading game {game_id} from DB: {e}", exc_info=True)
    return None


async def get_user_stats(app: App, session_id: str) -> UserSessionStats:  # Renamed
    db = get_db(app, name="get_user_stats")
    key = f"{DB_USER_STATS_PREFIX}_{session_id}"
    result = db.get(key)
    if result.is_data() and result.get():
        try:
            data = result.get()[0] if isinstance(result.get(), list) else result.get()
            return UserSessionStats.model_validate_json(data)
        except Exception:
            pass
    return UserSessionStats(session_id=session_id)


async def save_user_stats(app: App, stats: UserSessionStats):  # Renamed
    db = get_db(app, name="save_user_stats")
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
async def api_create_game(app: App, request: RequestData, data=None):
    try:
        payload = data or {}
        config_data = payload.get("config", {})
        config = GameConfig(**config_data)  # Validate grid_size here

        mode_str = payload.get("mode", GameMode.LOCAL.value) # Default to local if not specified
        mode = GameMode(mode_str)

        player1_name = payload.get("player1_name", "Player 1").strip()
        if not player1_name: player1_name = "Player 1"


        initial_status = GameStatus.IN_PROGRESS

        if mode == GameMode.ONLINE:
            initial_status = GameStatus.WAITING_FOR_OPPONENT

        game_state = GameState(config=config, mode=mode, status=initial_status)
        engine = UltimateTTTGameEngine(game_state)

        async def get_p1_id_helper():
            user = await get_user_from_request(app, request)
            if user and user.uid:
                return user.uid
            # Fallback for guests or environments where user.uid might not be set for P1.
            # Using a consistent session_id based part is better than pure random for potential reconnects.
            if request and request.session_id:
                 # Ensure a unique prefix to avoid clashes if P2 also becomes a guest with similar session ID part.
                return f"p1_guest_{request.session_id[:8]}"
            return f"p1_guest_{uuid.uuid4().hex[:8]}" # Ultimate fallback

        player1_id = LOCAL_PLAYER_X_ID if mode == GameMode.LOCAL else await get_p1_id_helper()
        engine.add_player(player1_id, player1_name)

        if mode == GameMode.LOCAL:
            # Default player2_type to "npc" if not provided in payload
            player2_type = payload.get("player2_type", "npc")
            player2_name_human = payload.get("player2_name", "Player 2").strip()
            if not player2_name_human and player2_type == "human": player2_name_human = "Player 2"


            if player2_type == "npc":
                npc_difficulty_str = payload.get("npc_difficulty", NPCDifficulty.EASY.value)
                try:
                    npc_difficulty = NPCDifficulty(npc_difficulty_str)
                except ValueError:
                    app.logger.warning(f"Invalid NPC difficulty '{npc_difficulty_str}' provided. Defaulting to Easy.")
                    npc_difficulty = NPCDifficulty.EASY
                npc_id = f"{NPC_PLAYER_ID_PREFIX}{npc_difficulty.value}"
                npc_name = f"NPC ({npc_difficulty.value.capitalize()})"
                engine.add_player(npc_id, npc_name, is_npc=True, npc_difficulty=npc_difficulty)
            else:  # Human Player 2
                engine.add_player(LOCAL_PLAYER_O_ID, player2_name_human)
            # game_state.status is already IN_PROGRESS by add_player when P2 is added.

        await save_game_to_db_final(app, game_state)
        app.logger.info(
            f"Created {mode.value} game {game_state.game_id} (Size: {config.grid_size}) P1: {player1_name} ({player1_id}). P2 setup: {payload.get('player2_type', 'npc' if mode == GameMode.LOCAL else 'online_joiner')}")

        response_data = game_state.model_dump_for_api()

        if mode == GameMode.ONLINE and game_state.status == GameStatus.WAITING_FOR_OPPONENT:
            app_base_url = os.environ.get('APP_BASE_URL')
            if not app_base_url:
                app.logger.warning("APP_BASE_URL environment variable is not set. Join links may be incomplete or use relative paths.")
                # Fallback to relative path if APP_BASE_URL is not set
                # This assumes the UI is served relative to the API path structure.
                # A more robust solution might involve request.host_url if available from the framework.
                # For now, using a path that should work if API and UI are on same domain.
                app_base_url = "" # Results in a relative path like /api/UltimateTTT/ui?...

            join_url = f"{app_base_url}/api/{GAME_NAME}/ui?join={game_state.game_id}"
            response_data['join_url'] = join_url

        return Result.json(data=response_data)
    except ValueError as e: # Handles GameMode, NPCDifficulty enum errors, Pydantic validation
        app.logger.warning(f"Create game input error: {e}")
        return Result.default_user_error(f"Invalid input: {str(e)}", 400)
    except Exception as e:
        app.logger.error(f"Error creating game: {e}", exc_info=True)
        return Result.default_internal_error("Could not create game.")

# --- END OF BLOCK 5 ---


# --- START OF BLOCK 3 (api_join_game) ---
# FILE: ultimate_ttt_api.py
# Replace the existing api_join_game function.

@export(mod_name=GAME_NAME, name="join_game", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_join_game(app: App, request: RequestData, data=None):
    try:
        payload = data or {}
        game_id = payload.get("game_id")
        player_name_from_join_attempt = payload.get("player_name", "Player 2").strip()
        if not player_name_from_join_attempt: player_name_from_join_attempt = "Player 2"

        if not game_id:
            return Result.default_user_error("Game ID required.", 400)

        game_state = await load_game_from_db_final(app, game_id)
        if not game_state:
            return Result.default_user_error("Game not found.", 404)

        user = await get_user_from_request(app, request)
        # Generate a more robust guest ID for joining if user.uid is not available
        joiner_id_from_request = user.uid if user and user.uid else f"p2_guest_{request.session_id[:8]}" if request and request.session_id else f"p2_guest_{uuid.uuid4().hex[:8]}"

        if game_state.mode != GameMode.ONLINE:
            return Result.default_user_error("Not an online game.", 400)

        engine = UltimateTTTGameEngine(game_state)
        already_in_game_as_player = game_state.get_player_info(joiner_id_from_request)

        if game_state.status == GameStatus.WAITING_FOR_OPPONENT:
            app.logger.info(
                f"Player {joiner_id_from_request} ({player_name_from_join_attempt}) attempting to join WAITING game {game_id}.")
            if already_in_game_as_player:  # P1 trying to "rejoin"
                if not already_in_game_as_player.is_connected:
                    engine.handle_player_reconnect(joiner_id_from_request)
            elif len(game_state.players) < 2:  # New player (P2) joins
                # Ensure P2's ID doesn't clash with P1's ID if P1 is also a guest from a similar session
                if game_state.players[0].id == joiner_id_from_request:
                    joiner_id_from_request = f"p2_guest_{uuid.uuid4().hex[:8]}"  # Force unique if P1 has same generated ID
                    app.logger.warning(
                        f"Potential P2 ID clash with P1 guest ID. Regenerated P2 ID to: {joiner_id_from_request}")

                if not engine.add_player(joiner_id_from_request, player_name_from_join_attempt):
                    return Result.default_user_error(
                        game_state.last_error_message or "Could not join (add player failed).", 400)
            else:
                return Result.default_user_error("Game is waiting but seems full. Cannot join.", 409)

        elif game_state.status == GameStatus.ABORTED:
            app.logger.info(
                f"Player {joiner_id_from_request} attempting to join/reconnect to ABORTED game {game_id} (paused by {game_state.player_who_paused}).")
            if already_in_game_as_player:
                # This player is one of the original players.
                engine.handle_player_reconnect(joiner_id_from_request)  # This will handle paused or hard-aborted cases.
            elif len(game_state.players) < 2 and not game_state.player_who_paused:
                # Game was WAITING, P1 left, game hard-aborted. Now a *new* P2 tries to join.
                # This is not directly supported by current handle_reconnect, treat as cannot join.
                return Result.default_user_error(
                    "Game was aborted before an opponent joined and cannot be joined by a new player now.", 403)
            else:  # A new player trying to join a game that was already started and then aborted
                return Result.default_user_error("Game was aborted. Only original players can attempt to resume.", 403)


        elif game_state.status == GameStatus.IN_PROGRESS:
            app.logger.info(
                f"Player {joiner_id_from_request} attempting to join/reconnect to IN_PROGRESS game {game_id}.")
            if already_in_game_as_player:
                if not already_in_game_as_player.is_connected:
                    engine.handle_player_reconnect(joiner_id_from_request)
            else:
                return Result.default_user_error("Game is already in progress and full.", 403)

        elif game_state.status == GameStatus.FINISHED:
            return Result.default_user_error(f"Game is {game_state.status.value} and cannot be joined.", 400)

        else:  # Should not be reached
            return Result.default_user_error(
                f"Game is in an unexpected state ({game_state.status.value}). Cannot join.", 500)

        await save_game_to_db_final(app, game_state)
        app.logger.info(
            f"Join/Reconnect attempt processed for player {joiner_id_from_request} in game {game_id}. New status: {game_state.status}")

        response_data = game_state.model_dump_for_api()
        # Ensure join_url is included if game becomes WAITING_FOR_OPPONENT after a reconnect
        if game_state.mode == GameMode.ONLINE and game_state.status == GameStatus.WAITING_FOR_OPPONENT:
            app_base_url = os.environ.get('APP_BASE_URL', "")
            join_url = f"{app_base_url}/api/{GAME_NAME}/ui?join={game_state.game_id}"
            response_data['join_url'] = join_url

        return Result.json(data=response_data)

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

    response_data = game_state.model_dump_for_api()

    if game_state.mode == GameMode.ONLINE and \
        game_state.status == GameStatus.WAITING_FOR_OPPONENT:
        # Generate join_url for sharing
        app_base_url = os.environ.get('APP_BASE_URL')
        if not app_base_url:
            app.logger.warning("APP_BASE_URL environment variable is not set for get_game_state. Join links may be incomplete or use relative paths.")
            app_base_url = "" # Results in a relative path

        join_url = f"{app_base_url}/api/{GAME_NAME}/ui?join={game_state.game_id}"
        response_data['join_url'] = join_url

        # Timeout logic
        if game_state.waiting_since and \
           (datetime.now(UTC) - game_state.waiting_since > timedelta(seconds=ONLINE_POLL_TIMEOUT_SECONDS)):
            game_state.status = GameStatus.ABORTED
            game_state.last_error_message = "Game aborted: Opponent didn't join in time."
            game_state.updated_at = datetime.now(UTC)
            await save_game_to_db_final(app, game_state)
            # Update response_data with the new status etc.
            response_data = game_state.model_dump_for_api()
            response_data['join_url'] = join_url # Re-add join_url as model_dump_for_api won't have it

    return Result.json(data=response_data)

# --- START OF BLOCK 6 (Modify api_make_move for NPC) ---
# FILE: ultimate_ttt_api.py
# Modify the api_make_move endpoint function

@export(mod_name=GAME_NAME, name="make_move", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_make_move(app: App, request: RequestData, data=None):
    move_payload = data or {}
    game_id: str = move_payload.get("game_id")
    human_player_id_making_move: str | None = move_payload.get("player_id")  # ID of human submitting the move
    game_state = None

    try:
        game_state = await load_game_from_db_final(app, game_id)
        if not game_state: return Result.default_user_error("Game not found.", 404)

        # Initial human move processing
        if "game_id" in move_payload: del move_payload["game_id"]

        # Validate human move first
        current_player_info = game_state.get_current_player_info()
        if not current_player_info or current_player_info.is_npc:
            # This should not happen if UI prevents human from moving for NPC
            # Or, if it's an NPC's turn initiated by server after a human move.
            # For now, assume make_move is always initiated by a human player's action.
            if current_player_info and current_player_info.is_npc:
                app.logger.warning(
                    f"make_move API called but current player {current_player_info.id} is NPC. Game: {game_id}. This implies server-side NPC turn logic is expected elsewhere or flow is mixed.")
                # This path is if the API is called FOR an NPC - which we will do internally.
                # If called EXTERNALLY for an NPC, it's an issue.
                # The code below assumes this API call is for a HUMAN move, then it triggers NPC if needed.

        human_move = Move(**move_payload)
        engine = UltimateTTTGameEngine(game_state)

        if not engine.make_move(human_move):
            return Result.default_user_error(
                game_state.last_error_message or "Invalid human move.", 400,
                data=game_state.model_dump_for_api()
            )

        # Loop for NPC moves if it's their turn after a human move
        while game_state.status == GameStatus.IN_PROGRESS:
            current_player_info = game_state.get_current_player_info()
            if not current_player_info:  # Should not happen
                game_state.last_error_message = "Error: No current player identified after a move."
                break

            if current_player_info.is_npc and current_player_info.npc_difficulty:
                app.logger.info(
                    f"NPC {current_player_info.name} ({current_player_info.id}) turn in game {game_id}. Diff: {current_player_info.npc_difficulty.value}")

                # Brief delay to make NPC seem like it's "thinking"
                # For local play, this might be too fast. For UI, it's good.
                # Consider only delaying if the *next* player is NPC, not if P1 is NPC on game start.
                await asyncio.sleep(0.3)  # Adjust delay as needed

                npc_logic_func = NPC_DISPATCHER.get(current_player_info.npc_difficulty)
                if not npc_logic_func:
                    game_state.last_error_message = f"NPC Error: No logic for difficulty {current_player_info.npc_difficulty.value}"
                    app.logger.error(game_state.last_error_message)
                    # Potentially abort game or mark as error state
                    break

                npc_move = npc_logic_func(game_state, current_player_info)

                if not npc_move:
                    # This means NPC couldn't find a move, game might be a draw or error.
                    # The NPC logic should ideally not return None if valid moves exist.
                    # The make_move in engine will check for overall game end if no moves left.
                    app.logger.warning(
                        f"NPC {current_player_info.name} could not determine a move. Game state: {game_state.status}")
                    # If no moves were possible, engine.make_move would have already set draw/finished
                    # or the NPC logic itself identified no moves.
                    # This break is if npc_logic_func returns None when it shouldn't.
                    if game_state.status == GameStatus.IN_PROGRESS:  # If NPC logic failed but game thinks it's on
                        game_state.is_draw = True  # Fallback, assume draw if NPC fails weirdly
                        game_state.status = GameStatus.FINISHED
                        game_state.last_error_message = "NPC failed to move; game ended as draw."
                    break

                app.logger.info(
                    f"NPC {current_player_info.name} chose move: G({npc_move.global_row},{npc_move.global_col}) L({npc_move.local_row},{npc_move.local_col})")

                if not engine.make_move(npc_move):
                    # This is a more critical error: NPC generated an invalid move.
                    game_state.last_error_message = f"NPC Error: Generated invalid move. {game_state.last_error_message or ''}"
                    app.logger.error(
                        f"CRITICAL NPC ERROR: NPC {current_player_info.name} made invalid move {npc_move.model_dump_json()} in game {game_id}. Error: {game_state.last_error_message}")
                    # Abort or handle error appropriately. For now, break and let current state be saved.
                    game_state.status = GameStatus.ABORTED  # Or some error status
                    break
            else:
                # It's a human player's turn now, or game ended. Break the NPC move loop.
                break

        # Save game state after all human and subsequent NPC moves are done
        await save_game_to_db_final(app, game_state)

        if game_state.status == GameStatus.FINISHED:
            await update_stats_after_game_final(app, game_state)

        return Result.json(data=game_state.model_dump_for_api())

    except ValueError as e:  # Pydantic validation error for human_move usually
        app.logger.warning(f"Make move input error for game {game_id}: {e}")
        if game_state:  # Try to return current game state with the error
            game_state.last_error_message = f"Invalid move data: {str(e)}"
            return Result.default_user_error(game_state.last_error_message, 400, data=game_state.model_dump_for_api())
        return Result.default_user_error(f"Invalid move data: {str(e)}", 400)
    except Exception as e:
        app.logger.error(f"Error making move in game {game_id}: {e}", exc_info=True)
        if game_state:
            game_state.last_error_message = "Internal server error during move processing."
            try:
                await save_game_to_db_final(app, game_state)  # Attempt to save error state
            except:
                pass
            return Result.default_internal_error("Could not process move.", data=game_state.model_dump_for_api())
        return Result.default_internal_error("Could not process move.")




@export(mod_name=GAME_NAME, name="get_session_stats", api=True, request_as_kwarg=True)
async def api_get_session_stats(app: App, request: RequestData, session_id: str | None = None):
    id_for_stats = session_id
    if not id_for_stats:  # Try to get from Toolbox user if no explicit session_id
        user = await get_user_from_request(app, request)
        if user and user.uid:
            id_for_stats = user.uid
        else:
            return Result.default_user_error("Session ID or user context required for stats.", 400)

    stats = await get_user_stats(app, id_for_stats)
    return Result.json(data=stats.model_dump(mode='json'))



# --- START OF MODIFIED FUNCTION: api_open_game_stream ---
# FILE: ultimate_ttt_api.py

@export(mod_name=GAME_NAME, name="open_game_stream", api=True, request_as_kwarg=True, api_methods=['GET'])
async def api_open_game_stream(app: App, request: RequestData, game_id: str, player_id: str | None = None):
    if not game_id:
        async def error_gen_no_id():
            yield {'event': 'error', 'data': {'message': 'game_id is required for stream'}}

        return Result.sse(stream_generator=error_gen_no_id())

    listening_player_id = player_id

    async def game_event_generator() -> AsyncGenerator[dict[str, Any], None]:
        app.logger.info(f"SSE: Stream opened for game_id: {game_id} by player_id: {listening_player_id or 'Unknown'}")
        last_known_updated_at = None
        last_status_sent = None
        last_players_connected_state: dict[str, bool] | None = None

        try:
            while True:
                game_state = await load_game_from_db_final(app, game_id)

                if not game_state:
                    app.logger.warning(f"SSE: Game {game_id} not found. Closing stream.")
                    yield {'event': 'error', 'data': {'message': 'Game not found. Stream closing.'}}
                    yield {'event': 'stream_end', 'data': {'message': 'Game not found.'}}  # Ensure client knows to stop
                    break

                # Timeout for games WAITING_FOR_OPPONENT (initial join)
                if game_state.status == GameStatus.WAITING_FOR_OPPONENT and \
                    game_state.waiting_since and \
                    (datetime.now(UTC) - game_state.waiting_since > timedelta(
                        seconds=ONLINE_POLL_TIMEOUT_SECONDS)):
                    app.logger.info(f"SSE: Game {game_id} timed out waiting for opponent (initial join). Aborting.")
                    game_state.status = GameStatus.ABORTED
                    game_state.last_error_message = "Game aborted: Opponent didn't join in time."
                    game_state.player_who_paused = None
                    game_state.updated_at = datetime.now(UTC)
                    await save_game_to_db_final(app, game_state)
                    yield {'event': 'game_update', 'data': game_state.model_dump_for_api()}
                    yield {'event': 'stream_end', 'data': {'message': 'Game timed out waiting for opponent.'}}
                    break

                # Timeout for games PAUSED (ABORTED with player_who_paused) - now 24 hours
                if game_state.status == GameStatus.ABORTED and \
                    game_state.player_who_paused and \
                    game_state.updated_at and \
                    (datetime.now(UTC) - game_state.updated_at > timedelta(
                        seconds=PAUSED_GAME_RESUME_WINDOW_SECONDS)):

                    disconnected_player_name = "Player"
                    paused_player_info = game_state.get_player_info(game_state.player_who_paused)
                    if paused_player_info: disconnected_player_name = paused_player_info.name

                    app.logger.info(
                        f"SSE: Game {game_id} (paused) timed out after {PAUSED_GAME_RESUME_WINDOW_SECONDS // 3600} hours waiting for {disconnected_player_name} ({game_state.player_who_paused}) to reconnect. Fully aborting.")
                    game_state.last_error_message = f"Game aborted: {disconnected_player_name} did not reconnect within the extended timeframe."
                    game_state.player_who_paused = None
                    game_state.updated_at = datetime.now(UTC)
                    await save_game_to_db_final(app, game_state)
                    yield {'event': 'game_update', 'data': game_state.model_dump_for_api()}
                    yield {'event': 'stream_end', 'data': {'message': 'Paused game timed out.'}}
                    break

                current_updated_at = game_state.updated_at
                current_status = game_state.status
                current_players_connected = {p.id: p.is_connected for p in
                                             game_state.players} if game_state.players else {}
                send_update = False

                if last_known_updated_at is None or \
                    current_updated_at > last_known_updated_at or \
                    current_status != last_status_sent or \
                    current_players_connected != last_players_connected_state:
                    send_update = True

                if send_update:
                    app.logger.debug(
                        f"SSE: Sending update for game {game_id}. Status: {current_status}, Updated: {current_updated_at}, Connected: {current_players_connected}")
                    yield {'event': 'game_update', 'data': game_state.model_dump_for_api()}
                    last_known_updated_at = current_updated_at
                    last_status_sent = current_status
                    last_players_connected_state = current_players_connected

                if game_state.status == GameStatus.FINISHED or \
                    (game_state.status == GameStatus.ABORTED and not game_state.player_who_paused):
                    app.logger.info(
                        f"SSE: Game {game_id} is {game_state.status.value} (final). Sent final update. Closing stream.")
                    yield {'event': 'stream_end', 'data': {'message': f'Game is {game_state.status.value}.'}}
                    break

                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            app.logger.info(
                f"SSE: Stream for game_id: {game_id}, listening_player_id: {listening_player_id} was CANCELLED (client likely disconnected).")
            if listening_player_id and game_id:
                game_state_on_disconnect = await load_game_from_db_final(app, game_id)
                if game_state_on_disconnect and game_state_on_disconnect.mode == GameMode.ONLINE:
                    player_info = game_state_on_disconnect.get_player_info(listening_player_id)

                    if player_info and player_info.is_connected and \
                        (game_state_on_disconnect.status == GameStatus.IN_PROGRESS or \
                         game_state_on_disconnect.status == GameStatus.WAITING_FOR_OPPONENT or \
                         (
                             game_state_on_disconnect.status == GameStatus.ABORTED and game_state_on_disconnect.player_who_paused)):

                        app.logger.info(
                            f"SSE: Processing server-side disconnect for player {listening_player_id} in game {game_id} due to stream cancellation.")
                        engine = UltimateTTTGameEngine(game_state_on_disconnect)
                        engine.handle_player_disconnect(
                            listening_player_id)  # This updates status, player_who_paused etc.
                        await save_game_to_db_final(app, game_state_on_disconnect)
                        app.logger.info(
                            f"SSE: Post-disconnect save for game {game_id}. New status: {game_state_on_disconnect.status}, Paused by: {game_state_on_disconnect.player_who_paused}")
                    else:
                        app.logger.info(
                            f"SSE: Player {listening_player_id} stream cancelled, but no server-side action needed. Player might have been already marked disconnected, or game status ({game_state_on_disconnect.status if game_state_on_disconnect else 'N/A'}) not actionable for this disconnect event.")
        except Exception as e:  # pragma: no cover
            app.logger.error(f"SSE: Stream error for game_id {game_id}, player {listening_player_id}: {e}",
                             exc_info=True)
            try:
                yield {'event': 'error', 'data': {'message': f'Server error in stream: {str(e)}'}}
            except Exception as yield_e:  # pragma: no cover
                app.logger.error(f"SSE: Error yielding error message for game_id {game_id}: {yield_e}", exc_info=True)
        finally:
            app.logger.info(f"SSE: Stream closed for game_id: {game_id}, listening_player_id: {listening_player_id}")

    return Result.sse(stream_generator=game_event_generator())
# --- END OF MODIFIED FUNCTION: api_open_game_stream ---

# --- END OF BLOCK 4 ---


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
# --- START OF MODIFIED FUNCTION: ultimate_ttt_ui_page ---
# FILE: ultimate_ttt_api.py

@export(mod_name=GAME_NAME, version=VERSION, level=0, api=True, name="ui", state=False)
def ultimate_ttt_ui_page(app_ref: App | None = None, **kwargs):
    app_instance = app_ref if app_ref else get_app(GAME_NAME)
    # Full HTML, CSS, and JS
    html_and_js_content = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultimate Tic-Tac-Toe</title>
    <meta property="og:image" content="https://simplecore.app/web/webapp/TTTimg.png">

<style>
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: var(--font-family-base);
    background-color: var(--theme-bg);
    color: var(--theme-text);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: var(--spacing);
    min-height: 100vh;
}

.app-header {
    width: 100%;
    max-width: 900px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding: 0.5rem 0;
}

.app-title {
    font-size: var(--font-size-3xl);
    color: var(--theme-primary);
    margin: 0;
    font-weight: var(--font-weight-semibold);
}

.theme-switcher {
    padding: 0.5rem var(--spacing);
    background-color: var(--glass-bg);
    color: var(--theme-text);
    border: 1px solid var(--theme-border);
    border-radius: var(--radius-md);
    cursor: pointer;
    font-size: var(--font-size-sm);
    box-shadow: var(--glass-shadow);
}

.theme-switcher:hover {
    opacity: 0.8;
}

.current-player-indicator-container {
    width: 100%;
    max-width: 900px;
    margin-bottom: var(--spacing);
    display: flex;
    justify-content: center;
}

.current-player-indicator {
    height: 10px;
    width: 100%;
    max-width: 500px;
    border-radius: 5px;
    background-color: var(--theme-border);
    transition: background-color var(--transition-medium);
    box-shadow: var(--glass-shadow);
}

.current-player-indicator.player-X {
    background-color: var(--tb-color-info-500);
}

.current-player-indicator.player-O {
    background-color: var(--tb-color-error-500);
}

.main-content-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    max-width: 900px;
}

.section-card {
    background-color: var(--glass-bg);
    padding: clamp(1rem, 3vw, 1.5rem);
    border-radius: var(--radius-lg);
    box-shadow: var(--glass-shadow);
    margin-bottom: 1.5rem;
    width: 100%;
    max-width: 500px;
    border: 1px solid var(--glass-border);
}

.section-card h2, .section-card h3 {
    font-size: var(--font-size-xl);
    margin-bottom: var(--spacing);
    color: var(--theme-primary);
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--theme-border);
}

.form-group {
    margin-bottom: var(--spacing);
}

.form-group label {
    display: block;
    font-weight: var(--font-weight-medium);
    margin-bottom: 0.5rem;
    color: var(--theme-text-muted);
    font-size: var(--font-size-sm);
}

.form-input, .form-select {
    width: 100%;
    padding: 0.6rem 0.75rem;
    border-radius: var(--radius-md);
    border: 1px solid var(--input-border);
    background-color: var(--input-bg);
    color: var(--theme-text);
    font-size: var(--font-size-base);
}

.form-input:focus, .form-select:focus {
    outline: none;
    border-color: var(--input-focus-border);
    box-shadow: 0 0 0 0.15rem rgba(59, 130, 246, 0.25);
}

.button-row {
    display: flex;
    gap: 0.75rem;
    margin-top: var(--spacing);
    flex-wrap: wrap;
}

.button {
    padding: 0.6rem 1.2rem;
    border-radius: var(--radius-md);
    border: none;
    cursor: pointer;
    font-weight: var(--font-weight-medium);
    font-size: var(--font-size-sm);
    transition: background-color var(--transition-fast), transform 0.1s;
    text-align: center;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background-color: var(--button-bg);
    color: var(--button-text);
}

.button:hover {
    background-color: var(--link-hover-color);
}

.button:active {
    transform: translateY(1px);
}

.button.secondary {
    background-color: var(--theme-secondary);
    color: var(--theme-text-on-primary);
}

.button.secondary:hover {
    filter: brightness(0.9);
}

.button.danger {
    background-color: var(--color-error);
    color: var(--theme-text-on-primary);
}

.button.danger:hover {
    filter: brightness(0.9);
}

.button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.status-bar {
    font-size: var(--font-size-base);
    font-weight: var(--font-weight-medium);
    text-align: center;
    padding: 0.75rem var(--spacing);
    border-radius: var(--radius-md);
    min-height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid transparent;
    margin-bottom: var(--spacing);
}

.status-bar.info {
    background-color: var(--color-info);
    color: var(--theme-text-on-primary);
    border-color: var(--color-info);
    opacity: 0.9;
}

.status-bar.error {
    background-color: var(--color-error);
    color: var(--theme-text-on-primary);
    border-color: var(--color-error);
}

.status-bar.success {
    background-color: var(--color-success);
    color: var(--theme-text-on-primary);
    border-color: var(--color-success);
}

.game-board-area {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.global-grid-display {
    display: grid;
    border: 3px solid var(--theme-border);
    background-color: var(--glass-bg);
    box-shadow: var(--glass-shadow);
    border-radius: var(--radius-lg);
    padding: clamp(3px, 1vw, 5px);
    gap: clamp(3px, 1vw, 5px);
}

.local-board-container {
    border: 2px solid var(--theme-border);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: box-shadow var(--transition-fast), border-color var(--transition-fast), background-color var(--transition-fast);
    position: relative;
}

.local-board-container.forced-target {
    box-shadow: var(--glass-shadow);
    border-color: var(--theme-primary) !important;
}

.local-board-container.playable-anywhere {
    border-color: var(--theme-secondary) !important;
    opacity: 0.9;
}

.local-board-container.won-X {
    background-color: rgba(13, 202, 240, 0.2);
}

.local-board-container.won-O {
    background-color: rgba(239, 68, 68, 0.2);
}

.local-board-container.won-DRAW {
    background-color: rgba(108, 117, 125, 0.2);
}

.local-board-container.preview-forced-for-x {
    outline: 3px dashed var(--tb-color-info-500);
    outline-offset: -3px;
}

.local-board-container.preview-forced-for-o {
    outline: 3px dashed var(--tb-color-error-500);
    outline-offset: -3px;
}

.local-board-container .winner-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: var(--font-weight-bold);
    pointer-events: none;
    opacity: 0;
}

.local-board-container.won-X .winner-overlay.player-X,
.local-board-container.won-O .winner-overlay.player-O,
.local-board-container.won-DRAW .winner-overlay.draw {
    opacity: 1;
}

.winner-overlay.player-X {
    color: var(--tb-color-info-500);
}

.winner-overlay.player-O {
    color: var(--tb-color-error-500);
}

.winner-overlay.draw {
    color: var(--theme-text-muted);
}

.local-grid {
    display: grid;
    width: 100%;
    height: 100%;
    gap: clamp(1px, 0.5vw, 2px);
}

.cell {
    border: 1px solid var(--theme-border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: var(--font-weight-bold);
    background-color: var(--input-bg);
    transition: background-color 0.1s;
    cursor: default;
    aspect-ratio: 1/1;
}

.cell.playable {
    cursor: pointer;
}

.cell.playable:hover {
    background-color: var(--glass-bg);
}

.cell.player-X {
    color: var(--tb-color-info-500);
}

.cell.player-O {
    color: var(--tb-color-error-500);
}

.cell.last-move {
    box-shadow: inset 0 0 0 2.5px gold;
}

.cell.last-move.player-X {
    box-shadow: inset 0 0 0 2.5px gold, 0 0 0 0 var(--tb-color-info-500);
}

.cell.last-move.player-O {
    box-shadow: inset 0 0 0 2.5px gold, 0 0 0 0 var(--tb-color-error-500);
}

.local-board-container.won-X .local-grid,
.local-board-container.won-O .local-grid,
.local-board-container.won-DRAW .local-grid {
    opacity: 0.5;
}

.local-board-container.won-X .cell,
.local-board-container.won-O .cell,
.local-board-container.won-DRAW .cell {
    cursor: not-allowed !important;
    background-color: transparent !important;
}

.game-controls-ingame {
    margin-top: 1.5rem;
    display: flex;
    gap: 0.75rem;
    justify-content: center;
}

.stats-area {
    text-align: center;
}

.stats-area p {
    margin: 0.2rem 0;
    font-size: var(--font-size-sm);
    color: var(--theme-text-muted);
}

.stats-area strong {
    color: var(--theme-text);
    font-weight: var(--font-weight-semibold);
}

.hidden {
    display: none !important;
}

.local-board-container.inactive-target {
    opacity: 0.6;
}

.local-board-container.inactive-target .cell.playable {
    cursor: not-allowed;
    background-color: var(--theme-border) !important;
}

.local-board-container.inactive-target .cell:not(.player-X):not(.player-O):hover {
    background-color: var(--input-bg);
}

.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0,0,0,0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: var(--z-modal);
}

.modal-content {
    background-color: var(--glass-bg);
    padding: 1.5rem 2rem;
    border-radius: var(--radius-lg);
    box-shadow: var(--glass-shadow);
    max-width: 420px;
    width: 90%;
    text-align: center;
    border: 1px solid var(--glass-border);
}

.modal-title {
    font-size: var(--font-size-xl);
    color: var(--theme-primary);
    margin-bottom: var(--spacing);
    font-weight: var(--font-weight-semibold);
}

.modal-message {
    margin-bottom: 1.5rem;
    font-size: var(--font-size-base);
    color: var(--theme-text-muted);
    line-height: var(--line-height-normal);
}

.modal-buttons {
    display: flex;
    gap: 0.75rem;
    justify-content: flex-end;
}

@media (max-width: 600px) {
    .app-header {
        flex-direction: column;
        gap: 0.75rem;
    }
    .button-row {
        flex-direction: column;
    }
    .button {
        width: 100%;
    }
    .button:not(:last-child) {
        margin-bottom: 0.5rem;
    }
}

.status-text {
    font-size: var(--font-size-sm);
    padding: 0.25rem 0.5rem;
    border-radius: var(--radius-sm);
    text-align: center;
}

.status-text.info {
    background-color: rgba(13, 202, 240, 0.15);
    color: var(--color-info);
}

.status-text.warning {
    background-color: rgba(234, 179, 8, 0.15);
    color: var(--color-warning);
}

.status-text.error {
    background-color: rgba(239, 68, 68, 0.15);
    color: var(--color-error);
}
</style>
</head>
<body>
    <header class="app-header">
        <h1 class="app-title">Ultimate TTT</h1>
    </header>

    <div id="currentPlayerIndicatorContainer" class="current-player-indicator-container hidden">
        <div id="currentPlayerIndicator" class="current-player-indicator"></div>
    </div>

    <main class="main-content-wrapper">
        <section id="gameSetupSection" class="section-card">
            <h2>New Game</h2>
            <div class="form-group">
                <label for="gridSizeSelect">Grid Size (N x N):</label>
                <select id="gridSizeSelect" class="form-select">
                    <option value="2">2x2</option>
                    <option value="3" selected>3x3 (Classic)</option>
                    <option value="4">4x4</option>
                    <option value="5" disabled info="Bug" class="none">5x5</option>
                </select>
            </div>
            <div class="form-group">
                <label for="player1NameInput">Player 1 (X) Name:</label>
                <input type="text" id="player1NameInput" class="form-input" value="Player X">
            </div>

            <!-- Player 2 Setup: Default to NPC. Hidden for Online Game creation. -->
            <div id="player2LocalOptions"> <!-- Wrapper for P2 local options -->
                <div class="form-group">
                    <label for="player2TypeSelect">Player 2 (O) - Local Game:</label>
                    <select id="player2TypeSelect" class="form-select">
                        <option value="npc" selected>NPC (Computer)</option>
                        <option value="human">Human</option>
                    </select>
                </div>
                <div id="localP2NameGroup" class="form-group hidden"> <!-- Hidden by default as NPC is default -->
                    <label for="player2NameInput">Player 2 (O) Name:</label>
                    <input type="text" id="player2NameInput" class="form-input" value="Player O">
                </div>
                <div id="npcDifficultyGroup" class="form-group"> <!-- Shown by default as NPC is default -->
                    <label for="npcDifficultySelect">NPC Difficulty:</label>
                    <select id="npcDifficultySelect" class="form-select">
                        <option value="easy" selected>Easy</option>
                        <option value="medium">Medium</option>
                        <option value="hard">Hard</option>
                        <!-- <option value="insane">Insane</option> -->
                    </select>
                </div>
            </div>

            <div class="button-row">
                <button id="startLocalGameBtn" class="button">Play Local Game</button>
                 <button id="resumeLocalGameBtn" class="button secondary hidden">Resume Local <span id="resumeGridSizeText"></span> Game</button>
                <button id="startOnlineGameBtn" class="button secondary">Create Online Game</button>
            </div>
            <div class="form-group" style="margin-top: 1.5rem;">
                <label for="joinGameIdInput">Join Online Game by ID or Link:</label>
                <div style="display: flex; gap: 0.5rem;">
                    <input type="text" id="joinGameIdInput" class="form-input" placeholder="Enter Game ID or Full Link">
                    <button id="joinOnlineGameBtn" class="button secondary">Join</button>
                </div>
            </div>
        </section>

        <section id="onlineWaitSection" class="section-card hidden">
            <h2>Online Game Lobby</h2>
            <p>Share this Link or Game ID with your opponent:</p>
            <div id="gameIdShare" class="form-input" style="font-weight: bold; text-align: center; margin: 0.75rem 0; cursor: pointer; background-color: var(--cell-hover); word-break: break-all;" title="Click to copy Game Link/ID">GAME_ID_OR_LINK_HERE</div>
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
                <button id="saveAndLeaveBtn" class="button secondary hidden">Save & Leave</button>
                <button id="backToMenuBtn" class="button secondary hidden">Back to Menu</button>
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
    <div id="rulesContainer" style="font-family: Arial, Helvetica, sans-serif; margin-bottom: 15px; max-width: 400px;">
        <button id="toggleRulesButton" onclick="toggleRulesVisibility()" style="padding: 8px 12px; cursor: pointer; border: 1px solid #bbb; background-color: var(--theme-secondary); border-radius: 4px; font-size: 0.9em; color:var(--text-primary);">
            Show Rules
        </button>
       <div id="rulesContent" style="display: none; border: 1px solid #d0d0d0; padding: 10px 15px; margin-top: 10px; background-color: var(--theme-bg);color:var(--text-color); border-radius: 4px; font-size: 0.95em; line-height: 1.4;">
        <h4 style="margin-top: 0; margin-bottom: 12px; color: #333;">Ultimate Tic-Tac-Toe: Quick Rules</h4>

        <p style="margin-bottom: 8px; color: #444;">
            <strong>Grid:</strong> Played on an <strong>N x N</strong> global grid (e.g., 3x3, 4x4, up to 5x5). Each cell of this global grid is a smaller, independent <strong>N x N</strong> "local board".
        </p>

        <p style="margin-bottom: 10px; color: #444;">
            <strong>Goal:</strong> Win <strong>N local boards in a row</strong> (horizontally, vertically, or diagonally) on the global grid.
        </p>

        <hr style="border: 0; border-top: 1px solid #e0e0e0; margin: 15px 0;">

        <strong style="display: block; margin-bottom: 5px; color: #333; font-size: 1.05em;">Gameplay Mechanics:</strong>
        <ol style="padding-left: 20px; margin-top: 0; margin-bottom: 12px; color: #555;">
            <li style="margin-bottom: 5px;">Players alternate turns placing their mark (X or O).</li>
            <li style="margin-bottom: 5px;">Winning a local board claims that corresponding cell on the global grid for that player.</li>
            <li style="margin-bottom: 5px;">
                <strong>The "Send" Rule:</strong> The specific cell (e.g., top-left) where you play within a local board dictates which local board (e.g., the top-left local board) your opponent <strong>must</strong> play in on their next turn.
            </li>
        </ol>

        <strong style="display: block; margin-bottom: 5px; color: #333; font-size: 1.05em;">Special Conditions:</strong>
        <ul style="padding-left: 20px; margin-top: 0; color: #555;">
            <li style="margin-bottom: 5px;">
                <strong>Sent to a Decided/Full Board:</strong> If the local board you are "sent" to is already won or completely full (a draw), you can then play your mark in <strong>any other available cell</strong> on <strong>any other local board</strong> that is not yet decided.
            </li>
            <li style="margin-bottom: 5px;">The first move of the game is a free choice anywhere.</li>
            <li style="margin-bottom: 5px;">A local board can end in a draw. The overall game can also end in a draw if no valid moves remain.</li>
        </ul>
    </div>

    <script unsave="true">
         function toggleRulesVisibility() {
            var rulesDiv = document.getElementById('rulesContent');
            var button = document.getElementById('toggleRulesButton');
            if (rulesDiv.style.display === 'none' || rulesDiv.style.display === '') {
                rulesDiv.style.display = 'block';
                button.textContent = 'Hide Rules';
            } else {
                rulesDiv.style.display = 'none';
                button.textContent = 'Show Rules';
            }
        }
    </script>

    <script unsave="true">
    (function() {
        "use strict";

        let gameSetupSection, statsSection, onlineWaitSection, gameArea, statusBar, globalGridDisplay;
        let gridSizeSelect, player1NameInput, player2NameInput, localP2NameGroup;
        let startLocalGameBtn, startOnlineGameBtn, joinGameIdInput, joinOnlineGameBtn, gameIdShareEl, waitingStatusEl, cancelOnlineWaitBtn;
        let resetGameBtn, backToMenuBtn;
        let statsGamesPlayedEl, statsWinsEl, statsLossesEl, statsDrawsEl, statsSessionIdEl;
        let modalOverlay, modalTitle, modalMessage, modalCancelBtn, modalConfirmBtn;
        let currentPlayerIndicatorContainer, currentPlayerIndicator
        let modalConfirmCallback = null;

        let player2TypeSelect, npcDifficultyGroup, npcDifficultySelect, player2LocalOptionsDiv;


        let resumeLocalGameBtn, resumeGridSizeTextEl;
        let saveAndLeaveBtn;

        const LOCAL_STORAGE_GAME_PREFIX = "uttt_local_game_";

        let currentSessionId = null;
        let currentGameId = null;
        let currentGameState = null;
        let clientPlayerInfo = null;
        let sseConnection = null;
        let currentSseGameIdPath = null
        let localPlayerActiveSymbol = 'X';
        const API_MODULE_NAME = "UltimateTTT";

        const LOCAL_P1_ID = "p1_local_utt";
        const LOCAL_P2_ID = "p2_local_utt";

        function initApp() {
            console.log("UTTT Initializing...");
            gameSetupSection = document.getElementById('gameSetupSection');
            statsSection = document.getElementById('statsSection');
            onlineWaitSection = document.getElementById('onlineWaitSection');
            gameArea = document.getElementById('gameSection');
            statusBar = document.getElementById('statusBar');
            globalGridDisplay = document.getElementById('globalGridDisplay');

            gridSizeSelect = document.getElementById('gridSizeSelect');
            player1NameInput = document.getElementById('player1NameInput');
            player2NameInput = document.getElementById('player2NameInput');

            player2LocalOptionsDiv = document.getElementById('player2LocalOptions');
            player2TypeSelect = document.getElementById('player2TypeSelect');
            localP2NameGroup = document.getElementById('localP2NameGroup');
            npcDifficultyGroup = document.getElementById('npcDifficultyGroup');
            npcDifficultySelect = document.getElementById('npcDifficultySelect');


            startLocalGameBtn = document.getElementById('startLocalGameBtn');
            startOnlineGameBtn = document.getElementById('startOnlineGameBtn');
            joinGameIdInput = document.getElementById('joinGameIdInput');
            joinOnlineGameBtn = document.getElementById('joinOnlineGameBtn');
            gameIdShareEl = document.getElementById('gameIdShare');
            waitingStatusEl = document.getElementById('waitingStatus');
            cancelOnlineWaitBtn = document.getElementById('cancelOnlineWaitBtn');

            resetGameBtn = document.getElementById('resetGameBtn');
            backToMenuBtn = document.getElementById('backToMenuBtn');

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

            currentPlayerIndicatorContainer = document.getElementById('currentPlayerIndicatorContainer');
            currentPlayerIndicator = document.getElementById('currentPlayerIndicator');

            resumeLocalGameBtn = document.getElementById('resumeLocalGameBtn');
            resumeGridSizeTextEl = document.getElementById('resumeGridSizeText');
            saveAndLeaveBtn = document.getElementById('saveAndLeaveBtn');

            setupEventListeners();
            determineSessionId();
            loadSessionStats();
            checkUrlForJoin(); // Detects ?join=GAMEID
            updateResumeButtonVisibility();
            showScreen('gameSetup'); // This will also call togglePlayer2Setup for initial state
        }

       function connectToGameStream(gameId) {
             let ssePath = `/sse/${API_MODULE_NAME}/open_game_stream?game_id=${gameId}`;
             if (clientPlayerInfo && clientPlayerInfo.id) {
                 ssePath += `&player_id=${clientPlayerInfo.id}`;
             }

             if (sseConnection && currentSseGameIdPath === ssePath) {
                 console.log("SSE: Already connected to stream for game:", gameId, "player:", clientPlayerInfo ? clientPlayerInfo.id : "N/A");
                 return;
             }
             disconnectFromGameStream();

             if (!gameId) {
                 console.error("SSE: Cannot connect, gameId is missing.");
                 return;
             }

             currentSseGameIdPath = ssePath;
             console.log(`SSE: Attempting to connect to ${currentSseGameIdPath}`);

             sseConnection = TB.sse.connect(currentSseGameIdPath, {
                 onOpen: (event) => {
                     console.log(`SSE: Connection opened to ${currentSseGameIdPath}`, event);
                 },
                 onError: (error) => {
                     console.error(`SSE: Connection error with ${currentSseGameIdPath}`, error);
                     if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Live connection failed. Refresh may be needed.", {duration: 3000});
                 },
                 listeners: {
                     'game_update': (eventPayload, event) => {
                         console.log('SSE Event (game_update):', eventPayload);
                         if (eventPayload && eventPayload.game_id) {
                             if (currentGameId === eventPayload.game_id || !currentGameId) {
                                 processGameStateUpdate(eventPayload);
                             } else {
                                 console.warn("SSE: Received game_update for a different game_id. Current:", currentGameId, "Received:", eventPayload.game_id);
                             }
                         } else {
                             console.warn("SSE: Received game_update event without valid data.", eventPayload);
                         }
                     },
                     'error': (eventPayload, event) => {
                         console.error('SSE Event (server error):', eventPayload);
                         let errorMessage = "An error occurred in the game stream.";
                         if (eventPayload && typeof eventPayload.message === 'string') {
                            errorMessage = eventPayload.message;
                         }
                         if(window.TB?.ui?.Toast) TB.ui.Toast.showError(`Stream error: ${errorMessage}`, {duration: 4000});

                         if (errorMessage.includes("Game not found") || errorMessage.includes("game_id is required")) {
                             disconnectFromGameStream();
                             showModal("Game Error", "The game session is no longer available. Returning to menu.", () => showScreen('gameSetup'));
                         }
                     },
                     'stream_end': (eventPayload, event) => {
                          console.log('SSE Event (stream_end): Server closed stream for', gameId, "player:", clientPlayerInfo ? clientPlayerInfo.id : "N/A", eventPayload);
                          if (sseConnection && currentSseGameIdPath === ssePath) {
                              disconnectFromGameStream();
                          }
                     }
                 }
             });
         }
        function disconnectFromGameStream() {
            if (sseConnection && currentSseGameIdPath) {
                console.log(`SSE: Disconnecting from ${currentSseGameIdPath}`);
                TB.sse.disconnect(currentSseGameIdPath);
                sseConnection = null;
                currentSseGameIdPath = null;
            }
        }

        function setupEventListeners() {
            startLocalGameBtn.addEventListener('click', () => createNewGame('local', false));
            startOnlineGameBtn.addEventListener('click', () => {
                if(player2LocalOptionsDiv) player2LocalOptionsDiv.classList.add('hidden'); // Hide P2 local options for online game
                createNewGame('online');
            });


            resumeLocalGameBtn.addEventListener('click', () => createNewGame('local', true));
            saveAndLeaveBtn.addEventListener('click', saveLocalGameAndLeave);
            gridSizeSelect.addEventListener('change', updateResumeButtonVisibility);

            joinOnlineGameBtn.addEventListener('click', joinOnlineGame);
            gameIdShareEl.addEventListener('click', copyGameIdToClipboard);
            cancelOnlineWaitBtn.addEventListener('click', () => {
                showScreen('gameSetup'); // P1 cancelling their own game just returns to menu
            });

            player2TypeSelect.addEventListener('change', togglePlayer2Setup)

            resetGameBtn.addEventListener('click', confirmResetGame);
            backToMenuBtn.addEventListener('click', confirmBackToMenu);
            globalGridDisplay.addEventListener('click', onBoardClickDelegation);

            globalGridDisplay.addEventListener('mouseover', handleCellMouseOver);
            globalGridDisplay.addEventListener('mouseout', handleCellMouseOut);

            globalGridDisplay.addEventListener('touchstart', handleCellMouseOver, { passive: true });
            globalGridDisplay.addEventListener('touchend', handleCellMouseOut, { passive: true });

            modalCancelBtn.addEventListener('click', hideModal);
            modalConfirmBtn.addEventListener('click', () => {
                if (modalConfirmCallback) modalConfirmCallback();
                hideModal();
            });
        }

        function onBoardClickDelegation(event) {
            const cell = event.target.closest('.cell.playable');
            if (cell && currentGameState && currentGameState.status === 'in_progress') {
                const gr = parseInt(cell.dataset.gr);
                const gc = parseInt(cell.dataset.gc);
                const lr = parseInt(cell.dataset.lr);
                const lc = parseInt(cell.dataset.lc);

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

        function togglePlayer2Setup() {
            const isNPC = player2TypeSelect.value === 'npc';
            localP2NameGroup.classList.toggle('hidden', isNPC);
            npcDifficultyGroup.classList.toggle('hidden', !isNPC);
        }


        function getLocalStorageKeyForSize(size) {
            return `${LOCAL_STORAGE_GAME_PREFIX}${size}x${size}`;
        }
        function saveLocalGame(gameState) {
            if (!gameState || gameState.mode !== 'local' || gameState.status !== 'in_progress') return false;
            try {
                localStorage.setItem(getLocalStorageKeyForSize(gameState.config.grid_size), JSON.stringify(gameState));
                if(window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("Game saved locally!", {duration: 1500});
                return true;
            } catch (e) { console.error("Error saving local game:", e); return false; }
        }
        function loadLocalGame(size) {
            try {
                const savedGameJSON = localStorage.getItem(getLocalStorageKeyForSize(size));
                if (savedGameJSON) {
                    const savedGameState = JSON.parse(savedGameJSON);
                    if (savedGameState && savedGameState.game_id && savedGameState.config && savedGameState.mode === 'local') {
                        return savedGameState;
                    }
                }
            } catch (e) { console.error("Error loading local game:", e); }
            return null;
        }
        function deleteLocalGame(size) {
            try { localStorage.removeItem(getLocalStorageKeyForSize(size)); updateResumeButtonVisibility(); }
            catch (e) { console.error("Error deleting local game:", e); }
        }
        function updateResumeButtonVisibility() {
            if (!gridSizeSelect || !resumeLocalGameBtn || !resumeGridSizeTextEl) return;
            const selectedSize = parseInt(gridSizeSelect.value);
            const savedGame = loadLocalGame(selectedSize);
            resumeLocalGameBtn.classList.toggle('hidden', !(savedGame && savedGame.status === 'in_progress'));
            if (savedGame && savedGame.status === 'in_progress') resumeGridSizeTextEl.textContent = `${selectedSize}x${selectedSize}`;
        }
        function saveLocalGameAndLeave() {
            if (currentGameState && currentGameState.mode === 'local' && currentGameState.status === 'in_progress') {
                if (saveLocalGame(currentGameState)) showScreen('gameSetup');
            } else if(window.TB?.ui?.Toast) TB.ui.Toast.showWarning("No active local game to save.", {duration: 2000});
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
        function getPlayerInfoById(playerId) {
            if (!currentGameState || !currentGameState.players) return null;
            return currentGameState.players.find(p => p.id === playerId);
        }
        function getOpponentInfo(playerId) {
            if (!currentGameState || !currentGameState.players) return null;
            return currentGameState.players.find(p => p.id !== playerId);
        }

        function handleCellMouseOver(event) { /* No change from original, keep as is */
            if (!currentGameState || currentGameState.status !== 'in_progress') return;
            const cell = event.target.closest('.cell');
            if (!cell || !cell.classList.contains('playable')) return;
            let isMyTurn = (currentGameState.mode === 'local') || (clientPlayerInfo && currentGameState.current_player_id === clientPlayerInfo.id);
            if (!isMyTurn) return;

            const N = currentGameState.config.grid_size;
            const hovered_lr = parseInt(cell.dataset.lr), hovered_lc = parseInt(cell.dataset.lc);
            const target_gr = hovered_lr, target_gc = hovered_lc;

            if (target_gr < 0 || target_gr >= N || target_gc < 0 || target_gc >= N) return;

            const opponent = getOpponentInfo(currentGameState.current_player_id);
            if (!opponent) return;

            const targetBoardElement = globalGridDisplay.querySelector(`.local-board-container[data-gr="${target_gr}"][data-gc="${target_gc}"]`);
            if (targetBoardElement) {
                const isTargetBoardWon = currentGameState.global_board_winners[target_gr][target_gc] !== 'NONE';
                let isTargetBoardFull = false;
                if (!isTargetBoardWon) isTargetBoardFull = currentGameState.local_boards_state[target_gr][target_gc].every(r => r.every(cs => cs !== '.'));

                if (!isTargetBoardWon && !isTargetBoardFull) {
                    targetBoardElement.classList.add(opponent.symbol === 'X' ? 'preview-forced-for-x' : 'preview-forced-for-o');
                }
            }
        }
        function handleCellMouseOut(event) { /* No change from original, keep as is */
             if (!currentGameState) return;
             const cell = event.target.closest('.cell');
             if (cell) {
                 const allBoardContainers = globalGridDisplay.querySelectorAll('.local-board-container');
                 allBoardContainers.forEach(board => board.classList.remove('preview-forced-for-x', 'preview-forced-for-o'));
             }
        }

        function checkUrlForJoin() {
            const urlParams = new URLSearchParams(window.location.search);
            const gameIdToJoinFromLink = urlParams.get('join'); // Changed from 'join_game_id'

            if (gameIdToJoinFromLink) {
                joinGameIdInput.value = gameIdToJoinFromLink; // Put ID in input for transparency
                if (window.TB?.ui?.Toast) TB.ui.Toast.showInfo(`Attempting to join game ${gameIdToJoinFromLink} from link...`, { duration: 2500 });
                joinOnlineGame(); // This function will use joinGameIdInput.value
                // Clean the URL query parameters
                window.history.replaceState({}, document.title, window.location.pathname + window.location.hash);
            }
        }


        function showScreen(screenName) {
            console.log("Showing screen:", screenName);
            ['gameSetup', 'onlineWait', 'game'].forEach(name => {
                document.getElementById(name + 'Section')?.classList.add('hidden');
            });
            const targetScreen = document.getElementById(screenName + 'Section');
            if (targetScreen) targetScreen.classList.remove('hidden');

            if(saveAndLeaveBtn) saveAndLeaveBtn.classList.add('hidden');
            // Show/hide P2 local options div based on screen
            if(player2LocalOptionsDiv) player2LocalOptionsDiv.classList.toggle('hidden', screenName !== 'gameSetup');


            if (screenName === 'gameSetup') {
                if(statsSection) statsSection.classList.remove('hidden');
                togglePlayer2Setup(); // Ensure P2 local options are correctly shown/hidden based on select
                disconnectFromGameStream();
                currentGameId = null; currentGameState = null; clientPlayerInfo = null;
                if(currentPlayerIndicatorContainer) currentPlayerIndicatorContainer.classList.add('hidden');
                updateResumeButtonVisibility();
            } else {
                if(statsSection) statsSection.classList.add('hidden');
                if (screenName === 'game') {
                    if(currentPlayerIndicatorContainer) currentPlayerIndicatorContainer.classList.remove('hidden');
                    if (currentGameState && currentGameState.mode === 'local' && currentGameState.status === 'in_progress' && saveAndLeaveBtn) {
                         saveAndLeaveBtn.classList.remove('hidden');
                    }
                } else if (screenName === 'onlineWait') {
                    if(currentPlayerIndicatorContainer) currentPlayerIndicatorContainer.classList.remove('hidden');
                }
            }
        }


        async function apiRequest(endpoint, payload = null, method = 'GET', queryParams = {}) {
            let url = `/api/${API_MODULE_NAME}/${endpoint}`;
            if (method === 'GET' && payload) { queryParams = {...queryParams, ...payload}; payload = null; }
            if ((endpoint.startsWith('get_game_state') || endpoint.startsWith('make_move')) && queryParams.game_id) {
                 url = `/api/${API_MODULE_NAME}/${endpoint.split('/')[0]}?game_id=${queryParams.game_id}`;
                 delete queryParams.game_id;
            }

            if (!window.TB?.api?.request) {
                showModal("API Error", "Framework error: Cannot communicate with server.", null, "OK", "");
                return { error: true, message: "API_UNAVAILABLE" };
            }
            if(window.TB?.ui?.Loader) TB.ui.Loader.show({text: "Processing...", hideMainContent:false, playAnimation: "Y2+41:R2+61", fullscreen:false});
            try {
                const response = await window.TB.api.request(API_MODULE_NAME, endpoint, payload, method, {queryParams});
                if(window.TB?.ui?.Loader) TB.ui.Loader.hide();
                if (response.error !== window.TB.ToolBoxError.none) {
                    const errorMsg = response.info?.help_text || response.data?.message || `API Error (${response.error})`;
                    if(window.TB?.ui?.Toast) TB.ui.Toast.showError(errorMsg.substring(0,150), {duration: 4000});
                    return { error: true, message: errorMsg, data: response.get() };
                }
                return { error: false, data: response.get() };
            } catch (err) {
                if(window.TB?.ui?.Loader) TB.ui.Loader.hide();
                if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Network or application error.", {duration: 4000});
                return { error: true, message: "NETWORK_ERROR" };
            }
        }

         async function createNewGame(mode, resumeIfAvailable = false) {
            const size = parseInt(gridSizeSelect.value);

            if (mode === 'local' && resumeIfAvailable) {
                const existingGame = loadLocalGame(size);
                if (existingGame && existingGame.status === 'in_progress') {
                    clientPlayerInfo = null;
                    processGameStateUpdate(existingGame);
                    showScreen('game');
                    if(saveAndLeaveBtn) saveAndLeaveBtn.classList.remove('hidden');
                    return;
                }
            }

            const config = { grid_size: size };
            const p1Name = player1NameInput.value.trim() || (mode === 'local' ? "Player X" : "Me");
            const payload = { config, mode, player1_name: p1Name };

            if (mode === 'local') {
                deleteLocalGame(size);
                payload.player2_type = player2TypeSelect.value; // Default is 'npc' in HTML and API if not sent
                if (payload.player2_type === 'npc') {
                    payload.npc_difficulty = npcDifficultySelect.value;
                } else {
                    payload.player2_name = player2NameInput.value.trim() || "Player O";
                }
            }
            // P2 local options div (player2LocalOptionsDiv) will be shown/hidden by showScreen('gameSetup')

            const response = await apiRequest('create_game', payload, 'POST', {hideMainContentWhileLoading: true});
            if (!response.error && response.data?.game_id) {
                if (mode === 'local') {
                    clientPlayerInfo = null;
                    processGameStateUpdate(response.data);
                    showScreen('game');
                    if(saveAndLeaveBtn) saveAndLeaveBtn.classList.remove('hidden');
                    updateResumeButtonVisibility();
                } else if (mode === 'online') {
                    clientPlayerInfo = response.data.players.find(p => p.id === currentSessionId);
                    if (!clientPlayerInfo && response.data.players.length > 0) clientPlayerInfo = response.data.players[0];
                    console.log(response.data.join_url)
                    if (response.data.join_url) {
                    navigator.clipboard.writeText(response.data.join_url).then(() => {
                           TB.ui.Toast.showInfo('link copied to clipboard');
                          }).catch(err => {
                            console.error('Error copying text: ', err);
                          });
                        gameIdShareEl.textContent = response.data.join_url;
                        gameIdShareEl.title = "Click to copy Join Link";
                        //clientPlayerInfo = response.data.join_url;
                    } else {
                        gameIdShareEl.textContent = response.data.game_id;
                        gameIdShareEl.title = "Click to copy Game ID";
                        //clientPlayerInfo = response.data.game_id;
                    }
                    waitingStatusEl.textContent = `Waiting for opponent...`;
                    showScreen('onlineWait');
                    processGameStateUpdate(response.data);
                }
            }
        }


        async function joinOnlineGame() {
            let gameIdToJoin = joinGameIdInput.value.trim();
            const playerName = player1NameInput.value.trim() || "Challenger";

            // Check if full URL is pasted, extract game ID if so
            try {
                const url = new URL(gameIdToJoin);
                if (url.searchParams.has('join')) {
                    gameIdToJoin = url.searchParams.get('join');
                    joinGameIdInput.value = gameIdToJoin; // Update input to show just the ID
                }
            } catch (e) { /* Not a valid URL, assume it's an ID */ }


            if (!gameIdToJoin) {
                if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Please enter a Game ID or Link."); return;
            }

            const response = await apiRequest('join_game', { game_id: gameIdToJoin, player_name: playerName }, 'POST', {hideMainContentWhileLoading: true});

            if (!response.error && response.data?.game_id) {
                 clientPlayerInfo = response.data.players.find(p => p.id === currentSessionId);
                 if (!clientPlayerInfo) {
                     if (response.data.players.length === 2) {
                          const playerO = response.data.players.find(p => p.symbol === 'O');
                          if (playerO) clientPlayerInfo = playerO; // Best guess for P2 joining
                     }
                     console.warn("Online game joined: ClientPlayerInfo might not be perfectly matched.", clientPlayerInfo);
                 }
                 processGameStateUpdate(response.data);

                 if (response.data.status === 'in_progress' || (response.data.status === 'ABORTED' && response.data.player_who_paused)) {
                      showScreen('game');
                 } else if (response.data.status === 'waiting_for_opponent'){
                     if (response.data.join_url) {
                        gameIdShareEl.textContent = response.data.join_url;
                        gameIdShareEl.title = "Click to copy Join Link";
                    } else {
                        gameIdShareEl.textContent = response.data.game_id;
                        gameIdShareEl.title = "Click to copy Game ID";
                    }
                     waitingStatusEl.textContent = `Waiting for opponent...`;
                     showScreen('onlineWait');
                 } else {
                      if(window.TB?.ui?.Toast) TB.ui.Toast.showError(response.data.last_error_message || "Could not join game.");
                      showScreen('gameSetup');
                 }
             } else if (response.data?.message && window.TB?.ui?.Toast) { // API returned an error with a message
                TB.ui.Toast.showError(response.data.message);
             }
        }


         function processGameStateUpdate(newGameState) {
             console.log("PROCESS_GAME_STATE_UPDATE - Received:", newGameState);
             let previousPlayerConnectedStates = {};
             let oldGameStatus = null, oldCurrentPlayerId = null;

             if (currentGameState) {
                 oldGameStatus = currentGameState.status;
                 oldCurrentPlayerId = currentGameState.current_player_id;
                 if (currentGameState.players && clientPlayerInfo) {
                     currentGameState.players.forEach(p => {
                         if (p.id !== clientPlayerInfo.id) previousPlayerConnectedStates[p.id] = p.is_connected;
                     });
                 }
             }
             currentGameState = newGameState;
             if (!currentGameState || !currentGameState.game_id) {
                 if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Error: Corrupted game update.");
                 disconnectFromGameStream(); showScreen('gameSetup'); return;
             }
             currentGameId = newGameState.game_id;

             if (currentGameState.mode === 'online' && (!clientPlayerInfo || !currentGameState.players.find(p => p.id === clientPlayerInfo.id))) {
                  clientPlayerInfo = currentGameState.players.find(p => p.id === currentSessionId);
             }

             if (currentGameState.mode === 'online' && clientPlayerInfo && currentGameState.players) {
                 currentGameState.players.forEach(opponent => {
                     if (opponent.id !== clientPlayerInfo.id) {
                         const wasConnected = previousPlayerConnectedStates[opponent.id];
                         const isConnected = opponent.is_connected;
                         if (wasConnected === true && isConnected === false && window.TB?.ui?.Toast) {
                             TB.ui.Toast.showWarning(`${opponent.name} disconnected. Waiting...`, {duration: 3500});
                         } else if (wasConnected === false && isConnected === true && previousPlayerConnectedStates.hasOwnProperty(opponent.id) && window.TB?.ui?.Toast) {
                             TB.ui.Toast.showSuccess(`${opponent.name} reconnected! Game resumes.`, {duration: 3000});
                         }
                     }
                 });
             }

            // Update gameIdShareEl in onlineWaitSection if it's active
            const onlineWaitScreenActive = !document.getElementById('onlineWaitSection').classList.contains('hidden');
            if (onlineWaitScreenActive && currentGameState.mode === 'online') {
                if (newGameState.join_url) {
                    gameIdShareEl.textContent = newGameState.join_url;
                    gameIdShareEl.title = "Click to copy Join Link";
                } else {
                    gameIdShareEl.textContent = newGameState.game_id;
                    gameIdShareEl.title = "Click to copy Game ID";
                }
            }


             if (currentGameState.mode === 'local') {
                 const currentPlayer = currentGameState.players.find(p => p.id === currentGameState.current_player_id);
                 localPlayerActiveSymbol = currentPlayer ? currentPlayer.symbol : '?';
                 disconnectFromGameStream();
                 if (saveAndLeaveBtn) saveAndLeaveBtn.classList.toggle('hidden', currentGameState.status !== 'in_progress');
             } else if (currentGameState.mode === 'online') {
                 if (onlineWaitScreenActive && currentGameState.status === 'in_progress') {
                     if(window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("Opponent connected! Game starting.", {duration: 2000});
                     showScreen('game');
                 }
                 const isMyTurnOnline = clientPlayerInfo && currentGameState.current_player_id === clientPlayerInfo.id;
                 if (currentGameState.status === 'in_progress') {
                     if (isMyTurnOnline) {
                         disconnectFromGameStream();
                         if (oldGameStatus === 'in_progress' && oldCurrentPlayerId !== currentGameState.current_player_id && window.TB?.ui?.Toast) TB.ui.Toast.showInfo("It's your turn!", {duration: 2000});
                         else if (oldGameStatus === 'ABORTED' && currentGameState.status === 'in_progress' && window.TB?.ui?.Toast) TB.ui.Toast.showInfo("Game resumed. It's your turn!", {duration: 2000});
                     } else connectToGameStream(currentGameState.game_id);
                 } else if (currentGameState.status === 'waiting_for_opponent') {
                     if (clientPlayerInfo && currentGameState.players.length > 0 && currentGameState.players[0].id === clientPlayerInfo.id) connectToGameStream(currentGameState.game_id);
                     else disconnectFromGameStream();
                 } else if (currentGameState.status === 'ABORTED' && currentGameState.player_who_paused) {
                     if (clientPlayerInfo && clientPlayerInfo.id !== currentGameState.player_who_paused) connectToGameStream(currentGameState.game_id);
                     else disconnectFromGameStream();
                 } else if (currentGameState.status === 'finished' || (currentGameState.status === 'ABORTED' && !currentGameState.player_who_paused)) {
                     disconnectFromGameStream();
                 }
                 if (saveAndLeaveBtn) saveAndLeaveBtn.classList.add('hidden');
             }

             renderBoard();
             updateStatusBar();

             if (currentGameState.status === 'finished' || (currentGameState.status === 'ABORTED' && !currentGameState.player_who_paused)) {
                 if (saveAndLeaveBtn) saveAndLeaveBtn.classList.add('hidden');
                 if (currentGameState.mode === 'local' && currentGameState.status === 'finished') deleteLocalGame(currentGameState.config.grid_size);
                 disconnectFromGameStream();
                 if (currentGameState.status === 'finished') { showGameOverModal(); loadSessionStats(); }
                 else showModal("Game Aborted", currentGameState.last_error_message || "The game was aborted.", () => showScreen('gameSetup'));
             }
             if (resetGameBtn) {
                 if (currentGameState.mode === 'online' && (currentGameState.status === 'in_progress' || (currentGameState.status === 'ABORTED' && currentGameState.player_who_paused))) {
                     resetGameBtn.textContent = 'Leave Game'; resetGameBtn.classList.add('danger'); resetGameBtn.classList.remove('secondary');
                 } else if (currentGameState.mode === 'local') {
                     resetGameBtn.textContent = 'Reset Game'; resetGameBtn.classList.add('danger'); resetGameBtn.classList.remove('secondary');
                 } else {
                     resetGameBtn.textContent = 'Back to Menu'; resetGameBtn.classList.remove('danger'); resetGameBtn.classList.add('secondary');
                 }
             }
         }


        async function makePlayerMove(globalR, globalC, localR, localC) {
            if (!currentGameState || !currentGameId || currentGameState.status !== 'in_progress') return;
            let playerIdForMove;
            const currentPlayerOnClient = currentGameState.players.find(p => p.id === currentGameState.current_player_id);

            if (currentGameState.mode === 'local') {
                playerIdForMove = currentGameState.current_player_id;
                if (currentPlayerOnClient && currentPlayerOnClient.is_npc) return; // Should be server-handled
            } else {
                if (!clientPlayerInfo || currentGameState.current_player_id !== clientPlayerInfo.id) {
                    if(window.TB?.ui?.Toast) TB.ui.Toast.showInfo("Not your turn."); return;
                }
                playerIdForMove = clientPlayerInfo.id;
            }
            const movePayload = { player_id: playerIdForMove, global_row: globalR, global_col: globalC, local_row: localR, local_col: localC, game_id: currentGameId };
            if (globalGridDisplay) globalGridDisplay.style.pointerEvents = 'none';
            const response = await apiRequest(`make_move`, movePayload, 'POST');
            if (globalGridDisplay) globalGridDisplay.style.pointerEvents = 'auto';
            if (!response.error && response.data) processGameStateUpdate(response.data);
            else if (response.data?.game_id) processGameStateUpdate(response.data);
        }

        function confirmResetGame() {
             if (!currentGameState) return;
             if (currentGameState.mode === 'online' && (currentGameState.status === 'in_progress' || (currentGameState.status === 'ABORTED' && currentGameState.player_who_paused))) {
                 showModal('Leave Game?', 'Are you sure you want to leave this online game?', () => {
                     disconnectFromGameStream(); showScreen('gameSetup');
                 });
             } else if (currentGameState.mode === 'local' && (currentGameState.status === 'in_progress' || currentGameState.status === 'finished' || currentGameState.status === 'ABORTED')) {
                 showModal('Reset Game?', 'Start a new local game with current settings?', async () => {
                     deleteLocalGame(currentGameState.config.grid_size);
                     await createNewGame('local');
                 });
             } else if (currentGameState.status === 'finished' || (currentGameState.status === 'ABORTED' && !currentGameState.player_who_paused) ) {
                 showModal('New Game?', 'Return to the menu to start a new game?', () => showScreen('gameSetup'));
             } else if(window.TB?.ui?.Toast) TB.ui.Toast.showInfo("Cannot reset/leave from current game state.");
         }
        function confirmBackToMenu() {
            if (currentGameState && currentGameState.status !== 'finished' && currentGameState.status !== 'aborted') {
                let message = 'Progress will be lost. Are you sure?';
                if (currentGameState.mode === 'local' && currentGameState.status === 'in_progress') message = 'Game not saved. Use "Save & Leave" or progress will be lost. Continue to menu?';
                showModal('Back to Menu?', message, () => showScreen('gameSetup'));
            } else showScreen('gameSetup');
        }

        async function loadSessionStats() {
            const response = await apiRequest('get_session_stats', {session_id: currentSessionId}, 'GET');
            if (!response.error && response.data) updateStatsDisplay(response.data);
            else updateStatsDisplay({ games_played:0, wins:0, losses:0, draws:0 });
        }
        function updateStatsDisplay(stats) {
            statsGamesPlayedEl.textContent = stats.games_played ?? 0;
            statsWinsEl.textContent = stats.wins ?? 0;
            statsLossesEl.textContent = stats.losses ?? 0;
            statsDrawsEl.textContent = stats.draws ?? 0;
        }

        function renderBoard() {
            if (!currentGameState || !globalGridDisplay) return;
            const N = currentGameState.config?.grid_size;
            dynamicallySetGridStyles(N);
            globalGridDisplay.innerHTML = '';
            globalGridDisplay.querySelectorAll('.local-board-container').forEach(b => b.classList.remove('preview-forced-for-x', 'preview-forced-for-o'));
            const lastMoveCoords = currentGameState.last_made_move_coords;

            for (let gr = 0; gr < N; gr++) {
                for (let gc = 0; gc < N; gc++) {
                    const localBoardContainer = document.createElement('div');
                    localBoardContainer.className = 'local-board-container';
                    localBoardContainer.dataset.gr = gr; localBoardContainer.dataset.gc = gc;
                    const localWinner = currentGameState.global_board_winners[gr][gc];
                    if (localWinner !== 'NONE') {
                        localBoardContainer.classList.add('won-' + localWinner);
                        const overlay = document.createElement('div');
                        overlay.className = 'winner-overlay player-' + (localWinner === 'DRAW' ? 'draw' : localWinner);
                        overlay.textContent = localWinner === 'DRAW' ? 'D' : localWinner;
                        localBoardContainer.appendChild(overlay);
                    }
                    let isThisBoardTheActiveTarget = false;
                    if (currentGameState.status === 'in_progress' && localWinner === 'NONE') {
                        const forcedTarget = currentGameState.next_forced_global_board;
                        if (forcedTarget) {
                            if (forcedTarget[0] === gr && forcedTarget[1] === gc) { localBoardContainer.classList.add('forced-target'); isThisBoardTheActiveTarget = true; }
                            else localBoardContainer.classList.add('inactive-target');
                        } else { localBoardContainer.classList.add('playable-anywhere'); isThisBoardTheActiveTarget = true; }
                    } else localBoardContainer.classList.add('inactive-target');

                    const localGrid = document.createElement('div');
                    localGrid.className = 'local-grid';
                    const localCells = currentGameState.local_boards_state[gr][gc];
                    for (let lr = 0; lr < N; lr++) {
                        for (let lc = 0; lc < N; lc++) {
                            const cell = document.createElement('div');
                            cell.className = 'cell';
                            cell.dataset.gr = gr; cell.dataset.gc = gc; cell.dataset.lr = lr; cell.dataset.lc = lc;
                            const cellState = localCells[lr][lc];
                            if (cellState !== '.') { cell.textContent = cellState; cell.classList.add('player-' + cellState); }
                            if (lastMoveCoords && gr === lastMoveCoords[0] && gc === lastMoveCoords[1] && lr === lastMoveCoords[2] && lc === lastMoveCoords[3]) cell.classList.add('last-move');
                            let isCellCurrentlyPlayable = false;
                            if (currentGameState.status === 'in_progress' && localWinner === 'NONE' && cellState === '.') {
                                let isThisClientsTurnToAct = (currentGameState.mode === 'local') || (clientPlayerInfo && currentGameState.current_player_id === clientPlayerInfo.id);
                                if (isThisClientsTurnToAct && isThisBoardTheActiveTarget) isCellCurrentlyPlayable = true;
                            }
                            if (isCellCurrentlyPlayable) cell.classList.add('playable');
                            localGrid.appendChild(cell);
                        }
                    }
                    localBoardContainer.appendChild(localGrid);
                    globalGridDisplay.appendChild(localBoardContainer);
                }
            }
        }

     function updateStatusBar() {
             if (!statusBar || !currentGameState || !currentPlayerIndicator) return;
             let message = ""; let msgType = "info";
             currentPlayerIndicator.className = 'current-player-indicator';

             if (currentGameState.status === 'waiting_for_opponent') {
                 message = "Waiting for opponent to join...";
                 if (currentGameState.players.length > 0 && currentGameState.players[0].is_connected) {
                     currentPlayerIndicator.classList.add(`player-${currentGameState.players[0].symbol}`);
                 }
             } else if (currentGameState.status === 'in_progress') {
                 const currentPlayer = currentGameState.players.find(p => p.id === currentGameState.current_player_id);
                 const pName = currentPlayer ? currentPlayer.name : "Player", pSymbol = currentPlayer ? currentPlayer.symbol : "?";
                 if (currentPlayer) currentPlayerIndicator.classList.add(`player-${pSymbol}`);

                 if (currentGameState.mode === 'local') message = `${pName} (${pSymbol})'s Turn.`;
                 else {
                     const amICurrentPlayer = clientPlayerInfo && clientPlayerInfo.id === currentGameState.current_player_id;
                     message = amICurrentPlayer ? `Your Turn (${clientPlayerInfo.symbol})` : `Waiting for ${pName} (${pSymbol})...`;
                     const opponent = currentGameState.players.find(p => p.id !== currentGameState.current_player_id);
                     if (opponent && !opponent.is_connected) {
                         message = `Waiting for ${opponent.name} (${opponent.symbol}) to reconnect...`; msgType = "warning";
                         if (currentPlayer && currentPlayer.is_connected) currentPlayerIndicator.className = `current-player-indicator player-${currentPlayer.symbol}`;
                         else if (clientPlayerInfo && clientPlayerInfo.is_connected) currentPlayerIndicator.className = `current-player-indicator player-${clientPlayerInfo.symbol}`;
                     }
                 }
                 if (currentGameState.next_forced_global_board) {
                     const [gr, gc] = currentGameState.next_forced_global_board;
                     if (currentGameState.mode === 'local' || (clientPlayerInfo && clientPlayerInfo.id === currentGameState.current_player_id)) message += ` Play in board (${gr+1},${gc+1}).`;
                 } else if (currentGameState.status === 'in_progress' && (currentGameState.mode === 'local' || (clientPlayerInfo && clientPlayerInfo.id === currentGameState.current_player_id))) {
                     message += " Play in any valid highlighted board.";
                 }
             } else if (currentGameState.status === 'ABORTED' && currentGameState.player_who_paused) {
                 msgType = "warning";
                 const disconnectedPlayerInfo = currentGameState.players.find(p => p.id === currentGameState.player_who_paused);
                 message = `Player ${disconnectedPlayerInfo ? disconnectedPlayerInfo.name : "Opponent"} disconnected. Waiting...`;
                 const waitingPlayer = currentGameState.players.find(p => p.id !== currentGameState.player_who_paused && p.is_connected);
                 if (waitingPlayer) currentPlayerIndicator.classList.add(`player-${waitingPlayer.symbol}`);
                 else if (clientPlayerInfo && clientPlayerInfo.id !== currentGameState.player_who_paused) currentPlayerIndicator.classList.add(`player-${clientPlayerInfo.symbol}`);
             } else if (currentGameState.status === 'finished') {
                 msgType = "success";
                 if (currentGameState.is_draw) message = "Game Over: It's a DRAW!";
                 else {
                     const winner = currentGameState.players.find(p => p.symbol === currentGameState.overall_winner_symbol);
                     message = `Game Over: ${winner ? winner.name : 'Player'} (${currentGameState.overall_winner_symbol}) WINS!`;
                     if (winner) currentPlayerIndicator.classList.add(`player-${winner.symbol}`);
                 }
             } else if (currentGameState.status === 'ABORTED') {
                  message = currentGameState.last_error_message || "Game Aborted."; msgType = "error";
             }
             if (currentGameState.last_error_message && (msgType === "info" || (currentGameState.status === 'in_progress' && !message.includes("Error")))) {
                 if (!currentGameState.last_error_message.toLowerCase().includes("resumed") && !currentGameState.last_error_message.toLowerCase().includes("reconnected")) {
                     if(!(currentGameState.status === 'ABORTED' && currentGameState.player_who_paused) && !(currentGameState.status === 'IN_PROGRESS' && message.includes("reconnect"))){
                         message = `Note: ${currentGameState.last_error_message}`;
                         if (currentGameState.last_error_message.toLowerCase().match(/must play|not your turn|invalid|occupied/)) msgType = "error"; else msgType = "info";
                     }
                 }
             }
             statusBar.textContent = message;
             statusBar.className = `status-bar ${msgType}`;
         }

        function showGameOverModal() {
            let title = "Game Over!", content = "";
            if (currentGameState.is_draw) content = "The game ended in a DRAW!";
            else {
                const winner = currentGameState.players.find(p => p.symbol === currentGameState.overall_winner_symbol);
                content = `${winner ? winner.name : 'Player'} (${currentGameState.overall_winner_symbol}) is victorious!`;
            }
            showModal(title, content, () => createNewGame(currentGameState.mode), "Play Again", "Menu");
        }
        function copyGameIdToClipboard() {
            const textToCopy = gameIdShareEl.textContent;
            if (navigator.clipboard && textToCopy) {
                navigator.clipboard.writeText(textToCopy)
                    .then(() => { if(window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("Copied!", {duration:1500}); })
                    .catch(err => { if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Copy failed."); });
            } else if(window.TB?.ui?.Toast) TB.ui.Toast.showWarning("Nothing to copy.");
        }
        function showModal(title, message, onConfirm = null, confirmText = "OK", cancelText = "Cancel") {
            modalTitle.textContent = title; modalMessage.textContent = message;
            modalConfirmBtn.textContent = confirmText; modalCancelBtn.textContent = cancelText;
            modalConfirmCallback = onConfirm; modalOverlay.classList.remove('hidden');
            modalConfirmBtn.classList.toggle('hidden', !onConfirm);
        }
        function hideModal() { modalOverlay.classList.add('hidden'); modalConfirmCallback = null; }

        function dynamicallySetGridStyles(N) {
            if (!globalGridDisplay) return;
            const mainWrap = document.querySelector('.main-content-wrapper');
            const availableWidth = mainWrap ? mainWrap.offsetWidth - 20 : window.innerWidth - 40;
            let boardPixelSize = Math.min(availableWidth, window.innerHeight * 0.65, N * 100 + (N-1)*5);
            boardPixelSize = Math.max(N * 45 + (N-1)*2, boardPixelSize);
            boardPixelSize = Math.min(boardPixelSize, 650);
            globalGridDisplay.style.width = `${boardPixelSize}px`; globalGridDisplay.style.height = `${boardPixelSize}px`;
            const globalGap = Math.max(2, Math.floor(boardPixelSize / (N * 30)));
            globalGridDisplay.style.gap = `${globalGap}px`; globalGridDisplay.style.padding = `${globalGap}px`;
            const localBoardOuterSize = (boardPixelSize - (N - 1) * globalGap - 2 * globalGap) / N;
            const localBoardInnerSize = Math.max(10, localBoardOuterSize - 4);
            const localCellGap = Math.max(1, Math.floor(localBoardInnerSize / (N * 40)));
            const estimatedCellSize = (localBoardInnerSize - (N - 1) * localCellGap) / N;
            const cellFontSize = Math.max(8, estimatedCellSize * 0.42 / Math.sqrt(N/2.5) );
            const winnerOverlayFontSize = Math.max(15, localBoardInnerSize * 0.55 / Math.sqrt(N/2.5) );
            const winnerOverlayDrawFontSize = Math.max(12, localBoardInnerSize * 0.35 / Math.sqrt(N/2.5) );
            let dynamicStyleSheet = document.getElementById('dynamicGameStylesUTTT');
            if (!dynamicStyleSheet) { dynamicStyleSheet = document.createElement('style'); dynamicStyleSheet.id = 'dynamicGameStylesUTTT'; document.head.appendChild(dynamicStyleSheet); }
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

# --- END OF MODIFIED FUNCTION: ultimate_ttt_ui_page ---
