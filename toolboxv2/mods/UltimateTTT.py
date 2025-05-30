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
    npc_difficulty: Optional[NPCDifficulty] = None


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

    last_made_move_coords: Optional[Tuple[int, int, int, int]] = None
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

    def _is_local_board_full(self, local_board_cells: List[List[CellState]], cell_type=CellState.EMPTY) -> bool:
        """Checks if a specific local board (passed as a 2D list of CellState) is full."""
        for r in range(self.size):
            for c in range(self.size):
                if local_board_cells[r][c] == cell_type:
                    return False
        return True

    def add_player(self, player_id: str, player_name: str,
                   is_npc: bool = False, npc_difficulty: Optional[NPCDifficulty] = None) -> bool:
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
                self.gs.updated_at = datetime.now(timezone.utc)

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
                self.gs.waiting_since = datetime.now(timezone.utc)
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
            self.gs.updated_at = datetime.now(timezone.utc)
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
            # If game was paused by them and they refresh, it might still be ABORTED.
            # Let's ensure status consistency if they were the pauser.
            if self.gs.status == GameStatus.ABORTED and self.gs.player_who_paused == player_id:
                opponent = self.gs.get_opponent_info(player_id)
                if opponent and opponent.is_connected:
                    self.gs.status = GameStatus.IN_PROGRESS
                    self.gs.last_error_message = f"Connection for {player.name} re-established. Game resumed."
                    self.gs.player_who_paused = None
                    self.gs.updated_at = datetime.now(timezone.utc)
                    app.logger.info(
                        f"Game {self.gs.game_id} resumed as already-connected pauser {player.name} re-interacted.")
                else:  # Opponent still not there
                    self.gs.last_error_message = f"Welcome back, {player.name}! Your opponent is still not connected."
                    # Status remains ABORTED, player_who_paused remains player_id
            return True  # Treat as successful if already connected

        # Player is in game, but marked as not connected. This is the main reconnect path.
        player.is_connected = True
        self.gs.updated_at = datetime.now(timezone.utc)
        app.logger.info(
            f"Player {player.name} ({player_id}) reconnected to game {self.gs.game_id}. Previous status: {self.gs.status}, Paused by: {self.gs.player_who_paused}")

        if self.gs.status == GameStatus.ABORTED:
            if self.gs.player_who_paused == player_id:  # The player who caused the pause has reconnected
                opponent = self.gs.get_opponent_info(player_id)
                if opponent and opponent.is_connected:
                    self.gs.status = GameStatus.IN_PROGRESS
                    self.gs.last_error_message = f"Player {player.name} reconnected. Game resumed!"
                    self.gs.player_who_paused = None
                    # current_player_id is NOT changed here. It remains whoever's turn it was.
                    app.logger.info(
                        f"Game {self.gs.game_id} RESUMED. Pauser {player.name} reconnected, opponent {opponent.name} is present.")
                else:
                    # Pauser reconnected, but opponent is (still) gone. Game stays paused, waiting for opponent now.
                    self.gs.last_error_message = f"Welcome back, {player.name}! Your opponent is not connected. Game remains paused."
                    # self.gs.player_who_paused should now be the ID of the *other* player if known, or null if game is truly stuck
                    # For now, let's keep it simple: game is ABORTED, this player is connected. If opponent also joins, it will resume.
                    # If opponent was never there (e.g. P1 created, P1 disconnected, P1 reconnected before P2 joined)
                    if not opponent:  # This implies it was a 1-player game waiting for P2, and P1 reconnected
                        self.gs.status = GameStatus.WAITING_FOR_OPPONENT
                        self.gs.player_who_paused = None
                        self.gs.current_player_id = player_id  # P1 is the active one again
                        self.gs.last_error_message = f"Creator {player.name} reconnected. Waiting for opponent."
                    else:  # Opponent was there but is now disconnected
                        self.gs.player_who_paused = opponent.id  # Now waiting for the other person
                        app.logger.info(
                            f"Game {self.gs.game_id} still PAUSED. {player.name} reconnected, but opponent {opponent.name} is NOT. Waiting for {opponent.name}.")


            elif self.gs.player_who_paused and self.gs.player_who_paused != player_id:
                # The *other* player (not the initial pauser) reconnected, while game was paused for initial pauser.
                # This means both players are now connected. Initial pauser is still `player_who_paused`.
                initial_pauser_info = self.gs.get_player_info(self.gs.player_who_paused)
                if initial_pauser_info and initial_pauser_info.is_connected:  # Should not happen if state is consistent
                    self.gs.status = GameStatus.IN_PROGRESS
                    self.gs.last_error_message = f"Both players are now connected. Game resumed!"
                    self.gs.player_who_paused = None
                    app.logger.info(
                        f"Game {self.gs.game_id} RESUMED. Waiting player {player.name} reconnected, initial pauser {initial_pauser_info.name} also (somehow) connected.")
                else:
                    # This player reconnected, but we are still waiting for the original pauser.
                    self.gs.last_error_message = f"Welcome back, {player.name}! Still waiting for {initial_pauser_info.name if initial_pauser_info else 'the other player'} to reconnect."
                    app.logger.info(
                        f"Game {self.gs.game_id} still PAUSED. Player {player.name} reconnected, but still waiting for original pauser {self.gs.player_who_paused}.")
            else:  # game is ABORTED but no specific player_who_paused (hard abort)
                self.gs.last_error_message = f"Player {player.name} reconnected, but the game was fully aborted."
                # Cannot resume a hard abort. Stays ABORTED.
                app.logger.info(
                    f"Game {self.gs.game_id} is HARD ABORTED. Player {player.name} reconnected, but game cannot resume.")

        elif self.gs.status == GameStatus.IN_PROGRESS:
            # Player reconnected to an already IN_PROGRESS game (e.g. refresh, brief network blip)
            # Check if opponent is still there.
            opponent = self.gs.get_opponent_info(player_id)
            if not opponent or not opponent.is_connected:
                # Opponent disconnected while this player was briefly gone. Game should pause.
                self.gs.status = GameStatus.ABORTED
                self.gs.player_who_paused = opponent.id if opponent else None  # Pause for the opponent
                self.gs.last_error_message = f"Welcome back, {player.name}! Your opponent disconnected while you were away. Waiting for them."
                app.logger.info(
                    f"Game {self.gs.game_id} transitions to PAUSED. {player.name} reconnected to IN_PROGRESS, but opponent {opponent.id if opponent else 'N/A'} is gone.")
            else:
                self.gs.last_error_message = f"Player {player.name} re-established connection during active game."
                app.logger.info(
                    f"Player {player.name} ({player_id}) re-established connection to IN_PROGRESS game {self.gs.game_id}.")

        elif self.gs.status == GameStatus.WAITING_FOR_OPPONENT:
            # This is likely P1 (creator) who disconnected and reconnected before P2 joined.
            if self.gs.players[0].id == player_id:  # Ensure it's P1
                self.gs.last_error_message = f"Creator {player.name} reconnected. Still waiting for opponent."
                self.gs.current_player_id = player_id  # P1 is active again
                app.logger.info(
                    f"Creator {player.name} ({player_id}) reconnected to WAITING_FOR_OPPONENT game {self.gs.game_id}.")
            else:  # Should not happen if game state is consistent (P2 trying to reconnect to WAITING game)
                app.logger.warning(
                    f"Non-creator {player.name} tried to reconnect to WAITING_FOR_OPPONENT game {self.gs.game_id}.")

        return True


# -- NPC Agents ---
import random  # Add this import at the top of your file if not already there


def get_npc_move_easy(game_state: GameState, npc_player_info: PlayerInfo) -> Optional[Move]:
    """
    Easy NPC: Plays "perfect" Tic-Tac-Toe on the current local board it's sent to,
    or a random valid move if it can play anywhere. Ignores global strategy.
    """
    gs = game_state
    size = gs.config.grid_size
    npc_symbol = npc_player_info.symbol

    possible_moves = []

    forced_gr, forced_gc = -1, -1
    play_anywhere = False

    if gs.next_forced_global_board:
        forced_gr, forced_gc = gs.next_forced_global_board
        if gs.global_board_winners[forced_gr][forced_gc] != BoardWinner.NONE or \
            UltimateTTTGameEngine(gs)._is_local_board_full(gs.local_boards_state[forced_gr][forced_gc]):
            play_anywhere = True  # Sent to a finished board
        else:  # Play in the forced board
            for lr in range(size):
                for lc in range(size):
                    if gs.local_boards_state[forced_gr][forced_gc][lr][lc] == CellState.EMPTY:
                        possible_moves.append({'gr': forced_gr, 'gc': forced_gc, 'lr': lr, 'lc': lc})
    else:
        play_anywhere = True  # Can play anywhere

    if play_anywhere:
        possible_moves = []  # Reset, as we now look at all valid boards
        for gr_idx in range(size):
            for gc_idx in range(size):
                if gs.global_board_winners[gr_idx][gc_idx] == BoardWinner.NONE and \
                    not UltimateTTTGameEngine(gs)._is_local_board_full(gs.local_boards_state[gr_idx][gc_idx]):
                    for lr in range(size):
                        for lc in range(size):
                            if gs.local_boards_state[gr_idx][gc_idx][lr][lc] == CellState.EMPTY:
                                possible_moves.append({'gr': gr_idx, 'gc': gc_idx, 'lr': lr, 'lc': lc})

    if not possible_moves:
        gs.last_error_message = "NPC Error: No possible moves found (should not happen in valid game state)."
        return None

    # Easy strategy:
    # 1. If playing in a specific local board (forced_gr, forced_gc is valid):
    #    - Try to win that local board.
    #    - If not, try to block opponent from winning that local board.
    #    - Else, pick a random valid cell in that local board.
    # 2. If play_anywhere:
    #    - Pick a random valid move from `possible_moves`.

    target_board_coords_for_ttt = None
    if not play_anywhere and forced_gr != -1:  # Focused on one board
        target_board_coords_for_ttt = (forced_gr, forced_gc)

    if target_board_coords_for_ttt:
        gr, gc = target_board_coords_for_ttt
        local_board_cells = gs.local_boards_state[gr][gc]

        # Check for winning move in this local board
        for move_coords in possible_moves:
            if move_coords['gr'] == gr and move_coords['gc'] == gc:
                temp_board = [row[:] for row in local_board_cells]
                temp_board[move_coords['lr']][move_coords['lc']] = CellState(npc_symbol.value)
                if UltimateTTTGameEngine(gs)._get_board_winner_symbol(temp_board, CellState) == CellState(
                    npc_symbol.value):
                    return Move(player_id=npc_player_info.id, global_row=gr, global_col=gc, local_row=move_coords['lr'],
                                local_col=move_coords['lc'])

        # Check for blocking move in this local board
        opponent_symbol = PlayerSymbol.O if npc_symbol == PlayerSymbol.X else PlayerSymbol.X
        for move_coords in possible_moves:
            if move_coords['gr'] == gr and move_coords['gc'] == gc:
                temp_board = [row[:] for row in local_board_cells]
                temp_board[move_coords['lr']][move_coords['lc']] = CellState(
                    opponent_symbol.value)  # Simulate opponent's move
                if UltimateTTTGameEngine(gs)._get_board_winner_symbol(temp_board, CellState) == CellState(
                    opponent_symbol.value):
                    # Block here
                    return Move(player_id=npc_player_info.id, global_row=gr, global_col=gc, local_row=move_coords['lr'],
                                local_col=move_coords['lc'])

        # If no win/block, pick a random move within the forced board
        moves_in_forced_board = [m for m in possible_moves if m['gr'] == gr and m['gc'] == gc]
        if moves_in_forced_board:
            chosen_move_coords = random.choice(moves_in_forced_board)
            return Move(player_id=npc_player_info.id, global_row=chosen_move_coords['gr'],
                        global_col=chosen_move_coords['gc'],
                        local_row=chosen_move_coords['lr'], local_col=chosen_move_coords['lc'])

    # If play_anywhere or specific board logic didn't yield a move (shouldn't happen if moves_in_forced_board existed)
    if possible_moves:
        chosen_move_coords = random.choice(possible_moves)
        return Move(player_id=npc_player_info.id, global_row=chosen_move_coords['gr'],
                    global_col=chosen_move_coords['gc'],
                    local_row=chosen_move_coords['lr'], local_col=chosen_move_coords['lc'])

    return None  # Should not be reached if logic is correct


def get_npc_move_medium(game_state: GameState, npc_player_info: PlayerInfo) -> Optional[Move]:
    """
    Medium NPC:
    1. Tries to win the current local board.
    2. Tries to block opponent on current local board.
    3. Global considerations:
        a. If it can win the GLOBAL game with a move, take it.
        b. If it can block opponent from winning GLOBAL game, take it.
        c. Avoid sending opponent to a local board where opponent can win that local board,
           IF there's an alternative move that doesn't lead to an immediate local loss.
        d. Prefer moves that send opponent to an already won/drawn local board (giving NPC free move).
        e. Prefer moves that set up a future local win for NPC.
    4. If none of the above, fallback to Easy's local TTT logic or a "good heuristic" move.
    (This is a simplified placeholder; true medium AI is complex)
    """
    # For now, let's make medium slightly better than easy by picking a center if available on forced board
    # or a random board's center if play_anywhere. This is a very basic heuristic.
    gs = game_state
    size = gs.config.grid_size

    # Fallback to Easy's logic for now until more complex strategy is built
    # This requires careful implementation of evaluating global board states,
    # simulating future moves, etc. which is non-trivial.

    # Basic Heuristic Enhancement for Medium (very simplified):
    # Try to take center of forced local board, or center of a random playable local board.
    # This is a placeholder for more advanced logic.

    possible_moves = []  # Collect all raw possible moves first
    forced_gr, forced_gc = gs.next_forced_global_board if gs.next_forced_global_board else (-1, -1)
    play_anywhere = not gs.next_forced_global_board or \
                    gs.global_board_winners[forced_gr][forced_gc] != BoardWinner.NONE or \
                    UltimateTTTGameEngine(gs)._is_local_board_full(gs.local_boards_state[forced_gr][forced_gc])

    if not play_anywhere:
        for lr in range(size):
            for lc in range(size):
                if gs.local_boards_state[forced_gr][forced_gc][lr][lc] == CellState.EMPTY:
                    possible_moves.append({'gr': forced_gr, 'gc': forced_gc, 'lr': lr, 'lc': lc})
    else:
        for gr_idx in range(size):
            for gc_idx in range(size):
                if gs.global_board_winners[gr_idx][gc_idx] == BoardWinner.NONE and \
                    not UltimateTTTGameEngine(gs)._is_local_board_full(gs.local_boards_state[gr_idx][gc_idx]):
                    for lr in range(size):
                        for lc in range(size):
                            if gs.local_boards_state[gr_idx][gc_idx][lr][lc] == CellState.EMPTY:
                                possible_moves.append({'gr': gr_idx, 'gc': gc_idx, 'lr': lr, 'lc': lc})

    if not possible_moves: return None

    # Try to find a move that wins a local board AND sends opponent to a non-losing board for NPC
    # (This is a sketch of a more advanced thought process)

    # Placeholder: Medium will just use Easy's logic for now due to complexity.
    # You would replace this with a more sophisticated evaluation.
    # For example, iterate through possible_moves, simulate each one, evaluate the resulting game state.
    # Key considerations for medium:
    # - Global win/block checks (most important)
    # - Local win/block checks
    # - "Sending" opponent to a safe/advantageous board.
    # - Avoiding sending opponent to a board where they can win locally.
    # - Setting up two-in-a-rows (locally or globally).

    return get_npc_move_easy(game_state, npc_player_info)  # Fallback for now


def get_npc_move_hard(game_state: GameState, npc_player_info: PlayerInfo) -> Optional[Move]:
    """
    Hard NPC: (Placeholder - very complex)
    - Minimax or Monte Carlo Tree Search (MCTS) on a simplified version or a few plies deep.
    - Stronger global awareness.
    - Forcing sequences ("Zwickmühlen" / Forks).
    - Definitely plays perfect local TTT.
    - Tries to control key global squares (center, corners).
    """
    # Placeholder: Hard will also use Easy's logic for now.
    # Implementing a truly "Hard" UTTT bot is a significant AI challenge.
    return get_npc_move_easy(game_state, npc_player_info)


NPC_DISPATCHER = {
    NPCDifficulty.EASY: get_npc_move_easy,
    NPCDifficulty.MEDIUM: get_npc_move_medium,
    NPCDifficulty.HARD: get_npc_move_hard,
    # NPCDifficulty.INSANE: get_npc_move_insane,
}


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
# --- START OF BLOCK 5 (Modify api_create_game) ---
# FILE: ultimate_ttt_api.py
# Modify the api_create_game endpoint function

@export(mod_name=GAME_NAME, name="create_game", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_create_game(app: App, request: RequestData, data=None):
    try:
        payload = data or {}
        config_data = payload.get("config", {})
        config = GameConfig(**config_data)  # Validate grid_size here

        mode = GameMode(payload.get("mode", "local"))
        player1_name = payload.get("player1_name", "Player 1").strip()

        initial_status = GameStatus.IN_PROGRESS  # Default for local, or online if P2 joins immediately

        if mode == GameMode.ONLINE:
            initial_status = GameStatus.WAITING_FOR_OPPONENT

        game_state = GameState(config=config, mode=mode, status=initial_status)
        engine = UltimateTTTGameEngine(game_state)

        # Add Player 1 (always human for now in create_game context)
        engine.add_player(LOCAL_PLAYER_X_ID if mode == GameMode.LOCAL else (await get_user_from_request(app,
                                                                                                        request)).uid or f"guest_{uuid.uuid4().hex[:6]}",
                          player1_name)

        if mode == GameMode.LOCAL:
            player2_type = payload.get("player2_type", "human")  # "human" or "npc"
            player2_name_human = payload.get("player2_name", "Player 2").strip()

            if player2_type == "npc":
                npc_difficulty_str = payload.get("npc_difficulty", NPCDifficulty.EASY.value)
                npc_difficulty = NPCDifficulty(npc_difficulty_str)
                npc_id = f"{NPC_PLAYER_ID_PREFIX}{npc_difficulty.value}"
                npc_name = f"NPC ({npc_difficulty.value.capitalize()})"
                engine.add_player(npc_id, npc_name, is_npc=True, npc_difficulty=npc_difficulty)
            else:  # Human Player 2
                engine.add_player(LOCAL_PLAYER_O_ID, player2_name_human)

            # For local games, P1 (X) always starts.
            # engine.add_player already sets current_player_id to X when 2 players are in.
            # game_state.status is already IN_PROGRESS by add_player when P2 is added.

        await save_game_to_db_final(app, game_state)
        app.logger.info(
            f"Created {mode.value} game {game_state.game_id} (Size: {config.grid_size}) P1: {player1_name}, P2 setup: {payload.get('player2_type', 'human')}")

        # If P1 is human and P2 is NPC, and it's P1's turn, game state is fine.
        # If P1 is NPC (not supported by this flow yet) and P2 human, would need NPC move.
        # For now, P1 is human starting.

        return Result.json(data=game_state.model_dump_for_api())
    except ValueError as e:
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
     player_name_from_join_attempt = payload.get("player_name", "Player 2")

     if not game_id:
         return Result.default_user_error("Game ID required.", 400)

     game_state = await load_game_from_db_final(app, game_id)
     if not game_state:
         return Result.default_user_error("Game not found.", 404)

     user = await get_user_from_request(app, request)
     joiner_id_from_request = user.uid if user and user.uid else f"guest_{uuid.uuid4().hex[:6]}"

     if game_state.mode != GameMode.ONLINE:
         return Result.default_user_error("Not an online game.", 400)

     engine = UltimateTTTGameEngine(game_state)
     already_in_game_as_player = game_state.get_player_info(joiner_id_from_request)

     # Case 1: Joining a game waiting for an opponent (as P2)
     if game_state.status == GameStatus.WAITING_FOR_OPPONENT:
         app.logger.info(f"Player {joiner_id_from_request} ({player_name_from_join_attempt}) attempting to join WAITING game {game_id}.")
         if already_in_game_as_player: # P1 trying to "rejoin" while game is still waiting (e.g. after refresh)
             if not already_in_game_as_player.is_connected:
                 engine.handle_player_reconnect(joiner_id_from_request)
             # else: P1 is already connected and in WAITING state, no action needed.
         elif len(game_state.players) < 2: # New player (P2) joins
             if not engine.add_player(joiner_id_from_request, player_name_from_join_attempt):
                 return Result.default_user_error(game_state.last_error_message or "Could not join (add player failed).", 400)
         else: # Game is WAITING but somehow full? Should not happen.
              return Result.default_user_error("Game is waiting but seems full. Cannot join.", 409)


     # Case 2: Reconnecting to a "paused" game (ABORTED + player_who_paused)
     elif game_state.status == GameStatus.ABORTED and game_state.player_who_paused:
         app.logger.info(f"Player {joiner_id_from_request} attempting to reconnect to PAUSED game {game_id} (paused by {game_state.player_who_paused}).")
         if already_in_game_as_player and already_in_game_as_player.id == game_state.player_who_paused:
             if not already_in_game_as_player.is_connected: # If they were marked disconnected
                 engine.handle_player_reconnect(joiner_id_from_request)
             # If they were already connected but game was paused (e.g. browser refresh), reconnect logic updates state
         elif already_in_game_as_player and already_in_game_as_player.is_connected:
              return Result.default_user_error("You are already connected. Game is paused waiting for opponent.", 400)
         elif already_in_game_as_player and not already_in_game_as_player.is_connected: # The *other* player trying to reconnect to a game paused by their opponent
             engine.handle_player_reconnect(joiner_id_from_request) # This will update their connected state
         else: # A new player trying to join a game paused for a specific player
             return Result.default_user_error(f"Game is paused and waiting for player {game_state.get_player_info(game_state.player_who_paused).name if game_state.get_player_info(game_state.player_who_paused) else 'previous player'} to reconnect.", 403)

     # Case 3: Reconnecting to an IN_PROGRESS game (e.g., after brief disconnect/refresh)
     elif game_state.status == GameStatus.IN_PROGRESS:
         app.logger.info(f"Player {joiner_id_from_request} attempting to join/reconnect to IN_PROGRESS game {game_id}.")
         if already_in_game_as_player:
             if not already_in_game_as_player.is_connected:
                 engine.handle_player_reconnect(joiner_id_from_request) # Mark as connected again
             # If already connected, no specific action, game state is current.
         else: # New player trying to join a full, in-progress game
             return Result.default_user_error("Game is already in progress and full.", 403)

     # Case 4: Game is truly ABORTED (no player_who_paused) or FINISHED
     elif game_state.status == GameStatus.ABORTED or game_state.status == GameStatus.FINISHED:
         return Result.default_user_error(f"Game is {game_state.status.value} and cannot be joined.", 400)

     else: # Should not be reached
         return Result.default_user_error(f"Game is in an unexpected state ({game_state.status.value}). Cannot join.", 500)

     await save_game_to_db_final(app, game_state)
     app.logger.info(f"Join/Reconnect attempt processed for player {joiner_id_from_request} in game {game_id}. New status: {game_state.status}")
     return Result.json(data=game_state.model_dump_for_api())

 except Exception as e:
     app.logger.error(f"Error joining game: {e}", exc_info=True)
     return Result.default_internal_error("Join game error.")
# --- END OF BLOCK 3 ---

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


# --- START OF BLOCK 6 (Modify api_make_move for NPC) ---
# FILE: ultimate_ttt_api.py
# Modify the api_make_move endpoint function

@export(mod_name=GAME_NAME, name="make_move", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_make_move(app: App, request: RequestData, data=None):
    move_payload = data or {}
    game_id: str = move_payload.get("game_id")
    human_player_id_making_move: Optional[str] = move_payload.get("player_id")  # ID of human submitting the move
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


import asyncio
from typing import AsyncGenerator, Dict, Any  # Ensure these are imported if not already


# --- START OF BLOCK 4 (api_open_game_stream) ---
# FILE: ultimate_ttt_api.py
# Replace the existing api_open_game_stream function.
# Ensure you have: from typing import AsyncGenerator, Dict, Any, Optional
# And: from datetime import datetime, timezone, timedelta

@export(mod_name=GAME_NAME, name="open_game_stream", api=True, request_as_kwarg=True, api_methods=['GET'])
async def api_open_game_stream(app: App, request: RequestData, game_id: str, player_id: Optional[str] = None):
    if not game_id:
        async def error_gen_no_id():
            yield {'event': 'error', 'data': {'message': 'game_id is required for stream'}}

        return Result.sse(stream_generator=error_gen_no_id())

    # The player_id param from query helps identify who disconnected if the stream is cancelled.
    listening_player_id = player_id

    async def game_event_generator() -> AsyncGenerator[Dict[str, Any], None]:
        app.logger.info(f"SSE: Stream opened for game_id: {game_id} by player_id: {listening_player_id or 'Unknown'}")
        last_known_updated_at = None
        last_status_sent = None
        last_players_connected_state: Optional[Dict[str, bool]] = None

        try:
            while True:
                game_state = await load_game_from_db_final(app, game_id)

                if not game_state:
                    app.logger.warning(f"SSE: Game {game_id} not found. Closing stream.")
                    yield {'event': 'error', 'data': {'message': 'Game not found. Stream closing.'}}
                    break

                # Timeout for games WAITING_FOR_OPPONENT (initial join)
                if game_state.status == GameStatus.WAITING_FOR_OPPONENT and \
                    game_state.waiting_since and \
                    (datetime.now(timezone.utc) - game_state.waiting_since > timedelta(
                        seconds=ONLINE_POLL_TIMEOUT_SECONDS)):
                    app.logger.info(f"SSE: Game {game_id} timed out waiting for opponent. Aborting.")
                    # Update game state to ABORTED
                    game_state.status = GameStatus.ABORTED
                    game_state.last_error_message = "Game aborted: Opponent didn't join in time."
                    game_state.player_who_paused = None  # No specific player paused
                    game_state.updated_at = datetime.now(timezone.utc)
                    await save_game_to_db_final(app, game_state)
                    yield {'event': 'game_update', 'data': game_state.model_dump_for_api()}
                    break  # End stream

                # Timeout for games PAUSED (ABORTED with player_who_paused)
                if game_state.status == GameStatus.ABORTED and \
                    game_state.player_who_paused and \
                    game_state.updated_at and \
                    (datetime.now(timezone.utc) - game_state.updated_at > timedelta(
                        seconds=ONLINE_POLL_TIMEOUT_SECONDS * 3)):  # e.g., 3x normal timeout
                    disconnected_player_name = "Player"
                    paused_player_info = game_state.get_player_info(game_state.player_who_paused)
                    if paused_player_info: disconnected_player_name = paused_player_info.name

                    app.logger.info(
                        f"SSE: Game {game_id} (paused) timed out waiting for {disconnected_player_name} ({game_state.player_who_paused}) to reconnect. Fully aborting.")
                    game_state.last_error_message = f"Game aborted: {disconnected_player_name} did not reconnect in time."
                    game_state.player_who_paused = None  # Mark as fully aborted
                    game_state.updated_at = datetime.now(timezone.utc)
                    # Status remains ABORTED
                    await save_game_to_db_final(app, game_state)
                    yield {'event': 'game_update', 'data': game_state.model_dump_for_api()}
                    break  # End stream

                current_updated_at = game_state.updated_at
                current_status = game_state.status
                current_players_connected = {p.id: p.is_connected for p in game_state.players}
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
                    (
                        game_state.status == GameStatus.ABORTED and not game_state.player_who_paused):  # Game truly finished or hard aborted
                    app.logger.info(
                        f"SSE: Game {game_id} is {game_state.status.value} (final). Sent final update. Closing stream.")
                    break

                await asyncio.sleep(0.75)  # Slightly increased sleep interval for polling

        except asyncio.CancelledError:
            app.logger.info(
                f"SSE: Stream for game_id: {game_id}, listening_player_id: {listening_player_id} was CANCELLED (client likely disconnected).")
            if listening_player_id and game_id:
                # This is where we detect a client closing the connection.
                # Load the latest game state to act upon it.
                game_state_on_disconnect = await load_game_from_db_final(app, game_id)
                if game_state_on_disconnect and game_state_on_disconnect.mode == GameMode.ONLINE:
                    player_info = game_state_on_disconnect.get_player_info(listening_player_id)
                    # Only process if the player was marked as connected and game is in a relevant state
                    if player_info and player_info.is_connected and \
                        (game_state_on_disconnect.status == GameStatus.IN_PROGRESS or game_state_on_disconnect.status == GameStatus.WAITING_FOR_OPPONENT or (
                             game_state_on_disconnect.status == GameStatus.ABORTED and game_state_on_disconnect.player_who_paused)):  # also if game was paused and the *other* player disconnects

                        app.logger.info(
                            f"SSE: Processing server-side disconnect for player {listening_player_id} in game {game_id} due to stream cancellation.")
                        engine = UltimateTTTGameEngine(game_state_on_disconnect)
                        engine.handle_player_disconnect(
                            listening_player_id)  # This updates status, player_who_paused etc.
                        await save_game_to_db_final(app, game_state_on_disconnect)
                        app.logger.info(
                            f"SSE: Post-disconnect save for game {game_id}. New status: {game_state_on_disconnect.status}")
                    else:
                        app.logger.info(
                            f"SSE: Player {listening_player_id} stream cancelled, but player already marked disconnected or game status ({game_state_on_disconnect.status if game_state_on_disconnect else 'N/A'}) not actionable for disconnect. No further action.")
            # No yield here as the stream is broken.
        except Exception as e:
            app.logger.error(f"SSE: Stream error for game_id {game_id}: {e}", exc_info=True)
            try:
                yield {'event': 'error', 'data': {'message': f'Server error in stream: {str(e)}'}}
            except Exception as yield_e:
                app.logger.error(f"SSE: Error yielding error message for game_id {game_id}: {yield_e}", exc_info=True)
        finally:
            app.logger.info(f"SSE: Stream closed for game_id: {game_id}, listening_player_id: {listening_player_id}")

    return Result.sse(stream_generator=game_event_generator())


# --- END OF BLOCK 4 ---


# --- UI Initialization ---
@export(mod_name=GAME_NAME, name="init_config", initial=True)  # Kept original name
def init_ultimate_ttt_module(app: App):
    app.run_any(("CloudM", "add_ui"),
                name=GAME_NAME,
                title="Ultimate SSE Tic-Tac-Toe",  # Simpler title
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
            .current-player-indicator {
                box-shadow: var(--shadow-dark);
            }
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

        .current-player-indicator-container {
            width: 100%;
            max-width: 900px; /* Angleichen an app-header Breite */
            margin-bottom: 1rem; /* Abstand nach unten */
            display: flex;
            justify-content: center; /* Zentriert den Indikator, falls er nicht volle Breite hat */
        }

        .current-player-indicator {
            height: 10px; /* Höhe des Farbbalkens */
            width: 100%; /* Volle Breite des Containers */
            max-width: 500px; /* Max-Breite, kann an section-card angepasst werden */
            border-radius: 5px;
            background-color: var(--border-color); /* Standardfarbe, wenn kein Spieler aktiv */
            transition: background-color 0.3s ease-in-out;
            box-shadow: var(--shadow-light); /* Optionaler leichter Schatten */
        }

        .current-player-indicator.player-X {
            background-color: var(--player-x-color);
        }

        .current-player-indicator.player-O {
            background-color: var(--player-o-color);
        }

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
        .button.secondary { background-color: var(--secondary-color); color: var(--text-color); }
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
            transition: box-shadow 0.2s, border-color 0.2s, background-color 0.2s; position: relative; outline-color 0.2s, outline-width 0.2s;
        }

        .local-board-container.forced-target {
            box-shadow: var(--forced-target-shadow); border-color: var(--forced-target-border) !important;
        }
        .local-board-container.playable-anywhere { border-color: var(--playable-anywhere-border) !important; opacity: 0.9; }

        .local-board-container.won-X { background-color: var(--local-won-x-bg); }
        .local-board-container.won-O { background-color: var(--local-won-o-bg); }
        .local-board-container.won-DRAW { background-color: var(--local-draw-bg); }
        .local-board-container.preview-forced-for-x {
            outline: 3px dashed var(--player-x-color);
            outline-offset: -3px; /* Damit der Outline innerhalb des Borders ist */
            /* box-shadow: 0 0 10px var(--player-x-color); Alternativ ein Glow */
        }
        .local-board-container.preview-forced-for-o {
            outline: 3px dashed var(--player-o-color);
            outline-offset: -3px;
            /* box-shadow: 0 0 10px var(--player-o-color); Alternativ ein Glow */
        }
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

        .cell.last-move {
            /* Beispiel: ein heller, auffälliger Innenrand */
            box-shadow: inset 0 0 0 2.5px gold;
            /* Oder ein subtilerer Hintergrund, je nach Präferenz */
            /* background-color: rgba(255, 215, 0, 0.15); */
        }
        .cell.last-move.player-X { /* Stellt sicher, dass der Rand auch bei Spieler-X sichtbar ist */
             box-shadow: inset 0 0 0 2.5px gold, 0 0 0 0 var(--player-x-color); /* Trick um Default-Schatten zu resetten, falls vorhanden */
        }
        .cell.last-move.player-O { /* Stellt sicher, dass der Rand auch bei Spieler-O sichtbar ist */
             box-shadow: inset 0 0 0 2.5px gold, 0 0 0 0 var(--player-o-color);
        }

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

        .status-text {
            font-size: 0.9em;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            text-align: center;
        }
        .status-text.info { background-color: rgba(var(--info-colorRGB, 13,202,240), 0.15); color: var(--info-color); }
        .status-text.warning { background-color: rgba(var(--warning-colorRGB, 255,193,7), 0.15); color: var(--warning-color); }
        .status-text.error { background-color: rgba(var(--danger-colorRGB, 220,53,69), 0.15); color: var(--danger-color); }
    </style>
</head>
<body>
    <header class="app-header">
        <h1 class="app-title">Ultimate TTT</h1>
        <button id="themeToggleBtn" class="theme-switcher none">Dark Mode</button>
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
            <div class="form-group"> <!-- Player 2 Setup Group -->
             <label for="player2TypeSelect">Player 2 (O):</label>
             <select id="player2TypeSelect" class="form-select">
                 <option value="human" selected>Human</option>
                 <option value="npc">NPC (Computer)</option>
             </select>
         </div>
         <div id="localP2NameGroup" class="form-group"> <!-- Shows if P2 is Human -->
             <label for="player2NameInput">Player 2 (O) Name:</label>
             <input type="text" id="player2NameInput" class="form-input" value="Player O">
         </div>
         <div id="npcDifficultyGroup" class="form-group hidden"> <!-- Shows if P2 is NPC -->
             <label for="npcDifficultySelect">NPC Difficulty:</label>
             <select id="npcDifficultySelect" class="form-select">
                 <option value="easy" selected>Easy</option>
                 <option value="medium">Medium</option>
                 <option value="hard">Hard</option>
                 <!-- <option value="insane">Insane</option> -->
             </select>
         </div>

            <div class="button-row">
                <button id="startLocalGameBtn" class="button">Play Local Game</button>
                 <button id="resumeLocalGameBtn" class="button secondary hidden">Resume Local <span id="resumeGridSizeText"></span> Game</button>
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
        <button id="toggleRulesButton" onclick="toggleRulesVisibility()" style="padding: 8px 12px; cursor: pointer; border: 1px solid #bbb; background-color: var(--theme-secondary); border-radius: 4px; font-size: 0.9em; color:var(--text-color);">
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
        let resetGameBtn, backToMenuBtn, themeToggleBtn;
        let statsGamesPlayedEl, statsWinsEl, statsLossesEl, statsDrawsEl, statsSessionIdEl;
        let modalOverlay, modalTitle, modalMessage, modalCancelBtn, modalConfirmBtn;
        let currentPlayerIndicatorContainer, currentPlayerIndicator
        let modalConfirmCallback = null;

        let player2TypeSelect, npcDifficultyGroup, npcDifficultySelect;

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
        // let onlineGamePollInterval = null;
        // const POLLING_RATE_MS = 2500; // Increased polling rate slightly
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

            player2TypeSelect = document.getElementById('player2TypeSelect');
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

            currentPlayerIndicatorContainer = document.getElementById('currentPlayerIndicatorContainer');
            currentPlayerIndicator = document.getElementById('currentPlayerIndicator');

            resumeLocalGameBtn = document.getElementById('resumeLocalGameBtn');
            resumeGridSizeTextEl = document.getElementById('resumeGridSizeText');
            saveAndLeaveBtn = document.getElementById('saveAndLeaveBtn');

            setupEventListeners();
            initializeTheme();
            determineSessionId();
            loadSessionStats();
            checkUrlForJoin();
            updateResumeButtonVisibility();
            showScreen('gameSetup');
        }

       function connectToGameStream(gameId) {
             let ssePath = `/sse/${API_MODULE_NAME}/open_game_stream?game_id=${gameId}`;
             // Pass client's player ID if known, so server can identify who disconnected if stream cancels
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

             currentSseGameIdPath = ssePath; // Store the full path including player_id
             console.log(`SSE: Attempting to connect to ${currentSseGameIdPath}`);

             sseConnection = TB.sse.connect(currentSseGameIdPath, {
                 onOpen: (event) => {
                     console.log(`SSE: Connection opened to ${currentSseGameIdPath}`, event);
                     // if(window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("Live game connection active!", {duration: 1500});
                 },
                 onError: (error) => {
                     console.error(`SSE: Connection error with ${currentSseGameIdPath}`, error);
                     if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Live connection failed. Refresh may be needed.", {duration: 3000});
                     // sseConnection = null; // TB.sse.connect might handle this, or we ensure it in disconnect
                     // currentSseGameIdPath = null;
                     // Consider more robust error handling, e.g., navigating to setup or auto-retry
                 },
                 listeners: {
                     'game_update': (eventPayload, event) => {
                         console.log('SSE Event (game_update):', eventPayload);
                         if (eventPayload && eventPayload.game_id) {
                             if (currentGameId === eventPayload.game_id || !currentGameId) { // Process if current game or if no game was set yet
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
                          // Server explicitly closed (e.g. game finished/aborted).
                          // Client-side sseConnection might be cleared by TB.sse or here.
                          if (sseConnection && currentSseGameIdPath === ssePath) { // Check if it's still our active stream
                              disconnectFromGameStream(); // Ensure client state is clean
                          }
                     }
                 }
             });
         }
        function disconnectFromGameStream() {
            if (sseConnection && currentSseGameIdPath) {
                console.log(`SSE: Disconnecting from ${currentSseGameIdPath}`);
                TB.sse.disconnect(currentSseGameIdPath); // Use Toolbox's method to close the connection
                sseConnection = null;
                currentSseGameIdPath = null;
                // if(window.TB?.ui?.Toast) TB.ui.Toast.showInfo("Live connection closed.", {duration: 1000}); // Optional: can be noisy
            }
        }

        function setupEventListeners() {
            startLocalGameBtn.addEventListener('click', () => createNewGame('local', false));
            startOnlineGameBtn.addEventListener('click', () => createNewGame('online'));

            resumeLocalGameBtn.addEventListener('click', () => createNewGame('local', true)); // true = fortsetzen
            saveAndLeaveBtn.addEventListener('click', saveLocalGameAndLeave);
            gridSizeSelect.addEventListener('change', updateResumeButtonVisibility);

            joinOnlineGameBtn.addEventListener('click', joinOnlineGame);
            gameIdShareEl.addEventListener('click', copyGameIdToClipboard);
            cancelOnlineWaitBtn.addEventListener('click', () => {
                // TODO: API call to abort game if P1 created it and cancels
                showScreen('gameSetup');
            });

            player2TypeSelect.addEventListener('change', togglePlayer2Setup)

            resetGameBtn.addEventListener('click', confirmResetGame);
            backToMenuBtn.addEventListener('click', confirmBackToMenu);
            themeToggleBtn.addEventListener('click', toggleTheme);
            globalGridDisplay.addEventListener('click', onBoardClickDelegation);

            globalGridDisplay.addEventListener('mouseover', handleCellMouseOver);
            globalGridDisplay.addEventListener('mouseout', handleCellMouseOut);

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

        function togglePlayer2Setup() {
            const isNPC = player2TypeSelect.value === 'npc';
            localP2NameGroup.classList.toggle('hidden', isNPC);
            npcDifficultyGroup.classList.toggle('hidden', !isNPC);

            // If switching to local, re-enable P2 name group if it was hidden by online mode logic
            if (currentGameState && currentGameState.mode === 'local' && !isNPC) {
                 localP2NameGroup.classList.remove('hidden');
            }
        }

        function getLocalStorageKeyForSize(size) {
            return `${LOCAL_STORAGE_GAME_PREFIX}${size}x${size}`;
        }

        function saveLocalGame(gameState) {
            if (!gameState || gameState.mode !== 'local' || gameState.status !== 'in_progress') {
                console.warn("Cannot save game: Not a local game in progress.", gameState);
                return false;
            }
            try {
                const key = getLocalStorageKeyForSize(gameState.config.grid_size);
                // Pydantic model_dump_for_api gibt bereits ein serialisierbares Objekt zurück.
                // Wir müssen sicherstellen, dass alle Datumsangaben Strings sind, was model_dump_for_api tun sollte.
                localStorage.setItem(key, JSON.stringify(gameState));
                console.log(`Local game ${key} saved.`, gameState);
                if(window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("Game saved locally!", {duration: 1500});
                return true;
            } catch (e) {
                console.error("Error saving local game to localStorage:", e);
                if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Could not save game locally.", {duration: 2000});
                return false;
            }
        }

        function loadLocalGame(size) {
            try {
                const key = getLocalStorageKeyForSize(size);
                const savedGameJSON = localStorage.getItem(key);
                if (savedGameJSON) {
                    const savedGameState = JSON.parse(savedGameJSON);
                    console.log(`Local game ${key} loaded.`, savedGameState);
                    // Wichtig: Hier könnten wir eine Validierung mit Pydantic-ähnlicher Struktur durchführen,
                    // aber für localStorage ist eine einfache Prüfung oft ausreichend, oder man vertraut den Daten.
                    // Sicherstellen, dass es ein valides GameState-Objekt ist (zumindest rudimentär).
                    if (savedGameState && savedGameState.game_id && savedGameState.config && savedGameState.mode === 'local') {
                        // Datumsfelder könnten als Strings gespeichert sein, Pydantic im Backend behandelt das beim Laden aus DB.
                        // Hier brauchen wir das nicht unbedingt, da JS damit umgehen kann, es sei denn, wir machen Berechnungen damit.
                        return savedGameState;
                    }
                }
            } catch (e) {
                console.error("Error loading local game from localStorage:", e);
            }
            return null;
        }

        function deleteLocalGame(size) {
            try {
                const key = getLocalStorageKeyForSize(size);
                localStorage.removeItem(key);
                console.log(`Local game ${key} deleted.`);
                updateResumeButtonVisibility(); // Aktualisiere Button-Sichtbarkeit
            } catch (e) {
                console.error("Error deleting local game from localStorage:", e);
            }
        }

        function updateResumeButtonVisibility() {
            if (!gridSizeSelect || !resumeLocalGameBtn || !resumeGridSizeTextEl) return;

            const selectedSize = parseInt(gridSizeSelect.value);
            const savedGame = loadLocalGame(selectedSize);

            if (savedGame && savedGame.status === 'in_progress') { // Nur fortsetzen, wenn es noch läuft
                resumeGridSizeTextEl.textContent = `${selectedSize}x${selectedSize}`;
                resumeLocalGameBtn.classList.remove('hidden');
            } else {
                resumeLocalGameBtn.classList.add('hidden');
            }
        }

        function saveLocalGameAndLeave() {
            if (currentGameState && currentGameState.mode === 'local' && currentGameState.status === 'in_progress') {
                if (saveLocalGame(currentGameState)) {
                    showScreen('gameSetup');
                    // currentGameState und currentGameId werden in showScreen('gameSetup') zurückgesetzt
                }
            } else {
                 if(window.TB?.ui?.Toast) TB.ui.Toast.showWarning("No active local game to save.", {duration: 2000});
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

                function getPlayerInfoById(playerId) {
            if (!currentGameState || !currentGameState.players) return null;
            return currentGameState.players.find(p => p.id === playerId);
        }

        function getOpponentInfo(playerId) {
            if (!currentGameState || !currentGameState.players) return null;
            return currentGameState.players.find(p => p.id !== playerId);
        }


        function handleCellMouseOver(event) {
            if (!currentGameState || currentGameState.status !== 'in_progress') return;

            const cell = event.target.closest('.cell');
            if (!cell || !cell.classList.contains('playable')) return; // Nur auf spielbaren Zellen reagieren

            // Ist der aktuelle Spieler am Zug?
            let isMyTurn;
            if (currentGameState.mode === 'local') {
                isMyTurn = true; // Im lokalen Modus kann man immer für den aktuellen Spieler hovern
            } else {
                isMyTurn = clientPlayerInfo && currentGameState.current_player_id === clientPlayerInfo.id;
            }
            if (!isMyTurn) return;


            const N = currentGameState.config.grid_size;
            const hovered_lr = parseInt(cell.dataset.lr);
            const hovered_lc = parseInt(cell.dataset.lc);

            // Zielkoordinaten für das nächste Board
            const target_gr = hovered_lr;
            const target_gc = hovered_lc;

            // Prüfen, ob das Ziel-Board überhaupt existiert (sollte es, wenn N korrekt ist)
            if (target_gr < 0 || target_gr >= N || target_gc < 0 || target_gc >= N) {
                console.warn("Preview: Target global coords out of bounds", target_gr, target_gc);
                return;
            }

            const currentPlayer = getPlayerInfoById(currentGameState.current_player_id);
            if (!currentPlayer) return;

            const opponent = getOpponentInfo(currentGameState.current_player_id);
            if (!opponent) return; // Sollte im 2-Spieler-Modus nicht passieren

            // Finde das DOM-Element des Ziel-Global-Boards
            const targetBoardElement = globalGridDisplay.querySelector(
                `.local-board-container[data-gr="${target_gr}"][data-gc="${target_gc}"]`
            );

            if (targetBoardElement) {
                // Ist das Ziel-Board bereits gewonnen oder voll?
                const isTargetBoardWon = currentGameState.global_board_winners[target_gr][target_gc] !== 'NONE';

                let isTargetBoardFull = false;
                if (!isTargetBoardWon) {
                    const targetLocalCells = currentGameState.local_boards_state[target_gr][target_gc];
                    isTargetBoardFull = targetLocalCells.every(row => row.every(cellState => cellState !== '.'));
                }

                if (isTargetBoardWon || isTargetBoardFull) {
                    // "Play Anywhere"-Szenario: Alle *noch nicht gewonnenen und nicht vollen* Boards hervorheben
                    // oder keine spezifische Vorschau anzeigen. Fürs Erste keine spezifische Vorschau,
                    // da die "playable-anywhere"-Klasse bereits alle gültigen Boards markiert.
                    // Man könnte hier eine subtile generische Vorschau für alle "playable-anywhere" Boards hinzufügen.
                    // console.log("Preview: Target board won/full, would be 'play anywhere'");
                } else {
                    // Das spezifische Board wird das Ziel sein
                    const previewClass = opponent.symbol === 'X' ? 'preview-forced-for-x' : 'preview-forced-for-o';
                    targetBoardElement.classList.add(previewClass);
                }
            }
        }

        function handleCellMouseOut(event) {
            if (!currentGameState) return; // Sicherstellen, dass ein Spielstatus existiert

            const cell = event.target.closest('.cell');
            // Auch wenn wir von einer Zelle zu einer anderen im selben Board wechseln,
            // ist es am einfachsten, alle Vorschauen zu entfernen und sie bei Bedarf neu zu erstellen.
            if (cell) { // Nur wenn der Mauszeiger eine Zelle verlässt
                const allBoardContainers = globalGridDisplay.querySelectorAll('.local-board-container');
                allBoardContainers.forEach(board => {
                    board.classList.remove('preview-forced-for-x', 'preview-forced-for-o');
                });
            }
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

            if(saveAndLeaveBtn) saveAndLeaveBtn.classList.add('hidden');

            if (screenName === 'gameSetup') {
                if(statsSection) statsSection.classList.remove('hidden');
                // Call togglePlayer2Setup to ensure correct P2 input fields are shown based on select value
                if(player2TypeSelect) togglePlayer2Setup(); // Important for resetting view

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
                 url = `/api/${API_MODULE_NAME}/${endpoint.split('/')[0]}?game_id=${queryParams.game_id}`; // Construct path
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

         async function createNewGame(mode, resumeIfAvailable = false) {
            const size = parseInt(gridSizeSelect.value);

            if (mode === 'local' && resumeIfAvailable) {
                const existingGame = loadLocalGame(size);
                if (existingGame && existingGame.status === 'in_progress') {
                    console.log("Resuming local game:", existingGame);
                    clientPlayerInfo = null;
                    processGameStateUpdate(existingGame);
                    showScreen('game');
                    if(saveAndLeaveBtn) saveAndLeaveBtn.classList.remove('hidden');
                    return;
                } else {
                    console.log("No local game to resume for size", size, "or game was finished. Starting new one.");
                }
            }

            const config = { grid_size: size };
            const p1Name = player1NameInput.value.trim() || (mode === 'local' ? "Player X" : "Me");
            const payload = { config, mode, player1_name: p1Name };

            if (mode === 'local') {
                deleteLocalGame(size); // Clear any old saved game of this size before starting new
                payload.player2_type = player2TypeSelect.value;
                if (payload.player2_type === 'npc') {
                    payload.npc_difficulty = npcDifficultySelect.value;
                    // Player 2 name for NPC is set server-side
                } else { // Human P2
                    payload.player2_name = player2NameInput.value.trim() || "Player O";
                }
            }

            // For local games, ensure player2 specific inputs are correctly shown/hidden before showing screen
            if(mode === 'local') togglePlayer2Setup();


            const response = await apiRequest('create_game', payload, 'POST', {hideMainContentWhileLoading: true});

            if (!response.error && response.data?.game_id) {
                if (mode === 'local') {
                    clientPlayerInfo = null; // No specific client info for local, turns are just X/O
                    processGameStateUpdate(response.data);
                    showScreen('game');
                    if(saveAndLeaveBtn) saveAndLeaveBtn.classList.remove('hidden');
                    updateResumeButtonVisibility();
                } else if (mode === 'online') {
                    clientPlayerInfo = response.data.players.find(p => p.id === currentSessionId);
                    if (!clientPlayerInfo && response.data.players.length > 0) {
                        clientPlayerInfo = response.data.players[0];
                        console.warn("Online game created: Forcing clientPlayerInfo to P1.", currentSessionId, clientPlayerInfo.id);
                    }
                    gameIdShareEl.textContent = response.data.game_id;
                    waitingStatusEl.textContent = `Waiting for opponent... Game ID: ${response.data.game_id}`;
                    showScreen('onlineWait');
                    processGameStateUpdate(response.data); // This will call connectToGameStream for P1
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
             // CRITICAL: Set clientPlayerInfo based on the response from the server and current session ID.
             // currentSessionId should be established by determineSessionId() and be consistent.
             clientPlayerInfo = response.data.players.find(p => p.id === currentSessionId);

             if (!clientPlayerInfo) {
                 // This is a fallback - ideally currentSessionId matches an ID in the game.
                 // This could happen if the server assigned a guest_id and the client's currentSessionId is
                 // a newly generated one that doesn't match.
                 // For a join, the player joining should be the last one added or the one whose symbol is 'O' if P1 is 'X'.
                 if (response.data.players.length === 2) {
                      const playerX = response.data.players.find(p => p.symbol === 'X');
                      const playerO = response.data.players.find(p => p.symbol === 'O');
                      // If I am joining, I am likely player O if player X exists and is not me.
                      if (playerO && playerO.id.startsWith("guest_") && currentSessionId.startsWith("guest_")) {
                         clientPlayerInfo = playerO; // Tentatively assume this guest is me
                      } else if (playerX && playerX.id !== currentSessionId && playerO) {
                         clientPlayerInfo = playerO;
                      }
                      // If currentSessionId IS one of the players, the initial find should work.
                 }
                 console.warn("Online game joined: ClientPlayerInfo might not be perfectly matched. My session ID:", currentSessionId, "Inferred ClientInfo:", clientPlayerInfo, "Players in game:", response.data.players);
             } else {
                 console.log("Online game joined/rejoined: ClientPlayerInfo set to:", clientPlayerInfo);
             }

             processGameStateUpdate(response.data); // Handles UI, SSE connection logic, rendering

             // Screen transition logic
             if (response.data.status === 'in_progress' || (response.data.status === 'ABORTED' && response.data.player_who_paused)) {
                  showScreen('game');
             } else if (response.data.status === 'waiting_for_opponent'){
                 gameIdShareEl.textContent = response.data.game_id; // For P1 rejoining their own waiting game
                 waitingStatusEl.textContent = `Waiting for opponent... Game ID: ${response.data.game_id}`;
                 showScreen('onlineWait');
             } else { // Finished, or hard aborted
                  if(window.TB?.ui?.Toast) TB.ui.Toast.showError(response.data.last_error_message || "Could not join game (unexpected status).");
                  showScreen('gameSetup');
             }
         }
        }

         function processGameStateUpdate(newGameState) {
         console.log("PROCESS_GAME_STATE_UPDATE - Received:", newGameState);

         let previousPlayerConnectedStates = {};
         let oldGameStatus = null;
         let oldCurrentPlayerId = null;

         if (currentGameState) { // Capture state *before* overwriting
             oldGameStatus = currentGameState.status;
             oldCurrentPlayerId = currentGameState.current_player_id;
             if (currentGameState.players && clientPlayerInfo) {
                 currentGameState.players.forEach(p => {
                     if (p.id !== clientPlayerInfo.id) { // Opponent
                         previousPlayerConnectedStates[p.id] = p.is_connected;
                     }
                 });
             }
         }

         currentGameState = newGameState; // Main state update

         if (!currentGameState || !currentGameState.game_id) {
             console.error("Invalid newGameState received!");
             if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Error: Corrupted game update.");
             disconnectFromGameStream();
             showScreen('gameSetup');
             return;
         }
         currentGameId = newGameState.game_id; // Ensure currentGameId is also updated

         // Re-confirm clientPlayerInfo, especially if game was joined and this is the first update
         if (currentGameState.mode === 'online' && (!clientPlayerInfo || !currentGameState.players.find(p => p.id === clientPlayerInfo.id))) {
              clientPlayerInfo = currentGameState.players.find(p => p.id === currentSessionId);
              console.log("PROCESS_GAME_STATE_UPDATE (Online) - ClientPlayerInfo (re)validated:", clientPlayerInfo);
         }

         // --- Handle Toasts for Opponent Connect/Disconnect ---
         if (currentGameState.mode === 'online' && clientPlayerInfo && currentGameState.players) {
             currentGameState.players.forEach(opponent => {
                 if (opponent.id !== clientPlayerInfo.id) {
                     const wasConnected = previousPlayerConnectedStates[opponent.id]; // Could be undefined if first update
                     const isConnected = opponent.is_connected;

                     if (wasConnected === true && isConnected === false) {
                         // Opponent just disconnected
                         if (window.TB?.ui?.Toast) TB.ui.Toast.showWarning(`${opponent.name} disconnected. Waiting for reconnect...`, {duration: 3500});
                     } else if (wasConnected === false && isConnected === true && previousPlayerConnectedStates.hasOwnProperty(opponent.id)) {
                         // Opponent just reconnected
                         if (window.TB?.ui?.Toast) TB.ui.Toast.showSuccess(`${opponent.name} reconnected! Game resumes.`, {duration: 3000});
                     }
                 }
             });
         }
         // --- End Toasts ---

         if (currentGameState.mode === 'local') {
             const currentPlayer = currentGameState.players.find(p => p.id === currentGameState.current_player_id);
             localPlayerActiveSymbol = currentPlayer ? currentPlayer.symbol : '?';
             disconnectFromGameStream();
             if (saveAndLeaveBtn) {
                 saveAndLeaveBtn.classList.toggle('hidden', currentGameState.status !== 'in_progress');
             }
         } else if (currentGameState.mode === 'online') {
             const onlineWaitScreenActive = !document.getElementById('onlineWaitSection').classList.contains('hidden');
             if (onlineWaitScreenActive && currentGameState.status === 'in_progress') {
                 if(window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("Opponent connected! Game starting.", {duration: 2000});
                 showScreen('game');
             }

             // Manage SSE connection
             const isMyTurnOnline = clientPlayerInfo && currentGameState.current_player_id === clientPlayerInfo.id;

             if (currentGameState.status === 'in_progress') {
                 if (isMyTurnOnline) {
                     disconnectFromGameStream();
                     if (oldGameStatus === 'in_progress' && oldCurrentPlayerId !== currentGameState.current_player_id) { // Check if turn just changed to me
                        if(window.TB?.ui?.Toast) TB.ui.Toast.showInfo("It's your turn!", {duration: 2000});
                     } else if (oldGameStatus === 'ABORTED' && currentGameState.status === 'in_progress') { // Game just resumed to my turn
                        if(window.TB?.ui?.Toast) TB.ui.Toast.showInfo("Game resumed. It's your turn!", {duration: 2000});
                     }
                 } else { // Opponent's turn or game not ready for my move
                     connectToGameStream(currentGameState.game_id);
                 }
             } else if (currentGameState.status === 'waiting_for_opponent') {
                 // If I am P1 and waiting, I should listen.
                 if (clientPlayerInfo && currentGameState.players.length > 0 && currentGameState.players[0].id === clientPlayerInfo.id) {
                     connectToGameStream(currentGameState.game_id);
                 } else {
                     disconnectFromGameStream(); // P2 doesn't need to listen if P1 is creating/waiting.
                 }
             } else if (currentGameState.status === 'ABORTED' && currentGameState.player_who_paused) {
                 // Game is paused. If I am NOT the one who paused, I should listen for their reconnect or timeout.
                 if (clientPlayerInfo && clientPlayerInfo.id !== currentGameState.player_who_paused) {
                     connectToGameStream(currentGameState.game_id);
                 } else { // I am the one who paused, I don't need to listen to my own pause.
                     disconnectFromGameStream();
                 }
             } else if (currentGameState.status === 'finished' || (currentGameState.status === 'ABORTED' && !currentGameState.player_who_paused)) {
                 disconnectFromGameStream(); // Game truly over.
             }
             if (saveAndLeaveBtn) saveAndLeaveBtn.classList.add('hidden'); // No save & leave for online
         }

         renderBoard(); // This is crucial for playability after reconnect
         updateStatusBar();

         if (currentGameState.status === 'finished' || (currentGameState.status === 'ABORTED' && !currentGameState.player_who_paused)) {
             if (saveAndLeaveBtn) saveAndLeaveBtn.classList.add('hidden');
             if (currentGameState.mode === 'local' && currentGameState.status === 'finished') {
                 deleteLocalGame(currentGameState.config.grid_size);
             }
             disconnectFromGameStream();
             if (currentGameState.status === 'finished') {
                 showGameOverModal();
                 loadSessionStats();
             } else { // Hard ABORTED
                  showModal("Game Aborted", currentGameState.last_error_message || "The game was aborted.", () => showScreen('gameSetup'));
             }
         }

         // Update Reset/Leave button text and style (from previous correct block)
         if (resetGameBtn) {
             if (currentGameState.mode === 'online' &&
                 (currentGameState.status === 'in_progress' || (currentGameState.status === 'ABORTED' && currentGameState.player_who_paused))) {
                 resetGameBtn.textContent = 'Leave Game';
                 resetGameBtn.classList.add('danger');
                 resetGameBtn.classList.remove('secondary');
             } else if (currentGameState.mode === 'local') {
                 resetGameBtn.textContent = 'Reset Game';
                 resetGameBtn.classList.add('danger');
                 resetGameBtn.classList.remove('secondary');
             } else {
                 resetGameBtn.textContent = 'Back to Menu';
                 resetGameBtn.classList.remove('danger');
                 resetGameBtn.classList.add('secondary');
             }
         }
     }


        async function makePlayerMove(globalR, globalC, localR, localC) {
            if (!currentGameState || !currentGameId || currentGameState.status !== 'in_progress') {
                console.warn("Make move called but game not in a playable state.");
                return;
            }

            let playerIdForMove;
            const currentPlayerOnClient = currentGameState.players.find(p => p.id === currentGameState.current_player_id);

            if (currentGameState.mode === 'local') {
                // In local mode, the current_player_id from game state IS the one making the move.
                // This could be LOCAL_P1_ID, LOCAL_P2_ID, or an NPC_ID.
                // The UI click should only be enabled for the human player if it's their turn.
                // If an NPC is current_player_id, this function shouldn't be directly triggerable by UI click for that NPC.
                // Server will handle NPC moves.
                playerIdForMove = currentGameState.current_player_id;
                if (currentPlayerOnClient && currentPlayerOnClient.is_npc) {
                    console.warn("UI tried to make a move for an NPC. This should be handled server-side. Ignoring.");
                    return;
                }
            } else { // Online mode
                if (!clientPlayerInfo || currentGameState.current_player_id !== clientPlayerInfo.id) {
                    if(window.TB?.ui?.Toast) TB.ui.Toast.showInfo("Not your turn.");
                    return;
                }
                playerIdForMove = clientPlayerInfo.id;
            }

            const movePayload = {
                player_id: playerIdForMove, global_row: globalR, global_col: globalC,
                local_row: localR, local_col: localC, game_id: currentGameId
            };

            if (globalGridDisplay) globalGridDisplay.style.pointerEvents = 'none'; // Prevent double-clicks
            const response = await apiRequest(`make_move`, movePayload, 'POST');
            if (globalGridDisplay) globalGridDisplay.style.pointerEvents = 'auto';

            if (!response.error && response.data) {
                // Server response includes state after human move AND any subsequent NPC move.
                processGameStateUpdate(response.data);
            } else if (response.data?.game_id) { // Error response but contains game state
                processGameStateUpdate(response.data);
            }
            // processGameStateUpdate manages SSE connection (e.g., connect if now opponent's turn in online)
        }


        function confirmResetGame() {
         if (!currentGameState) return;

         if (currentGameState.mode === 'online' &&
             (currentGameState.status === 'in_progress' || (currentGameState.status === 'ABORTED' && currentGameState.player_who_paused))) {
             showModal('Leave Game?', 'Are you sure you want to leave this online game? Your opponent will be notified.',
                 async () => {
                     // Client simply disconnects its SSE and goes to menu.
                     // The server-side SSE CancelledError will handle marking the player as disconnected.
                     disconnectFromGameStream();
                     showScreen('gameSetup');
                     // No explicit API call to "leave", relying on SSE stream cancellation to trigger backend logic.
                 }
             );
         } else if (currentGameState.mode === 'local' && (currentGameState.status === 'in_progress' || currentGameState.status === 'finished' || currentGameState.status === 'ABORTED')) {
             showModal('Reset Game?', 'Start a new local game with current settings?',
                 async () => {
                     deleteLocalGame(currentGameState.config.grid_size); // Clear any saved state for this size
                     await createNewGame('local'); // Uses current form values
                 }
             );
         } else if (currentGameState.status === 'finished' || (currentGameState.status === 'ABORTED' && !currentGameState.player_who_paused) ) {
             // For finished/hard-aborted online games, "Reset" means go to menu essentially.
             showModal('New Game?', 'Return to the menu to start a new game?', () => showScreen('gameSetup'));
         } else {
              if(window.TB?.ui?.Toast) TB.ui.Toast.showInfo("Cannot reset/leave from current game state like this.");
         }
     }
        function confirmBackToMenu() {
            if (currentGameState && currentGameState.status !== 'finished' && currentGameState.status !== 'aborted') {
                let message = 'Current game progress will be lost. Are you sure?';
                if (currentGameState.mode === 'local' && currentGameState.status === 'in_progress') {
                    message = 'Game is not saved. Progress will be lost. Use "Save & Leave" to save. Continue to menu?';
                }
                showModal('Back to Menu?', message, () => {
                    // Wenn ein lokales Spiel nicht gespeichert und verlassen wurde, könnte es hier explizit gelöscht werden,
                    // oder man verlässt sich darauf, dass beim Start eines neuen Spiels der alte Speicher gelöscht wird.
                    // deleteLocalGame(currentGameState.config.grid_size); // Optional: Sofort löschen
                    showScreen('gameSetup');
                });
            } else {
                showScreen('gameSetup');
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
            const N = currentGameState.config?.grid_size;
            dynamicallySetGridStyles(N); // Ensure styles are set for current N
            globalGridDisplay.innerHTML = '';

            const tempAllBoardContainers = globalGridDisplay.querySelectorAll('.local-board-container');
             tempAllBoardContainers.forEach(board => {
                 board.classList.remove('preview-forced-for-x', 'preview-forced-for-o');
             });

            console.log("RENDER_BOARD - Forced target from state:", currentGameState.next_forced_global_board);

            const lastMoveCoords = currentGameState.last_made_move_coords;

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

                            if (lastMoveCoords) {
                                const [last_gr, last_gc, last_lr, last_lc] = lastMoveCoords;
                                if (gr === last_gr && gc === last_gc && lr === last_lr && lc === last_lc) {
                                    cell.classList.add('last-move');
                                }
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

        // --- START OF BLOCK 7 (JS updateStatusBar) ---
     function updateStatusBar() {
             if (!statusBar || !currentGameState || !currentPlayerIndicator) return;
             let message = ""; let msgType = "info";

             currentPlayerIndicator.className = 'current-player-indicator'; // Reset classes

             if (currentGameState.status === 'waiting_for_opponent') {
                 message = "Waiting for opponent to join...";
                 if (currentGameState.players.length > 0 && currentGameState.players[0].is_connected) {
                     currentPlayerIndicator.classList.add(`player-${currentGameState.players[0].symbol}`);
                 }
             } else if (currentGameState.status === 'in_progress') {
                 const currentPlayer = currentGameState.players.find(p => p.id === currentGameState.current_player_id);
                 const pName = currentPlayer ? currentPlayer.name : "Player";
                 const pSymbol = currentPlayer ? currentPlayer.symbol : "?";

                 if (currentPlayer) {
                     currentPlayerIndicator.classList.add(`player-${pSymbol}`);
                 }

                 if (currentGameState.mode === 'local') {
                     message = `${pName} (${pSymbol})'s Turn.`;
                 } else { // Online
                     const amICurrentPlayer = clientPlayerInfo && clientPlayerInfo.id === currentGameState.current_player_id;
                     message = amICurrentPlayer ?
                               `Your Turn (${clientPlayerInfo.symbol})` :
                               `Waiting for ${pName} (${pSymbol})...`;

                     // Check for opponent disconnect if game is IN_PROGRESS but an opponent is not connected
                     const opponent = currentGameState.players.find(p => p.id !== currentGameState.current_player_id);
                     if (opponent && !opponent.is_connected) {
                         message = `Waiting for ${opponent.name} (${opponent.symbol}) to reconnect...`;
                         msgType = "warning"; // Indicate a problem
                         // Current player indicator should be for the one still connected and whose turn it effectively is to wait
                         if (currentPlayer && currentPlayer.is_connected) {
                              currentPlayerIndicator.className = `current-player-indicator player-${currentPlayer.symbol}`;
                         } else if (clientPlayerInfo && clientPlayerInfo.is_connected) {
                             currentPlayerIndicator.className = `current-player-indicator player-${clientPlayerInfo.symbol}`;
                         }
                     }
                 }

                 if (currentGameState.next_forced_global_board) {
                     const [gr, gc] = currentGameState.next_forced_global_board;
                     message += (currentGameState.mode === 'local' || (clientPlayerInfo && clientPlayerInfo.id === currentGameState.current_player_id)) ?
                                ` Play in board (${gr+1},${gc+1}).` : "";
                 } else if (currentGameState.status === 'in_progress') {
                      message += (currentGameState.mode === 'local' || (clientPlayerInfo && clientPlayerInfo.id === currentGameState.current_player_id)) ?
                                 " Play in any valid highlighted board." : "";
                 }

             } else if (currentGameState.status === 'ABORTED' && currentGameState.player_who_paused) {
                 // This is the "paused" state, waiting for a specific player.
                 msgType = "warning";
                 const disconnectedPlayerInfo = currentGameState.players.find(p => p.id === currentGameState.player_who_paused);
                 const disconnectedPlayerName = disconnectedPlayerInfo ? disconnectedPlayerInfo.name : "Opponent";
                 message = `Player ${disconnectedPlayerName} disconnected. Waiting for them to rejoin...`;

                 // Set indicator for the player who is still connected (if any)
                 const waitingPlayer = currentGameState.players.find(p => p.id !== currentGameState.player_who_paused && p.is_connected);
                 if (waitingPlayer) {
                     currentPlayerIndicator.classList.add(`player-${waitingPlayer.symbol}`);
                 } else if (clientPlayerInfo && clientPlayerInfo.id !== currentGameState.player_who_paused) {
                     // If this client is the one waiting
                     currentPlayerIndicator.classList.add(`player-${clientPlayerInfo.symbol}`);
                 }
             } else if (currentGameState.status === 'finished') {
                 msgType = "success";
                 if (currentGameState.is_draw) {
                     message = "Game Over: It's a DRAW!";
                 } else {
                     const winner = currentGameState.players.find(p => p.symbol === currentGameState.overall_winner_symbol);
                     message = `Game Over: ${winner ? winner.name : 'Player'} (${currentGameState.overall_winner_symbol}) WINS!`;
                     if (winner) {
                         currentPlayerIndicator.classList.add(`player-${winner.symbol}`);
                     }
                 }
             } else if (currentGameState.status === 'ABORTED') { // Hard abort (no player_who_paused)
                  message = currentGameState.last_error_message || "Game Aborted.";
                  msgType = "error";
             }

             // Override message if there's a specific last_error_message not covered by status logic
             // (but prioritize status-specific messages for clarity)
             if (currentGameState.last_error_message &&
                 (msgType === "info" || (currentGameState.status === 'in_progress' && !message.includes("Error")))) { // Only show if not already an error/warning from status
                 // Don't show "Game resumed" as an error.
                 if (!currentGameState.last_error_message.toLowerCase().includes("resumed") && !currentGameState.last_error_message.toLowerCase().includes("reconnected")) {
                     // If the current message isn't already a waiting/disconnect message
                     if(!(currentGameState.status === 'ABORTED' && currentGameState.player_who_paused) && !(currentGameState.status === 'IN_PROGRESS' && message.includes("reconnect"))){
                         message = `Note: ${currentGameState.last_error_message}`;
                         // Determine if it's an error or info based on content
                         if (currentGameState.last_error_message.toLowerCase().includes("must play") ||
                             currentGameState.last_error_message.toLowerCase().includes("not your turn") ||
                             currentGameState.last_error_message.toLowerCase().includes("invalid") ||
                             currentGameState.last_error_message.toLowerCase().includes("occupied") ) {
                             msgType = "error";
                         } else {
                             msgType = "info"; // Or "warning" if appropriate
                         }
                     }
                 }
             }

             statusBar.textContent = message;
             statusBar.className = `status-bar ${msgType}`; // msgType will be info, error, success, or warning
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
