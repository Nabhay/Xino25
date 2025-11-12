"""
================================================================================
TETRIS BOT - ML CHALLENGE
================================================================================

OBJECTIVE:
----------
Build an ML/AI bot that plays Tetris by implementing the `decide()` method below.
Your bot will receive game state information and must return the action to take.

HOW IT WORKS:
-------------
1. The game calls `bot.decide(obs)` approximately every 120ms (8-9 times per second)
2. Your bot receives a dictionary (`obs`) containing complete game state
3. You analyze the state and return one action string
4. The game executes your action and continues

IMPORTS YOU MIGHT NEED:
-----------------------
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
import os
import math

try:
    import torch
    import torch.nn as nn
except Exception:  # Torch optional; we'll fallback to heuristics if unavailable
    torch = None  # type: ignore
    nn = None  # type: ignore

# Add your imports here (numpy, tensorflow, pytorch, etc.)
# import numpy as np
# import torch
# from your_model import YourModel


"""
================================================================================
IMPLEMENT YOUR BOT HERE
================================================================================
"""


class Bot:
    """
    Your Tetris Bot Implementation.
    
    Initialize your model, load weights, set up any required state in __init__().
    Implement your decision logic in the decide() method.
    """
    
    def __init__(self) -> None:
        """
        Initialize your bot here.
        
        Examples:
        ---------
        # Load a trained model
        # self.model = YourModel()
        # self.model.load_weights('model.pth')
        
        # Initialize any state tracking
        # self.move_history = []
        # self.strategy = "aggressive"
        """
        # Runtime plan: queue of actions to emit across decide() calls
        self._action_plan: List[str] = []
        # Leaky-bucket control: only plan once per new piece spawn
        self._switch_this_turn: bool = False
        self._active_piece_type: Optional[str] = None
        self._last_y: Optional[int] = None
        self._target_config: Optional[Tuple[int, int]] = None  # (target_x, target_rot_idx)
        self._model = None
        self._use_model = False
        # Async planning executor to avoid blocking decide()
        try:
            self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1)
        except Exception:
            self._executor = None
        self._pending_future: Optional[Future] = None

        # Try to initialize a small policy network matching the GA-style model
        # Input: 4 features [lines_cleared, holes, bumpiness, height] -> 1 score
        # Hidden sizes chosen to match typical GA defaults
        if torch is not None:
            class PolicyNetwork(nn.Module):
                def __init__(self, in_features: int = 4, h1: int = 32, h2: int = 32) -> None:
                    super().__init__()
                    self.fc1 = nn.Linear(in_features, h1)
                    self.fc2 = nn.Linear(h1, h2)
                    self.out = nn.Linear(h2, 1)

                def forward(self, x):  # x: [N,4]
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    x = self.out(x)
                    return x.view(-1)

            self._model = PolicyNetwork()

            # Candidate model paths (first existing will be loaded)
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            candidates = [
                os.path.join(os.path.dirname(__file__), "model.pth"),
                os.path.join(repo_root, "AML-Tetris-Reinforcement-Learning", "models", "genetic_score_1845942.pth"),
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), None)
            if model_path:
                try:
                    state = torch.load(model_path, map_location="cpu")
                    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                        state = {k.replace("module.", "", 1): v for k, v in state.items()}
                    self._model.load_state_dict(state, strict=False)
                    self._model.eval()
                    self._use_model = True
                except Exception:
                    # If load fails, we'll use heuristic fallback
                    self._use_model = False
        # If torch wasn't importable, we will use heuristic scoring

    def decide(self, obs: Optional[dict]) -> Optional[str]:
        """
        ====================================================================
        MAIN DECISION FUNCTION - IMPLEMENT YOUR BOT LOGIC HERE
        ====================================================================
        
        This method is called every ~120ms during gameplay.
        Analyze the game state and return the action you want to take.
        
        PARAMETERS:
        -----------
        obs : dict or None
            Game state observation dictionary (see structure below)
            Will be None if game state is unavailable (rare)
        
        RETURNS:
        --------
        str or None
            One of the following action strings:
            - 'w'  : Rotate piece clockwise
            - 'a'  : Move piece left
            - 's'  : Soft drop (move down faster)
            - 'd'  : Move piece right
            - ' '  : Hard drop (instant drop to bottom)
            - None : Do nothing this frame
        
        ====================================================================
        GAME STATE STRUCTURE (obs dictionary):
        ====================================================================
        
        obs = {
            "grid": List[List[int]],        # The game board
            "current_piece": dict,          # Currently falling piece
            "next_piece": dict,             # Next piece in queue
            "level": int                    # Current difficulty level
        }
        
        --------------------------------------------------------------------
        1. GRID (obs["grid"])
        --------------------------------------------------------------------
        A 20x10 matrix representing the game board (20 rows, 10 columns).
        - Index [0][0] is top-left corner
        - Index [19][9] is bottom-right corner
        - Values:
            0 = Empty cell
            1-7 = Occupied cell (different colors/piece types)
        
        Example access:
            grid = obs["grid"]
            top_row = grid[0]           # First row (list of 10 values)
            bottom_row = grid[19]       # Last row
            cell = grid[row][col]       # Individual cell value
            
        Example usage:
            # Check if bottom row is full
            if all(cell != 0 for cell in grid[19]):
                return ' '  # Hard drop!
                
            # Count empty cells in column 5
            empty_in_col5 = sum(1 for row in grid if row[5] == 0)
        
        --------------------------------------------------------------------
        2. CURRENT PIECE (obs["current_piece"])
        --------------------------------------------------------------------
        The piece currently falling. None if no piece is active.
        
        Structure:
            {
                "type": int,            # Piece type (0-6)
                                        # 0=I, 1=O, 2=T, 3=S, 4=Z, 5=J, 6=L
                
                "x": int,               # Column position (0-9)
                "y": int,               # Row position (0-19)
                
                "rotation": int,        # Rotation state (0-3)
                                        # 0=0°, 1=90°, 2=180°, 3=270°
                
                "color": int,           # Color code (1-7)
                
                "cells": List[tuple]    # Occupied cells as (row, col) pairs
                                        # Example: [(5,3), (5,4), (6,3), (6,4)]
            }
        
        Example usage:
            current = obs["current_piece"]
            if current is not None:
                # Check if piece is in danger zone (top rows)
                if current["y"] < 5:
                    return ' '  # Drop it fast!
                
                # Check if piece is too far right
                if current["x"] > 6:
                    return 'a'  # Move left
                
                # Rotate if it's an I-piece
                if current["type"] == 0:
                    return 'w'
        
        --------------------------------------------------------------------
        3. NEXT PIECE (obs["next_piece"])
        --------------------------------------------------------------------
        The next piece that will appear after current piece is placed.
        Useful for planning ahead!
        
        Structure:
            {
                "type": int,            # Piece type (0-6)
                "rotation": int,        # Initial rotation (usually 0)
                "color": int            # Color code (1-7)
            }
        
        Example usage:
            next_p = obs["next_piece"]
            if next_p["type"] == 0:  # Next is I-piece
                # Save space for vertical placement
                return 'd'  # Move to right side
        
        --------------------------------------------------------------------
        4. LEVEL (obs["level"])
        --------------------------------------------------------------------
        Current game level (difficulty increases with level).
        Higher levels = pieces fall faster
        
        Example usage:
            if obs["level"] > 5:
                # High speed - make quick decisions
                return ' '  # Hard drop
        
        ====================================================================
        EXAMPLE IMPLEMENTATIONS:
        ====================================================================
        
        Example 1: Simple rule-based bot
        ---------------------------------
        def decide(self, obs):
            if obs is None:
                return None
            
            current = obs["current_piece"]
            if current is None:
                return None
            
            # Move pieces to the left side
            if current["x"] > 3:
                return 'a'
            
            # Drop when in position
            return ' '
        
        Example 2: Grid analysis bot
        -----------------------------
        def decide(self, obs):
            if obs is None:
                return None
            
            grid = obs["grid"]
            current = obs["current_piece"]
            
            # Find the emptiest column
            col_heights = []
            for col in range(10):
                height = sum(1 for row in range(20) if grid[row][col] != 0)
                col_heights.append(height)
            
            best_col = col_heights.index(min(col_heights))
            
            # Move toward the best column
            if current["x"] < best_col:
                return 'd'
            elif current["x"] > best_col:
                return 'a'
            else:
                return ' '  # Drop!
        
        Example 3: ML model bot
        -----------------------
        def decide(self, obs):
            if obs is None:
                return None
            
            # Convert grid to numpy array
            grid_array = np.array(obs["grid"])
            
            # Get model prediction
            action_probs = self.model.predict(grid_array)
            action_idx = np.argmax(action_probs)
            
            # Map to action
            actions = ['w', 'a', 's', 'd', ' ', None]
            return actions[action_idx]
        
        ====================================================================
        TIPS & STRATEGIES:
        ====================================================================
        
        1. Start Simple: Begin with basic rules before adding ML
        2. Test Incrementally: Print debug info to see what's happening
        3. Consider Speed: Your code runs every ~120ms, keep it fast
        4. Plan Ahead: Use next_piece to inform your strategy
        5. Avoid Gaps: Try to minimize holes in your grid
        6. Clear Lines: Prioritize completing rows when possible
        7. Think Vertical: Keep some columns clear for I-pieces
        
        ====================================================================
        """
        
        # Safety check
        if obs is None:
            return None
        
        # ================================================================
        # IMPLEMENT YOUR BOT LOGIC BELOW THIS LINE
        # ================================================================
        
        # Extract game state components
        grid: List[List[int]] = obs.get("grid")  # rows x cols, 0 empty, >0 filled
        current_piece: Optional[Dict[str, Any]] = obs.get("current_piece")
        # next_piece and level are available but not required for single-move planning

        # If no active piece, clear plan and wait
        if not current_piece:
            self._action_plan.clear()
            self._target_config = None
            return None

        p_type = str(current_piece.get("type"))
        p_y = int(current_piece.get("y", 0))

        # Detect switch: new piece type or spawn reset upward
        is_switch = False
        if self._active_piece_type is None:
            is_switch = True
        elif p_type != self._active_piece_type:
            is_switch = True
        elif self._last_y is not None and (p_y < self._last_y - 3 or (p_y <= 1 and self._last_y > 1)):
            # Jump upwards or appeared at top -> new piece spawned
            is_switch = True

        # Update last seen for next call
        self._active_piece_type = p_type
        self._last_y = p_y

        if is_switch:
            # Mark switch and clear existing plan
            self._switch_this_turn = True
            self._action_plan.clear()
            self._target_config = None
            clean_grid = self._strip_current_from_grid(grid, current_piece)
            # Launch background planning (append when done)
            try:
                self._pending_future = self._executor.submit(self._plan_actions_for_piece, clean_grid, current_piece)
            except Exception:
                self._pending_future = None
            self._switch_this_turn = False

        # If a background plan finished, append its actions
        if self._pending_future is not None and self._pending_future.done():
            try:
                plan, target = self._pending_future.result()
                self._action_plan.extend(plan)
                self._target_config = target
            except Exception:
                pass
            finally:
                self._pending_future = None

        # If no plan and no compute running, start one now
        if not self._action_plan and self._pending_future is None and current_piece is not None:
            try:
                clean_grid = self._strip_current_from_grid(grid, current_piece)
                self._pending_future = self._executor.submit(self._plan_actions_for_piece, clean_grid, current_piece)
            except Exception:
                self._pending_future = None

        # Emit next planned action if available
        if self._action_plan:
            return self._action_plan.pop(0)

        # Fallback while waiting: soft drop to keep active
        return "s"

    # ------------------------------
    # Planning and scoring utilities
    # ------------------------------
    def _plan_actions_for_piece(self, grid: List[List[int]], piece: Dict[str, Any]) -> Tuple[List[str], Tuple[int, int]]:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 10

        # Enumerate all feasible placements (rotation, x) for the current piece
        piece_type: str = str(piece.get("type"))
        cur_x: int = int(piece.get("x", 0))
        cur_y: int = int(piece.get("y", 0))
        cur_rot: int = int(piece.get("rotation", 0))

        rotations = self._get_rotations_for_type(piece_type)
        features_batch: List[List[float]] = []
        placements_meta: List[Tuple[int, int, int, int]] = []  # (x, final_y, rot_index, width)

        for r_idx, cells in enumerate(rotations):
            min_col, max_col = self._piece_minmax_cols(cells)
            width = max_col - min_col + 1

            # Valid x so that piece within [0, cols-1]
            min_x = -min_col
            max_x = cols - width
            for x in range(min_x, max_x + 1):
                # Start from current y; if immediate collision at current y for the
                # rotated shape, skip this placement. This also prevents placements
                # that only work by "sliding" the piece into a gap where the
                # rotated shape overlaps while moving.
                y = cur_y
                if self._collides(grid, cells, x, y):
                    continue

                # Ensure horizontal path from current x to target x is clear
                # when the piece is in its current rotation. If the bot would
                # need to "slide" (i.e., move through positions where the
                # piece in current rotation collides), then skip this
                # placement.
                path_clear = True
                if x != cur_x:
                    step = 1 if x > cur_x else -1
                    # iterate through each step from cur_x towards x
                    xi = cur_x
                    while xi != x:
                        xi += step
                        # If the piece in its current rotation would collide at
                        # this intermediate x at the spawn y, the move is not
                        # possible without sliding in.
                        if self._collides(grid, rotations[cur_rot], xi, cur_y):
                            path_clear = False
                            break
                if not path_clear:
                    continue
                # Drop until collision
                while not self._collides(grid, cells, x, y + 1):
                    y += 1

                # Place piece and compute resulting board and features
                placed = self._place_on_board(grid, cells, x, y)
                lines_cleared, board_after = self._clear_full_rows(placed)
                holes = self._count_holes(board_after)
                bumpiness, height = self._bumpiness_and_total_height(board_after)
                features = [float(lines_cleared), float(holes), float(bumpiness), float(height)]

                features_batch.append(features)
                placements_meta.append((x, y, r_idx, width))

        # If no placements found (should be rare), do a hard drop as a fallback
        if not features_batch:
            return [" "], (cur_x, cur_rot)

        # Score placements: prefer model if available, else heuristic
        if self._use_model and torch is not None:
            with torch.no_grad():
                x_in = torch.tensor(features_batch, dtype=torch.float32)
                try:
                    scores_t = self._model(x_in)  # type: ignore[arg-type]
                    scores = scores_t.detach().cpu().tolist()
                except Exception:
                    scores = [self._heuristic_score(f) for f in features_batch]
        else:
            scores = [self._heuristic_score(f) for f in features_batch]

        # Try placements in score-descending order but validate that the
        # sequence of actions to reach the placement will actually execute
        # in the game (no rotation or move will be rejected). If a rotation
        # would collide we attempt small horizontal kicks (left/right up to
        # 2 cells) by inserting moves before the rotation. If we cannot make
        # the rotation succeed, skip this placement.
        nR = len(rotations)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])

        def validate_and_build(x_target: int, rot_target: int) -> Optional[List[str]]:
            # initial virtual state
            vx = cur_x
            vy = cur_y
            vrot = cur_rot
            plan: List[str] = []

            # horizontal moves to bring frame origin to target x
            if x_target < vx:
                moves = ["a"] * (vx - x_target)
            else:
                moves = ["d"] * (x_target - vx)

            # apply horizontal moves step-by-step, ensuring none are blocked
            for m in moves:
                dx = -1 if m == "a" else 1
                if self._collides(grid, rotations[vrot], vx + dx, vy):
                    return None
                vx += dx
                plan.append(m)

            # compute how many rotations needed
            rot_steps = (rot_target - vrot) % nR
            for _ in range(rot_steps):
                next_rot = (vrot + 1) % nR
                # if rotating in-place is possible, do it
                if not self._collides(grid, rotations[next_rot], vx, vy):
                    plan.append("w")
                    vrot = next_rot
                    continue

                # otherwise try small kicks (left/right up to 2 cells)
                kicked = False
                for kick in ( -1, 1, -2, 2 ):
                    # check path to kick position is clear in current rotation
                    step = 1 if kick > 0 else -1
                    ok = True
                    tx = vx
                    for _ in range(abs(kick)):
                        tx += step
                        if self._collides(grid, rotations[vrot], tx, vy):
                            ok = False
                            break
                    if not ok:
                        continue

                    # now check rotation at kicked x succeeds
                    if not self._collides(grid, rotations[next_rot], vx + kick, vy):
                        # insert the horizontal moves to perform the kick
                        if kick > 0:
                            plan.extend(["d"] * kick)
                        else:
                            plan.extend(["a"] * (-kick))
                        vx += kick
                        plan.append("w")
                        vrot = next_rot
                        kicked = True
                        break

                if not kicked:
                    # cannot rotate into this orientation at this x/vy
                    return None

            # finally hard drop
            plan.append(" ")
            return plan

        for idx in order:
            tx, _, trot, _ = placements_meta[idx]
            plan = validate_and_build(tx, trot)
            if plan is not None:
                return plan, (tx, trot)

        # No valid placement found that can be executed safely; fallback
        return [" "], (cur_x, cur_rot)

    def _heuristic_score(self, f: List[float]) -> float:
        # Weighted sum heuristic if no model: reward lines, penalize holes, bumpiness, height
        lines, holes, bumpiness, height = f
        return 5.0 * lines - 1.5 * holes - 0.5 * bumpiness - 0.05 * height

    # ---------------
    # Piece utilities
    # ---------------
    def _get_rotations_for_type(self, piece_type: str) -> List[List[Tuple[int, int]]]:
        """Return list of rotations as lists of (i,j) cell coordinates in 4x4 frame."""
        # Mirrors tetris.shape.version definitions
        version = {
            "I": [[1, 5, 9, 13], [4, 5, 6, 7]],
            "Z": [[4, 5, 9, 10], [2, 6, 5, 9]],
            "S": [[6, 7, 9, 10], [1, 5, 6, 10]],
            "L": [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
            "J": [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
            "T": [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
            "O": [[1, 2, 5, 6]],
        }
        idx_lists = version.get(piece_type, version["O"])  # default to O if unknown
        rotations: List[List[Tuple[int, int]]] = []
        for lst in idx_lists:
            coords = [(p // 4, p % 4) for p in lst]
            rotations.append(coords)
        return rotations

    def _piece_minmax_cols(self, cells: List[Tuple[int, int]]) -> Tuple[int, int]:
        cols = [c for (_, c) in cells]
        return (min(cols), max(cols))

    def _collides(self, board: List[List[int]], cells: List[Tuple[int, int]], x: int, y: int) -> bool:
        rows = len(board)
        cols = len(board[0]) if rows > 0 else 10
        for (i, j) in cells:
            br = y + i
            bc = x + j
            if br >= rows or bc < 0 or bc >= cols:
                return True
            if br >= 0 and board[br][bc] > 0:
                return True
        return False

    def _place_on_board(self, board: List[List[int]], cells: List[Tuple[int, int]], x: int, y: int) -> List[List[int]]:
        placed = [row[:] for row in board]
        for (i, j) in cells:
            br = y + i
            bc = x + j
            if 0 <= br < len(placed) and 0 <= bc < len(placed[0]):
                placed[br][bc] = 1  # treat as filled; color not needed for features
        return placed

    def _clear_full_rows(self, board: List[List[int]]) -> Tuple[int, List[List[int]]]:
        rows = len(board)
        cols = len(board[0]) if rows > 0 else 10
        keep = []
        cleared = 0
        for r in range(rows):
            if all(board[r][c] > 0 for c in range(cols)):
                cleared += 1
            else:
                keep.append(board[r][:])
        # Add empty rows at top for each cleared
        new_board = [[0 for _ in range(cols)] for _ in range(cleared)] + keep
        return cleared, new_board

    def _count_holes(self, board: List[List[int]]) -> int:
        rows = len(board)
        cols = len(board[0]) if rows > 0 else 10
        holes = 0
        for c in range(cols):
            seen_block = False
            for r in range(rows):
                if board[r][c] > 0:
                    seen_block = True
                elif seen_block and board[r][c] == 0:
                    holes += 1
        return holes

    def _bumpiness_and_total_height(self, board: List[List[int]]) -> Tuple[int, int]:
        rows = len(board)
        cols = len(board[0]) if rows > 0 else 10
        heights: List[int] = []
        for c in range(cols):
            h = 0
            for r in range(rows):
                if board[r][c] > 0:
                    h = rows - r
                    break
            heights.append(h)
        bump = sum(abs(heights[i] - heights[i + 1]) for i in range(cols - 1))
        total_h = sum(heights)
        return bump, total_h

    def _strip_current_from_grid(self, grid: List[List[int]], piece: Dict[str, Any]) -> List[List[int]]:
        """Return a copy of grid with the current piece cells removed (set to 0)."""
        g = [row[:] for row in grid]
        cells = piece.get("cells") or []
        for (r, c) in cells:
            if 0 <= r < len(g) and 0 <= c < len(g[0]):
                g[r][c] = 0
        return g
        
        # ================================================================
        # END OF YOUR IMPLEMENTATION
        # ================================================================