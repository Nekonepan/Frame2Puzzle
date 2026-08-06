import random
import cv2
import numpy as np


class PuzzlePiece:
    """Represents an individual puzzle piece slice."""

    def __init__(self, piece_id, correct_row, correct_col, image_slice):
        self.piece_id = piece_id
        self.correct_row = correct_row
        self.correct_col = correct_col

        self.current_row = correct_row
        self.current_col = correct_col

        self.image_slice = image_slice
        self.is_placed = False

        # Screen bounding box coordinates (x1, y1, x2, y2)
        self.bounds = (0, 0, 0, 0)


class PuzzleManager:
    """Manages puzzle generation, shuffling, grid rendering, drag interaction, and state evaluation."""

    def __init__(self, rows=3, cols=3):
        self.rows = rows
        self.cols = cols
        self.num_pieces = rows * cols

        self.pieces = []
        self.grid_slots = []  # Target grid slot positions on screen
        self.is_generated = False
        self.is_solved = False

        # Board layout dimensions on screen
        self.board_x = 0
        self.board_y = 0
        self.board_w = 0
        self.board_h = 0
        self.cell_w = 0
        self.cell_h = 0

        # --- Phase 6: Drag Interaction State ---
        self.dragged_piece = None       # Currently grabbed PuzzlePiece reference
        self.drag_cursor_x = 0          # Current pinch cursor X on screen
        self.drag_cursor_y = 0          # Current pinch cursor Y on screen
        self.drag_origin_row = -1       # Row where the piece was picked up from
        self.drag_origin_col = -1       # Col where the piece was picked up from
        self.hover_slot = (-1, -1)      # Grid slot (row, col) currently hovered by drag cursor

    # ------------------------------------------------------------------ #
    #                    PUZZLE GENERATION & SHUFFLE                      #
    # ------------------------------------------------------------------ #

    def generate_puzzle(self, image, board_area=None):
        """Splits the input image into grid pieces and shuffles their initial grid positions."""
        img_h, img_w = image.shape[:2]

        self.pieces = []
        piece_h = img_h // self.rows
        piece_w = img_w // self.cols

        # 1. Slice image into pieces
        piece_id = 0
        for r in range(self.rows):
            for c in range(self.cols):
                y1, y2 = r * piece_h, (r + 1) * piece_h
                x1, x2 = c * piece_w, (c + 1) * piece_w

                slice_img = image[y1:y2, x1:x2].copy()
                piece = PuzzlePiece(piece_id, r, c, slice_img)
                self.pieces.append(piece)
                piece_id += 1

        # 2. Shuffle piece positions
        self.shuffle_puzzle()
        self.is_generated = True
        self.is_solved = False

        # Reset drag state on new puzzle
        self.dragged_piece = None
        self.hover_slot = (-1, -1)

        print(f"\n[PUZZLE ENGINE] Puzzle successfully generated ({self.rows}x{self.cols} = {self.num_pieces} pieces) and shuffled!")

    def shuffle_puzzle(self):
        """Shuffles the current grid positions of all puzzle pieces."""
        grid_positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]

        # Ensure initial state is shuffled (not already solved)
        while True:
            shuffled_positions = grid_positions.copy()
            random.shuffle(shuffled_positions)

            # Check if at least some pieces are moved
            mismatches = sum(
                1 for i, pos in enumerate(shuffled_positions)
                if pos != (self.pieces[i].correct_row, self.pieces[i].correct_col)
            )
            if mismatches >= (self.num_pieces // 2):
                break

        for i, piece in enumerate(self.pieces):
            piece.current_row, piece.current_col = shuffled_positions[i]
            piece.is_placed = (piece.current_row == piece.correct_row) and (piece.current_col == piece.correct_col)

    # ------------------------------------------------------------------ #
    #                         BOARD LAYOUT                                #
    # ------------------------------------------------------------------ #

    def calculate_board_layout(self, frame_w, frame_h):
        """Calculates board grid coordinates and cell dimensions on screen."""
        margin_top = 100
        margin_bottom = 40
        available_h = frame_h - margin_top - margin_bottom
        available_w = frame_w - 60

        cell_size = min(available_w // self.cols, available_h // self.rows)
        self.cell_w = cell_size
        self.cell_h = cell_size

        self.board_w = self.cell_w * self.cols
        self.board_h = self.cell_h * self.rows
        self.board_x = (frame_w - self.board_w) // 2
        self.board_y = margin_top + (available_h - self.board_h) // 2

        # Update target grid slot bounds
        self.grid_slots = []
        for r in range(self.rows):
            row_slots = []
            for c in range(self.cols):
                sx1 = self.board_x + c * self.cell_w
                sy1 = self.board_y + r * self.cell_h
                sx2 = sx1 + self.cell_w
                sy2 = sy1 + self.cell_h
                row_slots.append((sx1, sy1, sx2, sy2))
            self.grid_slots.append(row_slots)

    def check_solved(self):
        """Evaluates whether all pieces are placed in their correct target grid slots."""
        solved_count = 0
        for piece in self.pieces:
            if piece.current_row == piece.correct_row and piece.current_col == piece.correct_col:
                piece.is_placed = True
                solved_count += 1
            else:
                piece.is_placed = False

        self.is_solved = (solved_count == self.num_pieces)
        return self.is_solved, solved_count

    # ------------------------------------------------------------------ #
    #               PHASE 6: DRAG, SNAP & COLLISION SWAP                 #
    # ------------------------------------------------------------------ #

    def _get_piece_at_grid(self, row, col):
        """Returns the PuzzlePiece currently occupying the given grid slot, or None."""
        for piece in self.pieces:
            if piece.current_row == row and piece.current_col == col:
                return piece
        return None

    def _get_piece_at_screen(self, sx, sy):
        """Returns the PuzzlePiece whose on-screen bounding box contains the point (sx, sy)."""
        for piece in self.pieces:
            x1, y1, x2, y2 = piece.bounds
            if x1 <= sx <= x2 and y1 <= sy <= y2:
                return piece
        return None

    def _screen_to_grid(self, sx, sy):
        """Converts screen coordinates to the nearest grid slot (row, col). Returns (-1,-1) if off-grid."""
        if not self.grid_slots:
            return -1, -1

        for r in range(self.rows):
            for c in range(self.cols):
                gx1, gy1, gx2, gy2 = self.grid_slots[r][c]
                if gx1 <= sx <= gx2 and gy1 <= sy <= gy2:
                    return r, c
        return -1, -1

    def handle_pinch_start(self, cursor_x, cursor_y):
        """Called when a PINCH gesture begins. Picks up the tile under the cursor.

        Args:
            cursor_x: Pinch center X in screen coordinates.
            cursor_y: Pinch center Y in screen coordinates.
        """
        if self.is_solved or self.dragged_piece is not None:
            return

        target_piece = self._get_piece_at_screen(cursor_x, cursor_y)
        if target_piece is None:
            return

        self.dragged_piece = target_piece
        self.drag_origin_row = target_piece.current_row
        self.drag_origin_col = target_piece.current_col
        self.drag_cursor_x = cursor_x
        self.drag_cursor_y = cursor_y

        print(f"[DRAG] Picked up piece #{target_piece.piece_id + 1} from grid ({target_piece.current_row},{target_piece.current_col})")

    def handle_pinch_move(self, cursor_x, cursor_y):
        """Called every frame while PINCH is held. Updates the dragged tile position.

        Args:
            cursor_x: Current pinch center X in screen coordinates.
            cursor_y: Current pinch center Y in screen coordinates.
        """
        if self.dragged_piece is None:
            return

        self.drag_cursor_x = cursor_x
        self.drag_cursor_y = cursor_y

        # Determine which grid slot the cursor is currently hovering over
        self.hover_slot = self._screen_to_grid(cursor_x, cursor_y)

    def handle_pinch_release(self):
        """Called when the PINCH gesture ends. Snaps the tile to the nearest grid slot and swaps if occupied.

        Returns:
            True if a swap was performed, False otherwise.
        """
        if self.dragged_piece is None:
            return False

        piece = self.dragged_piece
        target_row, target_col = self._screen_to_grid(self.drag_cursor_x, self.drag_cursor_y)

        did_swap = False

        # Valid drop target: within grid bounds
        if target_row >= 0 and target_col >= 0:
            # Check if different from origin slot
            if target_row != self.drag_origin_row or target_col != self.drag_origin_col:
                # Collision Check: find the piece occupying the target slot
                occupant = self._get_piece_at_grid(target_row, target_col)

                if occupant is not None:
                    # SWAP: move occupant to the dragged piece's origin slot
                    occupant.current_row = self.drag_origin_row
                    occupant.current_col = self.drag_origin_col
                    print(f"[SWAP] Piece #{occupant.piece_id + 1} moved to ({self.drag_origin_row},{self.drag_origin_col})")

                # Snap dragged piece to target slot
                piece.current_row = target_row
                piece.current_col = target_col
                did_swap = True
                print(f"[SNAP] Piece #{piece.piece_id + 1} placed at ({target_row},{target_col})")
            else:
                # Dropped on origin — snap back, no swap
                print(f"[SNAP] Piece #{piece.piece_id + 1} returned to origin ({self.drag_origin_row},{self.drag_origin_col})")
        else:
            # Off-grid drop — snap back to origin
            print(f"[SNAP] Off-grid drop. Piece #{piece.piece_id + 1} returned to origin.")

        # Reset drag state
        self.dragged_piece = None
        self.hover_slot = (-1, -1)
        self.drag_origin_row = -1
        self.drag_origin_col = -1

        # Re-evaluate solved state after potential swap
        self.check_solved()
        if self.is_solved:
            print("\n[PUZZLE ENGINE] *** PUZZLE SOLVED! Congratulations! ***")

        return did_swap

    # ------------------------------------------------------------------ #
    #                           RENDERING                                #
    # ------------------------------------------------------------------ #

    def render_puzzle(self, display_frame):
        """Renders the puzzle board grid, piece slices, drag visuals, and HUD onto the display frame."""
        if not self.is_generated:
            return display_frame

        h, w = display_frame.shape[:2]
        self.calculate_board_layout(w, h)

        # 1. Board Background
        cv2.rectangle(
            display_frame,
            (self.board_x - 6, self.board_y - 6),
            (self.board_x + self.board_w + 6, self.board_y + self.board_h + 6),
            (40, 40, 40),
            -1,
        )
        cv2.rectangle(
            display_frame,
            (self.board_x - 6, self.board_y - 6),
            (self.board_x + self.board_w + 6, self.board_y + self.board_h + 6),
            (0, 255, 255) if not self.is_solved else (0, 255, 0),
            3,
        )

        # 2. Render hover highlight on target grid slot (when dragging)
        if self.dragged_piece is not None and self.hover_slot != (-1, -1):
            hr, hc = self.hover_slot
            hx1, hy1, hx2, hy2 = self.grid_slots[hr][hc]
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (0, 200, 255), -1)
            cv2.addWeighted(overlay, 0.25, display_frame, 0.75, 0, display_frame)

        # 3. Render stationary pieces (skip the currently dragged piece)
        for piece in self.pieces:
            if piece is self.dragged_piece:
                continue  # Render dragged piece last (on top)

            r, c = piece.current_row, piece.current_col
            sx1, sy1, sx2, sy2 = self.grid_slots[r][c]

            # Resize piece image slice to fit target cell size
            piece_img = cv2.resize(piece.image_slice, (self.cell_w, self.cell_h))
            display_frame[sy1:sy2, sx1:sx2] = piece_img

            # Store screen bounding box for interaction
            piece.bounds = (sx1, sy1, sx2, sy2)

            # Draw border around each piece
            border_color = (0, 255, 0) if piece.is_placed else (200, 200, 200)
            thickness = 2 if piece.is_placed else 1
            cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), border_color, thickness)

            # Render piece ID badge if not yet solved
            if not self.is_solved:
                badge_text = f"#{piece.piece_id + 1}"
                cv2.putText(
                    display_frame, badge_text, (sx1 + 10, sy1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
                )

        # 4. Render the dragged piece floating at cursor position (on top layer)
        if self.dragged_piece is not None:
            dp = self.dragged_piece
            piece_img = cv2.resize(dp.image_slice, (self.cell_w, self.cell_h))

            # Center the tile image on the drag cursor
            half_w = self.cell_w // 2
            half_h = self.cell_h // 2
            dx1 = self.drag_cursor_x - half_w
            dy1 = self.drag_cursor_y - half_h
            dx2 = dx1 + self.cell_w
            dy2 = dy1 + self.cell_h

            # Clamp to frame boundaries to prevent out-of-bounds writes
            src_x1 = max(0, -dx1)
            src_y1 = max(0, -dy1)
            src_x2 = self.cell_w - max(0, dx2 - w)
            src_y2 = self.cell_h - max(0, dy2 - h)
            dst_x1 = max(0, dx1)
            dst_y1 = max(0, dy1)
            dst_x2 = min(w, dx2)
            dst_y2 = min(h, dy2)

            if dst_x2 > dst_x1 and dst_y2 > dst_y1:
                # Draw shadow behind the floating piece
                shadow_offset = 6
                cv2.rectangle(
                    display_frame,
                    (dst_x1 + shadow_offset, dst_y1 + shadow_offset),
                    (dst_x2 + shadow_offset, dst_y2 + shadow_offset),
                    (20, 20, 20), -1,
                )

                # Draw the floating piece image
                display_frame[dst_y1:dst_y2, dst_x1:dst_x2] = piece_img[src_y1:src_y2, src_x1:src_x2]

                # Bright cyan highlight border for the dragged piece
                cv2.rectangle(display_frame, (dst_x1, dst_y1), (dst_x2, dst_y2), (0, 255, 255), 3)

                # Piece ID badge on dragged tile
                badge_text = f"#{dp.piece_id + 1}"
                cv2.putText(
                    display_frame, badge_text, (dst_x1 + 10, dst_y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
                )

            # Also render origin slot as dimmed placeholder
            or_x1, or_y1, or_x2, or_y2 = self.grid_slots[self.drag_origin_row][self.drag_origin_col]
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (or_x1, or_y1), (or_x2, or_y2), (80, 80, 80), -1)
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
            cv2.rectangle(display_frame, (or_x1, or_y1), (or_x2, or_y2), (0, 200, 255), 2)

            # Update dragged piece bounds to cursor position for next frame detection
            dp.bounds = (dx1, dy1, dx2, dy2)

        # 5. Top Banner HUD
        _, solved_count = self.check_solved()
        banner_color = (0, 128, 0) if self.is_solved else (30, 30, 30)
        cv2.rectangle(display_frame, (0, 0), (w, 75), banner_color, -1)
        cv2.line(display_frame, (0, 75), (w, 75), (0, 255, 0), 2)

        if self.is_solved:
            cv2.putText(
                display_frame,
                "CONGRATULATIONS! PUZZLE SOLVED!",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "Press 'r' to Play Again",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            status_text = "DRAGGING..." if self.dragged_piece is not None else f"Progress: {solved_count}/{self.num_pieces} Pieces"
            cv2.putText(
                display_frame,
                f"PUZZLE BOARD ({self.rows}x{self.cols}) - {status_text}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            hint_text = (
                "Move your PINCH to target slot, then release to swap!"
                if self.dragged_piece is not None
                else "Use PINCH Gesture to pick and swap puzzle pieces!"
            )
            cv2.putText(
                display_frame,
                hint_text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return display_frame
