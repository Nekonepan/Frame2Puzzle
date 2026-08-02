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
    """Manages puzzle generation (slicing), shuffling, grid rendering, and state evaluation."""

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

    def calculate_board_layout(self, frame_w, frame_h):
        """Calculates board grid coordinates and cell dimensions on screen."""
        margin_top = 100
        margin_bottom = 40
        available_h = frame_h - margin_top - margin_bottom
        available_w = frame_w - 60

        # Maintain 16:9 or 4:3 aspect ratio matching the grid
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

    def render_puzzle(self, display_frame):
        """Renders the puzzle board grid and piece slices onto the display frame."""
        if not self.is_generated:
            return display_frame

        h, w = display_frame.shape[:2]
        self.calculate_board_layout(w, h)

        # 1. Render Puzzle Board Grid Background
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

        # 2. Render Grid Slots and Puzzle Pieces
        for piece in self.pieces:
            r, c = piece.current_row, piece.current_col
            sx1, sy1, sx2, sy2 = self.grid_slots[r][c]

            # Resize piece image slice to fit target cell size
            piece_img = cv2.resize(piece.image_slice, (self.cell_w, self.cell_h))

            # Draw piece onto display frame
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
                    display_frame,
                    badge_text,
                    (sx1 + 10, sy1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # 3. Render Top Banner Info
        _, solved_count = self.check_solved()
        banner_color = (0, 128, 0) if self.is_solved else (30, 30, 30)
        cv2.rectangle(display_frame, (0, 0), (w, 75), banner_color, -1)
        cv2.line(display_frame, (0, 75), (w, 75), (0, 255, 0), 2)

        if self.is_solved:
            cv2.putText(
                display_frame,
                "🎉 CONGRATULATIONS! PUZZLE SOLVED!",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "Show OPEN PALM Gesture or Press 'r' to Play Again",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                display_frame,
                f"PUZZLE BOARD ({self.rows}x{self.cols}) - Progress: {solved_count}/{self.num_pieces} Pieces",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "Use PINCH Gesture to pick and swap puzzle pieces!",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return display_frame
