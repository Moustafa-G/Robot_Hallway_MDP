import sys
import random
from typing import Dict, List, Optional
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QIcon


@dataclass
class Move:
    """Represents a single robot movement"""
    from_state: int
    action: str
    success: bool
    to_state: int
    reward: float


class RobotHallwayMDP:
    """Core MDP logic for robot hallway problem"""
    
    def __init__(self):
        # MDP Parameters
        self.states = [0, 1, 2, 3]
        self.actions = ['LEFT', 'RIGHT']
        self.discount_factor = 0.9
        self.theta = 0.0001
        self.max_value_iters = 1000
        self.simulation_speed = 500  

        # Simulation state
        self.robot_position = 0
        self.total_reward = 0.0
        self.sim_history: List[Move] = []
        self.current_move: Optional[Move] = None
        self.sim_start_position: Optional[int] = None
        self.is_simulating = False

        # Planning results
        self.policy: Dict[int, str] = {s: 'RIGHT' if s < 3 else 'TERMINAL' for s in self.states}
        self.values: Dict[int, float] = {s: 0.0 for s in self.states}
        self.iteration_history: List[dict] = []

        # Unified history for navigation
        self.unified_history: List[dict] = []

    def get_next_state(self, state: int, action: str, actual_move: bool) -> int:
        """Calculate next state given current state and action"""
        if state == 3:
            return 3
        if not actual_move:
            return state
        if action == 'RIGHT':
            return min(state + 1, 3)
        else:
            return max(state - 1, 0)

    def get_reward(self, state: int, action: str, next_state: int) -> float:
        """Get reward for transition"""
        return 10.0 if next_state == 3 else -1.0

    def compute_value_iteration(self):
        """Perform value iteration to compute optimal policy"""
        self.iteration_history = []
        V = {s: 0.0 for s in self.states}
        policy = {s: 'RIGHT' if s < 3 else 'TERMINAL' for s in self.states}

        for it in range(self.max_value_iters):
            state_computations = {}
            delta = 0.0
            new_V = V.copy()
            new_policy = {}

            for state in self.states:
                if state == 3:
                    new_policy[state] = 'TERMINAL'
                    state_computations[state] = {
                        'old_value': V[state],
                        'new_value': 0.0,
                        'action_values': {},
                        'best_action': 'TERMINAL',
                        'is_terminal': True
                    }
                    continue

                action_values = {}
                best_value = float('-inf')
                best_action = None

                for action in self.actions:
                    s_succ = self.get_next_state(state, action, True)
                    r_succ = self.get_reward(state, action, s_succ)
                    succ_comp = 0.8 * (r_succ + self.discount_factor * V[s_succ])

                    s_fail = state
                    r_fail = self.get_reward(state, action, s_fail)
                    fail_comp = 0.2 * (r_fail + self.discount_factor * V[s_fail])

                    total = succ_comp + fail_comp

                    action_values[action] = {
                        'total': total,
                        'success_component': succ_comp,
                        'fail_component': fail_comp,
                        'next_state_success': s_succ,
                        'next_state_fail': s_fail,
                        'reward_success': r_succ,
                        'reward_fail': r_fail
                    }

                    if total > best_value:
                        best_value = total
                        best_action = action

                new_V[state] = best_value
                new_policy[state] = best_action
                delta = max(delta, abs(V[state] - new_V[state]))

                state_computations[state] = {
                    'old_value': V[state],
                    'new_value': best_value,
                    'action_values': action_values,
                    'best_action': best_action,
                    'delta': abs(V[state] - new_V[state]),
                    'is_terminal': False
                }

            self.iteration_history.append({
                'iteration': it,
                'values': new_V.copy(),
                'policy': new_policy.copy(),
                'delta': delta,
                'state_computations': state_computations
            })

            V = new_V
            policy = new_policy

            if delta < self.theta:
                break

        self.values = V
        self.policy = policy

    def start_simulation(self, start_position: int):
        """Initialize simulation from given starting position"""
        self.robot_position = start_position
        self.sim_start_position = start_position
        self.total_reward = 0.0
        self.sim_history = []
        self.current_move = None
        self.is_simulating = True

    def stop_simulation(self):
        """Stop current simulation"""
        self.is_simulating = False

    def simulate_step(self) -> Optional[Move]:
        """Execute one simulation step, returns Move or None if terminal"""
        if self.robot_position == 3:
            self.current_move = None
            self.is_simulating = False
            return None

        action = self.policy.get(self.robot_position, 'RIGHT')
        action_succeeds = random.random() < 0.8
        next_state = self.get_next_state(self.robot_position, action, action_succeeds)
        reward = self.get_reward(self.robot_position, action, next_state)

        move = Move(
            from_state=self.robot_position,
            action=action,
            success=action_succeeds,
            to_state=next_state,
            reward=reward
        )

        self.robot_position = next_state
        self.total_reward += reward
        self.sim_history.append(move)
        self.current_move = move

        if self.robot_position == 3:
            self.is_simulating = False

        return move

    def build_unified_history(self):
        """Build unified history from iterations"""
        self.unified_history = []
        for it_entry in self.iteration_history:
            self.unified_history.append({
                'kind': 'iter',
                'iter_index': it_entry['iteration'],
                'values': it_entry['values']
            })

    def append_move_to_unified_history(self, move: Move):
        """Add a move to unified history"""
        move_index = len([e for e in self.unified_history if e.get('kind') == 'move'])
        self.unified_history.append({
            'kind': 'move',
            'move_index': move_index,
            'move': move
        })


class HallwayCanvas(QWidget):
    """Custom widget for drawing the hallway visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mdp = None
        self.history_index = 0
        self.unified_history = []
        self.setMinimumSize(800, 180)
        
    def set_mdp(self, mdp: RobotHallwayMDP):
        """Set the MDP to visualize"""
        self.mdp = mdp
        
    def set_history_state(self, history_index: int, unified_history: List[dict]):
        """Update display based on history navigation"""
        self.history_index = history_index
        self.unified_history = unified_history
        self.update()

    def paintEvent(self, event):
        """Paint the hallway visualization"""
        if not self.mdp:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get current snapshot data
        if not self.mdp.iteration_history:
            self._draw_empty_hallway(painter)
            return
            
        # Determine which iteration snapshot to display
        snapshot_iter_idx = self._get_snapshot_index()
        iter_snapshot = self.mdp.iteration_history[snapshot_iter_idx]
        values = iter_snapshot['values']
        policy = iter_snapshot['policy']
        
        # Determine robot position
        robot_pos = self._get_robot_position()
        
        # Draw cells
        self._draw_cells(painter, values, policy, robot_pos)

    def _get_snapshot_index(self) -> int:
        """Get the iteration snapshot index to display"""
        if not self.unified_history:
            return len(self.mdp.iteration_history) - 1
            
        idx = max(0, min(self.history_index, len(self.unified_history) - 1))
        if self.unified_history[idx]['kind'] == 'iter':
            return self.unified_history[idx]['iter_index']
        return len(self.mdp.iteration_history) - 1

    def _get_robot_position(self) -> int:
        """Get robot position to display based on history"""
        if not self.unified_history:
            return self.mdp.robot_position
            
        idx = max(0, min(self.history_index, len(self.unified_history) - 1))
        entry = self.unified_history[idx]
        
        if entry['kind'] == 'iter':
            return self.mdp.sim_start_position if self.mdp.sim_start_position is not None else 0
        else:
            move_idx = entry['move_index']
            if move_idx < len(self.mdp.sim_history):
                return self.mdp.sim_history[move_idx].to_state
        return self.mdp.robot_position

    def _draw_empty_hallway(self, painter: QPainter):
        """Draw empty hallway when no data available"""
        cell_width, cell_height = 180, 150
        start_x, start_y = 40, 15
        
        for i, state in enumerate(self.mdp.states):
            x = start_x + i * cell_width
            color = QColor('#06d6a0') if state == 3 else QColor('#2d3748')
            
            painter.fillRect(x, start_y, cell_width - 20, cell_height, color)
            painter.setPen(QPen(QColor('#e94560'), 3))
            painter.drawRect(x, start_y, cell_width - 20, cell_height)
            
            painter.setPen(QColor('#a8b2d1'))
            painter.setFont(QFont('Arial', 12, QFont.Weight.Bold))
            painter.drawText(x + 20, start_y + 20, f"State {state}")

    def _draw_cells(self, painter: QPainter, values: dict, policy: dict, robot_pos: int):
        """Draw all hallway cells with state information"""
        cell_width, cell_height = 180, 150
        start_x, start_y = 40, 15
        
        for i, state in enumerate(self.mdp.states):
            x = start_x + i * cell_width
            
            # Cell background
            if state == 3:
                color = QColor('#06d6a0')
            elif state == robot_pos:
                color = QColor('#4a5568')
            else:
                color = QColor('#2d3748')
                
            painter.fillRect(x, start_y, cell_width - 20, cell_height, color)
            painter.setPen(QPen(QColor('#e94560'), 3))
            painter.drawRect(x, start_y, cell_width - 20, cell_height)
            
            # State label
            painter.setPen(QColor('#a8b2d1'))
            painter.setFont(QFont('Arial', 12, QFont.Weight.Bold))
            painter.drawText(x + 20, start_y + 35, f"State {state}")
            
            # Goal label
            if state == 3:
                painter.setFont(QFont('Arial', 14, QFont.Weight.Bold))
                painter.setPen(QColor('white'))
                painter.drawText(x + 20, start_y + 95, "Charge")
                painter.drawText(x + 20, start_y + 115, "Station")
            
            # Robot emoji
            if state == robot_pos:
                painter.setFont(QFont('Arial', 45))
                painter.drawText(x + cell_width//2 - 30, start_y + 90, "🤖")
            
            # Policy arrow
            if state != 3 and state != robot_pos:
                arrow = "←" if policy[state] == "LEFT" else "→"
                painter.setFont(QFont('Arial', 35, QFont.Weight.Bold))
                painter.setPen(QColor('#a8b2d1'))
                painter.drawText(x + cell_width//2 - 20, start_y + 85, arrow)
            
            # Value
            if state != 3:
                painter.setFont(QFont('Arial', 10))
                painter.setPen(QColor('#a8b2d1'))
                value_text = f"V: {values[state]:.3f}"
                painter.drawText(x + cell_width//2 - 30, start_y + 140, value_text)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.mdp = RobotHallwayMDP()
        self.mdp.compute_value_iteration()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._timer_callback)
        
        self.play_index = 0
        self.history_index = 0
        self.is_playing_iterations = False
        
        self._setup_ui()
        self._update_display()

    def _setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("🤖 Robot Hallway MDP - PyQt6")
        self.setMinimumSize(1400, 820)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QWidget { background-color: #1a1a2e; color: #a8b2d1; }
            QPushButton {
                background-color: #2d3748;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #4a5568; }
            QPushButton:pressed { background-color: #1a202c; }
            QLabel { color: #a8b2d1; }
            QTextEdit {
                background-color: #0f3460;
                color: #a8b2d1;
                border: 1px solid #2d3748;
                border-radius: 5px;
                padding: 10px;
            }
            QFrame { background-color: #16213e; }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        self._create_title(main_layout)
        
        # Content area
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # Left column
        left_widget = self._create_left_column()
        content_layout.addWidget(left_widget, stretch=2)
        
        # Right column
        right_widget = self._create_right_column()
        content_layout.addWidget(right_widget, stretch=1)

    def _create_title(self, parent_layout):
        """Create title section"""
        title_frame = QFrame()
        title_frame.setStyleSheet("QFrame { background-color: #16213e; padding: 12px; }")
        title_layout = QVBoxLayout(title_frame)
        
        title = QLabel("🤖 Robot Hallway MDP - Value Iteration Visualization")
        title.setStyleSheet("font-size: 22pt; font-weight: bold; color: #e94560;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(title)
        
        subtitle = QLabel("Click a 'Start at ...' button → robot moves automatically following the optimal policy")
        subtitle.setStyleSheet("font-size: 11pt; color: #a8b2d1;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(subtitle)
        
        parent_layout.addWidget(title_frame)

    def _create_left_column(self) -> QWidget:
        """Create left column with visualization"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Canvas
        self.canvas = HallwayCanvas()
        self.canvas.set_mdp(self.mdp)
        layout.addWidget(self.canvas)
        
        # Statistics
        stats_frame = self._create_stats_frame()
        layout.addWidget(stats_frame)
        
        # Controls
        controls_frame = self._create_controls()
        layout.addWidget(controls_frame)
        
        # History
        history_frame = self._create_history_frame()
        layout.addWidget(history_frame, stretch=1)
        
        return widget

    def _create_stats_frame(self) -> QFrame:
        """Create statistics display"""
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #16213e; border-radius: 5px; padding: 10px; }")
        layout = QHBoxLayout(frame)
        
        self.iter_label = QLabel("Value Iter: 0")
        self.iter_label.setStyleSheet("font-size: 13pt; font-weight: bold;")
        self.iter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.iter_label)
        
        self.pos_label = QLabel("Robot at: 0")
        self.pos_label.setStyleSheet("font-size: 13pt; font-weight: bold;")
        self.pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.pos_label)
        
        self.reward_label = QLabel("Total Reward: 0.0")
        self.reward_label.setStyleSheet("font-size: 13pt; font-weight: bold;")
        self.reward_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.reward_label)
        
        return frame

    def _create_controls(self) -> QWidget:
        """Create control buttons"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Navigation
        nav_layout = QHBoxLayout()
        
        btn_first = QPushButton("⏮")
        btn_first.clicked.connect(self._history_first)
        btn_first.setFixedSize(60, 50)
        nav_layout.addWidget(btn_first)
        
        btn_prev = QPushButton("◀")
        btn_prev.clicked.connect(self._history_prev)
        btn_prev.setFixedSize(60, 50)
        nav_layout.addWidget(btn_prev)
        
        self.history_label = QLabel("0/0")
        self.history_label.setStyleSheet("font-size: 13pt; font-weight: bold; color: #e94560;")
        self.history_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_label.setMinimumWidth(80)
        nav_layout.addWidget(self.history_label)
        
        btn_next = QPushButton("▶")
        btn_next.clicked.connect(self._history_next)
        btn_next.setFixedSize(60, 50)
        nav_layout.addWidget(btn_next)
        
        btn_last = QPushButton("⏭")
        btn_last.clicked.connect(self._history_last)
        btn_last.setFixedSize(60, 50)
        nav_layout.addWidget(btn_last)
        
        layout.addLayout(nav_layout)
        
        # Start buttons
        start_layout = QHBoxLayout()
        
        for i in range(3):
            btn = QPushButton(f"⏳ Start at {i}")
            btn.setStyleSheet("QPushButton { background-color: #06d6a0; }")
            btn.clicked.connect(lambda checked, pos=i: self._play_full_sequence(pos))
            start_layout.addWidget(btn)
        
        layout.addLayout(start_layout)
        
        return widget

    def _create_history_frame(self) -> QWidget:
        """Create history display"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel("History")
        label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #e94560;")
        layout.addWidget(label)
        
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setFont(QFont('Courier', 10))
        layout.addWidget(self.history_text)
        
        return widget

    def _create_right_column(self) -> QWidget:
        """Create right column with iteration details"""
        widget = QFrame()
        widget.setStyleSheet("QFrame { background-color: #16213e; border-radius: 5px; }")
        layout = QVBoxLayout(widget)
        
        title = QLabel("Value Iteration Trace & Details")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #e94560; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setFont(QFont('Courier', 9))
        layout.addWidget(self.details_text)
        
        return widget

    def _update_display(self):
        """Update all display elements"""
        self.canvas.set_history_state(self.history_index, self.mdp.unified_history)
        self._update_stats()
        self._update_details()
        self._update_history_text()
        self._update_history_label()

    def _update_stats(self):
        """Update statistics labels"""
        if self.mdp.iteration_history:
            snapshot_idx = self._get_snapshot_index()
            iter_data = self.mdp.iteration_history[snapshot_idx]
            self.iter_label.setText(f"Value Iter: {iter_data['iteration']}")
        
        self.pos_label.setText(f"Robot at: {self.mdp.robot_position}")
        self.reward_label.setText(f"Total Reward: {self.mdp.total_reward:.1f}")

    def _get_snapshot_index(self) -> int:
        """Get current iteration snapshot index"""
        if not self.mdp.unified_history:
            return len(self.mdp.iteration_history) - 1 if self.mdp.iteration_history else 0
            
        idx = max(0, min(self.history_index, len(self.mdp.unified_history) - 1))
        if self.mdp.unified_history[idx]['kind'] == 'iter':
            return self.mdp.unified_history[idx]['iter_index']
        return len(self.mdp.iteration_history) - 1

    def _update_details(self):
        """Update iteration details panel"""
        if not self.mdp.iteration_history:
            return
            
        snapshot_idx = self._get_snapshot_index()
        iter_data = self.mdp.iteration_history[snapshot_idx]
        
        html = self._generate_details_html(iter_data)
        self.details_text.setHtml(html)

    def _generate_details_html(self, iter_data: dict) -> str:
        """Generate HTML for iteration details"""
        html = '<div style="font-family: Courier; color: #a8b2d1;">'
        html += f'<h3 style="color: #e94560;">ITERATION {iter_data["iteration"]}</h3>'
        html += '<hr style="border-color: #4a5568;">'
        
        html += '<h4 style="color: #06d6a0;">BELLMAN OPTIMALITY UPDATE:</h4>'
        
        state_comps = iter_data['state_computations']
        for state in self.mdp.states:
            if state == 3:
                continue
                
            comp = state_comps[state]
            html += f'<p><strong style="color: #ffd700;">State {state}:</strong></p>'
            html += f'<p style="margin-left: 20px;">Old V[{state}] = {comp["old_value"]:.6f}</p>'
            html += '<p style="margin-left: 20px;">Evaluating actions:</p>'
            
            for action in ['LEFT', 'RIGHT']:
                action_val = comp['action_values'][action]
                arrow = "←" if action == "LEFT" else "→"
                
                html += f'<p style="margin-left: 40px; color: #87ceeb;">{arrow} {action}:</p>'
                html += f'<p style="margin-left: 60px;">Success (p=0.8): s\'={action_val["next_state_success"]}, r={action_val["reward_success"]:.1f}</p>'
                html += f'<p style="margin-left: 80px; color: #98d8c8;">= {action_val["success_component"]:.6f}</p>'
                html += f'<p style="margin-left: 60px;">Fail (p=0.2): s\'={action_val["next_state_fail"]}, r={action_val["reward_fail"]:.1f}</p>'
                html += f'<p style="margin-left: 80px; color: #98d8c8;">= {action_val["fail_component"]:.6f}</p>'
                html += f'<p style="margin-left: 60px; color: #98d8c8;">TOTAL Q({state},{action}) = {action_val["total"]:.6f}</p>'
            
            best_action = comp['best_action']
            arrow = "←" if best_action == "LEFT" else "→"
            html += f'<p style="margin-left: 20px; color: #ff6b9d;"><strong>{arrow} Best action: {best_action}</strong></p>'
            html += f'<p style="margin-left: 20px; color: #ff6b9d;"><strong>New V[{state}] = {comp["new_value"]:.6f}</strong></p>'
            html += f'<p style="margin-left: 20px; color: #98d8c8;">Change: {comp["delta"]:.6f}</p>'
            html += '<hr style="border-color: #2d3748;">'
        
        html += '<h4 style="color: #06d6a0;">UPDATED POLICY:</h4>'
        for state in self.mdp.states:
            if state == 3:
                continue
            action = iter_data['policy'][state]
            arrow = "←" if action == "LEFT" else "→"
            html += f'<p style="margin-left: 20px;">π[{state}] = {arrow} {action}</p>'
        
        html += f'<p style="color: #98d8c8;">Max Delta = {iter_data["delta"]:.6f}</p>'
        html += f'<p style="color: #98d8c8;">Threshold = {self.mdp.theta:.6f}</p>'
        
        if iter_data['delta'] < self.mdp.theta:
            html += '<p style="color: #ff6b9d;"><strong>✓ CONVERGED!</strong></p>'
        else:
            html += '<p>→ Continue iterating...</p>'
        
        html += '<hr style="border-color: #4a5568;">'
        html += '<h4 style="color: #06d6a0;">LATEST ROBOT MOVEMENT:</h4>'
        
        if self.mdp.current_move:
            move = self.mdp.current_move
            success = "✓" if move.success else "✗"
            arrow = "←" if move.action == "LEFT" else "→"
            html += f'<p>From State: {move.from_state}</p>'
            html += f'<p>Action: {arrow} {move.action}</p>'
            html += f'<p style="color: #ff6b9d;"><strong>Result: {success} {"Success" if move.success else "Failed"}</strong></p>'
            html += f'<p>To State: {move.to_state}</p>'
            html += f'<p style="color: #98d8c8;">Reward: {move.reward:.1f}</p>'
        else:
            if self.mdp.robot_position == 3:
                html += '<p style="color: #ff6b9d;">Robot at goal (no movement)</p>'
            else:
                html += '<p>No simulated move yet</p>'
        
        html += '</div>'
        return html

    def _update_history_text(self):
        """Update history text area"""
        if not self.mdp.unified_history:
            self.history_text.clear()
            return
            
        idx = max(0, min(self.history_index, len(self.mdp.unified_history) - 1))
        text = ""
        
        for i in range(idx + 1):
            entry = self.mdp.unified_history[i]
            if entry['kind'] == 'iter':
                iter_idx = entry['iter_index']
                vals = entry['values']
                vals_str = ", ".join([f"{s}:{vals[s]:.2f}" for s in sorted(vals.keys())])
                text += f"[ITER {iter_idx}]  V = {{{vals_str}}}\n"
            else:
                mv = entry['move']
                success = "✓" if mv.success else "✗"
                arrow = "←" if mv.action == "LEFT" else "→"
                reward = f"+{mv.reward}" if mv.reward > 0 else str(mv.reward)
                move_index = entry['move_index']
                text += f"[STEP {move_index}] From {mv.from_state} {arrow} {mv.action} {success} → {mv.to_state}  (R: {reward})\n"
        
        self.history_text.setPlainText(text)
        self.history_text.verticalScrollBar().setValue(
            self.history_text.verticalScrollBar().maximum()
        )

    def _update_history_label(self):
        """Update history navigation label"""
        total = len(self.mdp.unified_history)
        if total == 0:
            self.history_label.setText("0/0")
        else:
            self.history_label.setText(f"{self.history_index}/{total-1}")

    # History navigation methods
    def _history_first(self):
        """Jump to first history entry"""
        if self.mdp.unified_history:
            self.history_index = 0
            self._update_display()

    def _history_prev(self):
        """Go to previous history entry"""
        if self.mdp.unified_history and self.history_index > 0:
            self.history_index -= 1
            self._update_display()

    def _history_next(self):
        """Go to next history entry"""
        if self.mdp.unified_history and self.history_index < len(self.mdp.unified_history) - 1:
            self.history_index += 1
            self._update_display()

    def _history_last(self):
        """Jump to last history entry"""
        if self.mdp.unified_history:
            self.history_index = len(self.mdp.unified_history) - 1
            self._update_display()

    # Simulation control methods
    def _play_full_sequence(self, start_position: int):
        """Start playing value iteration then simulation"""
        self.timer.stop()
        
        # Recompute and prepare
        self.mdp.compute_value_iteration()
        self.mdp.build_unified_history()
        self.mdp.start_simulation(start_position)
        
        # Reset playback state
        self.play_index = 0
        self.history_index = 0
        self.is_playing_iterations = True
        
        self._update_display()
        
        # Start timer for value iteration playback
        self.timer.start(self.mdp.simulation_speed)

    def _timer_callback(self):
        """Timer callback for automatic playback"""
        if self.is_playing_iterations:
            # Playing through value iteration snapshots
            if self.play_index < len(self.mdp.iteration_history):
                self.history_index = self.play_index
                self.play_index += 1
                self._update_display()
            else:
                # Finished iterations, start robot simulation
                self.is_playing_iterations = False
                self._step_simulation()
        else:
            # Playing robot simulation
            self._step_simulation()

    def _step_simulation(self):
        """Execute one simulation step"""
        if not self.mdp.is_simulating:
            self.timer.stop()
            self._update_display()
            return
        
        move = self.mdp.simulate_step()
        
        if move:
            self.mdp.append_move_to_unified_history(move)
            self.history_index = len(self.mdp.unified_history) - 1
        
        self._update_display()
        
        if not self.mdp.is_simulating:
            self.timer.stop()


def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look across platforms
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()