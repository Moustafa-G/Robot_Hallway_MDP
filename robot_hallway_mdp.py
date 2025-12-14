import tkinter as tk
from tkinter import ttk
import random
from typing import Dict, Tuple, List, Optional

class RobotHallwayMDP:
    def __init__(self):
        # MDP Parameters
        self.states = [0, 1, 2, 3]
        self.actions = ['LEFT', 'RIGHT']
        self.discount_factor = 0.9
        self.theta = 0.0001
        self.max_value_iters = 1000

        # Simulation state (for episodes)
        self.robot_position = 0
        self.total_reward = 0
        self.sim_history: List[dict] = []   # movement history during simulation
        self.current_move: Optional[dict] = None  # latest move (for UI)
        self.sim_start_position: Optional[int] = None  # starting position for current sim

        # Results of planning
        self.policy: Dict[int, str] = {s: 'RIGHT' if s < 3 else 'TERMINAL' for s in self.states}
        self.values: Dict[int, float] = {s: 0.0 for s in self.states}

        # Value iteration trace (for the iteration viewer)
        # each entry: {'iteration': i, 'values': {...}, 'policy': {...}, 'delta': d, 'state_computations': {...}}
        self.iteration_history: List[dict] = []

        # Simulation controls
        self.is_simulating = False
        self.simulation_speed = 700  # milliseconds between automatic steps

    def get_next_state(self, state: int, action: str, actual_move: bool) -> int:
        if state == 3:
            return 3
        if not actual_move:
            return state
        if action == 'RIGHT':
            return min(state + 1, 3)
        else:
            return max(state - 1, 0)

    def get_reward(self, state: int, action: str, next_state: int) -> float:
        return 10.0 if next_state == 3 else -1.0

    def compute_value_iteration(self):
        """Perform value iteration to compute optimal values and policy.
           Stores iteration snapshots in self.iteration_history for the UI.
        """
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
                    # deterministic "success" next state based on action
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

            # store snapshot for UI
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

        # save final results into object
        self.values = V
        self.policy = policy

    def start_simulation(self, start_position: int):
        """Prepare simulation (one episode) using the computed optimal policy"""
        # Reset episode variables
        self.robot_position = start_position
        self.sim_start_position = start_position
        self.total_reward = 0.0
        self.sim_history = []
        self.current_move = None
        self.is_simulating = True

    def stop_simulation(self):
        self.is_simulating = False

    def simulate_step(self) -> Optional[dict]:
        """Simulate one step using the computed policy.
           Returns move_data dict if a move happened, otherwise None (already at goal).
        """
        if self.robot_position == 3:
            # already terminal
            self.current_move = None
            self.is_simulating = False
            return None

        action = self.policy.get(self.robot_position, 'RIGHT')
        # probabilistic outcome: 0.8: succeed (move), 0.2: stay
        action_succeeds = random.random() < 0.8
        next_state = self.get_next_state(self.robot_position, action, action_succeeds)
        reward = self.get_reward(self.robot_position, action, next_state)

        move_data = {
            'from_state': self.robot_position,
            'action': action,
            'success': action_succeeds,
            'to_state': next_state,
            'reward': reward
        }

        # update episode state
        self.robot_position = next_state
        self.total_reward += reward
        self.sim_history.append(move_data)
        self.current_move = move_data

        # stop if terminal
        if self.robot_position == 3:
            self.is_simulating = False

        return move_data


class MDPGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Synchronized Robot Hallway MDP (Fixed)")
        self.root.geometry("1400x820")
        self.root.configure(bg='#1a1a2e')

        self.mdp = RobotHallwayMDP()
        # generate the value-iteration trace (policy + values)
        self.mdp.compute_value_iteration()

        self.after_id = None

        # Unified history (list of dicts). Each element has type:
        # {'kind': 'iter', 'iter_index': i, 'values': {...}}  OR
        # {'kind': 'move', 'move_index': j, 'move': {...}}
        self.unified_history: List[dict] = []
        self.history_index = 0  # current index into unified_history when navigating
        self.play_index = 0     # used for playing value-iteration frames
        self.total_history_len = 0

        self.create_widgets()
        self.update_display()

    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#16213e', pady=12)
        title_frame.pack(fill='x')

        title = tk.Label(title_frame, text="🤖 Robot Hallway MDP - Auto Simulation (Value Iteration)",
                         font=('Arial', 22, 'bold'), bg='#16213e', fg='#e94560')
        title.pack()

        subtitle = tk.Label(title_frame, text="Click a 'Start at ...' button → robot moves automatically following the optimal policy",
                            font=('Arial', 11), bg='#16213e', fg='#a8b2d1')
        subtitle.pack()

        # Main container with two columns
        main_container = tk.Frame(self.root, bg='#1a1a2e')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Left column - Main visualization
        left_frame = tk.Frame(main_container, bg='#1a1a2e')
        left_frame.pack(side='left', fill='both', expand=True)

        # Hallway Canvas
        canvas_frame = tk.Frame(left_frame, bg='#1a1a2e', pady=10)
        canvas_frame.pack()

        self.canvas = tk.Canvas(canvas_frame, width=800, height=180,
                                bg='#16213e', highlightthickness=0)
        self.canvas.pack()

        # Statistics
        stats_frame = tk.Frame(left_frame, bg='#1a1a2e', pady=10)
        stats_frame.pack()

        stats_container = tk.Frame(stats_frame, bg='#16213e', relief='raised', bd=2)
        stats_container.pack()

        self.iter_num_label = tk.Label(stats_container, text="Value Iter: 0",
                                    font=('Arial', 13, 'bold'), bg='#16213e',
                                    fg='#a8b2d1', width=18, pady=8)
        self.iter_num_label.grid(row=0, column=0, padx=8)

        self.robot_pos_label = tk.Label(stats_container, text="Robot at: 0",
                                     font=('Arial', 13, 'bold'), bg='#16213e',
                                     fg='#a8b2d1', width=18, pady=8)
        self.robot_pos_label.grid(row=0, column=1, padx=8)

        self.total_reward_label = tk.Label(stats_container, text="Total Reward: 0.0",
                                       font=('Arial', 13, 'bold'), bg='#16213e',
                                       fg='#a8b2d1', width=18, pady=8)
        self.total_reward_label.grid(row=0, column=2, padx=8)

        # Iteration controls (REPLACED with unified history navigation)
        iter_control = tk.Frame(left_frame, bg='#1a1a2e', pady=12)
        iter_control.pack()

        # New unified history navigation bar (replaces the old iteration viewer bar)
        nav_frame = tk.Frame(iter_control, bg='#1a1a2e')
        nav_frame.pack()

        tk.Button(nav_frame, text="⏮", command=self.history_first,
                 font=('Arial', 11, 'bold'), bg='#2d3748', fg='white',
                 width=6, height=2).pack(side='left', padx=4)

        tk.Button(nav_frame, text="◀", command=self.history_prev,
                 font=('Arial', 11, 'bold'), bg='#2d3748', fg='white',
                 width=6, height=2).pack(side='left', padx=4)

        self.history_label = tk.Label(nav_frame, text="0/0",
                                      font=('Arial', 13, 'bold'), bg='#1a1a2e',
                                      fg='#e94560', width=8)
        self.history_label.pack(side='left', padx=8)

        tk.Button(nav_frame, text="▶", command=self.history_next,
                 font=('Arial', 11, 'bold'), bg='#2d3748', fg='white',
                 width=6, height=2).pack(side='left', padx=4)

        tk.Button(nav_frame, text="⏭", command=self.history_last,
                 font=('Arial', 11, 'bold'), bg='#2d3748', fg='white',
                 width=6, height=2).pack(side='left', padx=4)

        # Reset / Start controls (start simulation)
        reset_frame = tk.Frame(iter_control, bg='#1a1a2e')
        reset_frame.pack(side='left', padx=12)

        tk.Button(reset_frame, text="⟳ Start at 0", command=lambda: self.play_full_sequence(0),
                 font=('Arial', 11, 'bold'), bg='#06d6a0', fg='white',
                 width=12, height=2).pack(side='left', padx=4)

        tk.Button(reset_frame, text="⟳ Start at 1", command=lambda: self.play_full_sequence(1),
                 font=('Arial', 11, 'bold'), bg='#06d6a0', fg='white',
                 width=12, height=2).pack(side='left', padx=4)

        tk.Button(reset_frame, text="⟳ Start at 2", command=lambda: self.play_full_sequence(2),
                 font=('Arial', 11, 'bold'), bg='#06d6a0', fg='white',
                 width=12, height=2).pack(side='left', padx=4)

        # Note: Stop button removed entirely per request

        # History (unified) area
        history_frame = tk.Frame(left_frame, bg='#1a1a2e', pady=10)
        history_frame.pack(fill='both', expand=True)

        tk.Label(history_frame, text="History",
                font=('Arial', 12, 'bold'),
                bg='#1a1a2e', fg='#e94560').pack()

        history_container = tk.Frame(history_frame, bg='#16213e', relief='sunken', bd=2)
        history_container.pack(fill='both', expand=True, pady=5)

        scrollbar = tk.Scrollbar(history_container)
        scrollbar.pack(side='right', fill='y')

        self.history_text = tk.Text(history_container, height=6, width=80,
                                   bg='#0f3460', fg='#a8b2d1',
                                   font=('Courier', 10),
                                   yscrollcommand=scrollbar.set,
                                   relief='flat')
        self.history_text.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.history_text.yview)

        # Right column - Iteration viewer
        right_frame = tk.Frame(main_container, bg='#16213e', relief='raised', bd=2)
        right_frame.pack(side='right', fill='both', padx=(10, 0))

        tk.Label(right_frame, text="Value Iteration Trace & Details",
                font=('Arial', 14, 'bold'), bg='#16213e', fg='#e94560',
                pady=10).pack()

        # Iteration details
        details_container = tk.Frame(right_frame, bg='#16213e')
        details_container.pack(fill='both', expand=True, padx=10, pady=10)

        scrollbar2 = tk.Scrollbar(details_container)
        scrollbar2.pack(side='right', fill='y')

        self.iter_details = tk.Text(details_container, width=60,
                                    bg='#0f3460', fg='#a8b2d1',
                                    font=('Courier', 9), relief='flat',
                                    padx=10, pady=10,
                                    yscrollcommand=scrollbar2.set,
                                    wrap='word')
        self.iter_details.pack(side='left', fill='both', expand=True)
        scrollbar2.config(command=self.iter_details.yview)

    def draw_hallway(self):
        self.canvas.delete('all')

        # If no iteration history available, just draw plain hallway
        if not self.mdp.iteration_history:
            # draw empty cells
            cell_width = 180
            cell_height = 150
            start_x = 40
            start_y = 15
            for i, state in enumerate(self.mdp.states):
                x = start_x + i * cell_width
                color = '#06d6a0' if state == 3 else '#2d3748'
                self.canvas.create_rectangle(x, start_y, x + cell_width - 20,
                                            start_y + cell_height,
                                            fill=color, outline='#e94560', width=3)
                self.canvas.create_text(x + 20, start_y + 20, text=f"State {state}",
                                       font=('Arial', 12, 'bold'), fill='#a8b2d1',
                                       anchor='nw')
            return

        # Use the currently-selected iteration snapshot for values/policy display
        # We'll decide which iteration snapshot to use based on history index:
        if self.unified_history:
            # determine which iteration snapshot to show on right panel:
            # If history_index points to an 'iter' item, show that iter's snapshot.
            # If it points to a 'move' (robot step), show the final iteration snapshot (last iter).
            idx = max(0, min(self.history_index, len(self.unified_history) - 1))
            snapshot_iter_idx = None
            if self.unified_history[idx]['kind'] == 'iter':
                snapshot_iter_idx = self.unified_history[idx]['iter_index']
            else:
                # show final iteration snapshot if we're in the move portion
                if self.mdp.iteration_history:
                    snapshot_iter_idx = len(self.mdp.iteration_history) - 1

            if snapshot_iter_idx is None:
                # fallback to final snapshot
                snapshot_iter_idx = len(self.mdp.iteration_history) - 1 if self.mdp.iteration_history else 0

            iter_snapshot = self.mdp.iteration_history[snapshot_iter_idx]
            values = iter_snapshot['values']
            policy = iter_snapshot['policy']
        else:
            # no unified history yet, use final iteration snapshot as default
            iter_snapshot = self.mdp.iteration_history[-1] if self.mdp.iteration_history else {'values': {s: 0.0 for s in self.mdp.states}, 'policy': {s: 'RIGHT' for s in self.mdp.states}}
            values = iter_snapshot['values']
            policy = iter_snapshot['policy']

        # Determine robot position to draw based on current history_index:
        robot_pos_to_draw = self.mdp.robot_position
        if self.unified_history:
            idx = max(0, min(self.history_index, len(self.unified_history) - 1))
            entry = self.unified_history[idx]
            if entry['kind'] == 'iter':
                # during iteration snapshots: show the sim_start_position if available, else 0
                robot_pos_to_draw = self.mdp.sim_start_position if self.mdp.sim_start_position is not None else 0
            else:
                # move entry: show the 'to_state' of that move
                move_idx = entry['move_index']
                # if move exists in sim_history, fetch its to_state; else fallback
                if move_idx < len(self.mdp.sim_history):
                    robot_pos_to_draw = self.mdp.sim_history[move_idx]['to_state']
                else:
                    robot_pos_to_draw = self.mdp.robot_position

        robot_pos = robot_pos_to_draw

        cell_width = 180
        cell_height = 150
        start_x = 40
        start_y = 15

        for i, state in enumerate(self.mdp.states):
            x = start_x + i * cell_width

            # Cell background
            if state == 3:
                color = '#06d6a0'
            elif state == robot_pos:
                color = '#4a5568'  # Highlight current position
            else:
                color = '#2d3748'

            self.canvas.create_rectangle(x, start_y, x + cell_width - 20,
                                        start_y + cell_height,
                                        fill=color, outline='#e94560', width=3)

            # State number
            self.canvas.create_text(x + 20, start_y + 20, text=f"State {state}",
                                   font=('Arial', 12, 'bold'), fill='#a8b2d1',
                                   anchor='nw')

            # Goal
            if state == 3:
                self.canvas.create_text(x + cell_width//2 - 10, start_y + 95,
                                       text="Charge Station", font=('Arial', 14, 'bold'),
                                       fill='white')

            # Robot at current position
            if state == robot_pos:
                self.canvas.create_text(x + cell_width//2 - 10, start_y + 60,
                                       text="🤖", font=('Arial', 45))

            # Policy arrow (for non-terminal, non-robot positions)
            if state != 3 and state != robot_pos:
                arrow = "←" if policy[state] == "LEFT" else "→"
                self.canvas.create_text(x + cell_width//2 - 10, start_y + 70,
                                       text=arrow, font=('Arial', 35, 'bold'),
                                       fill='#a8b2d1')

            # Value
            if state != 3:
                value_text = f"V: {values[state]:.3f}"
                self.canvas.create_text(x + cell_width//2 - 10, start_y + 140,
                                       text=value_text, font=('Arial', 10),
                                       fill='#a8b2d1')

    def update_iteration_display(self):
        # This function now shows the Bellman breakdown for the currently-selected iteration snapshot.
        if not self.mdp.iteration_history:
            return

        # Determine which iteration snapshot to show on the right panel:
        idx = max(0, min(self.history_index, len(self.unified_history) - 1)) if self.unified_history else len(self.mdp.iteration_history) - 1
        snapshot_iter_idx = None
        if self.unified_history:
            if self.unified_history[idx]['kind'] == 'iter':
                snapshot_iter_idx = self.unified_history[idx]['iter_index']
            else:
                snapshot_iter_idx = len(self.mdp.iteration_history) - 1
        else:
            snapshot_iter_idx = len(self.mdp.iteration_history) - 1

        snapshot_iter_idx = max(0, min(snapshot_iter_idx, len(self.mdp.iteration_history) - 1))
        iter_data = self.mdp.iteration_history[snapshot_iter_idx]
        total_iters = len(self.mdp.iteration_history)

        # Update labels (iteration number uses the snapshot)
        self.iter_num_label.config(text=f"Value Iter: {iter_data['iteration']}")
        self.robot_pos_label.config(text=f"Robot at: {self.mdp.robot_position}")
        self.total_reward_label.config(text=f"Total Reward: {self.mdp.total_reward:.1f}")

        # Update details (value-iteration breakdown)
        self.iter_details.delete(1.0, tk.END)

        # Header
        self.iter_details.insert(tk.END, "="*50 + "\n", 'separator')
        self.iter_details.insert(tk.END, f"ITERATION {iter_data['iteration']}\n", 'header')
        self.iter_details.insert(tk.END, "="*50 + "\n\n", 'separator')

        # Bellman Updates
        state_comps = iter_data['state_computations']

        self.iter_details.insert(tk.END, "BELLMAN OPTIMALITY UPDATE:\n", 'subheader')
        self.iter_details.insert(tk.END, "-"*50 + "\n\n", 'separator')

        for state in self.mdp.states:
            if state == 3:
                continue

            comp = state_comps[state]

            self.iter_details.insert(tk.END, f"State {state}:\n", 'state_header')
            self.iter_details.insert(tk.END, f"  Old V[{state}] = {comp['old_value']:.6f}\n", 'value')
            self.iter_details.insert(tk.END, f"\n  Evaluating actions:\n")

            # Show each action's computation
            for action in ['LEFT', 'RIGHT']:
                action_val = comp['action_values'][action]
                arrow = "←" if action == "LEFT" else "→"

                self.iter_details.insert(tk.END, f"\n    {arrow} {action}:\n", 'action')
                self.iter_details.insert(tk.END,
                    f"      Success (p=0.8): s'={action_val['next_state_success']}, "
                    f"r={action_val['reward_success']:.1f}\n")
                self.iter_details.insert(tk.END,
                    f"        = {action_val['success_component']:.6f}\n", 'value')

                self.iter_details.insert(tk.END,
                    f"      Fail (p=0.2): s'={action_val['next_state_fail']}, "
                    f"r={action_val['reward_fail']:.1f}\n")
                self.iter_details.insert(tk.END,
                    f"        = {action_val['fail_component']:.6f}\n", 'value')

                self.iter_details.insert(tk.END,
                    f"      TOTAL Q({state},{action}) = {action_val['total']:.6f}\n", 'value')

            # Show max and new value
            best_action = comp['best_action']
            arrow = "←" if best_action == "LEFT" else "→"
            self.iter_details.insert(tk.END, f"\n  {arrow} Best action: {best_action}\n", 'highlight')
            self.iter_details.insert(tk.END, f"  New V[{state}] = {comp['new_value']:.6f}\n", 'highlight')
            self.iter_details.insert(tk.END, f"  Change: {comp['delta']:.6f}\n\n", 'value')
            self.iter_details.insert(tk.END, "-"*50 + "\n", 'separator')

        # Updated Policy
        self.iter_details.insert(tk.END, "\nUPDATED POLICY:\n", 'subheader')
        for state in self.mdp.states:
            if state == 3:
                continue
            action = iter_data['policy'][state]
            arrow = "←" if action == "LEFT" else "→"
            self.iter_details.insert(tk.END, f"  π[{state}] = {arrow} {action}\n")

        # Convergence info
        self.iter_details.insert(tk.END, f"\nMax Delta = {iter_data['delta']:.6f}\n", 'value')
        self.iter_details.insert(tk.END, f"Threshold = {self.mdp.theta:.6f}\n", 'value')
        if iter_data['delta'] < self.mdp.theta:
            self.iter_details.insert(tk.END, "✓ CONVERGED!\n\n", 'highlight')
        else:
            self.iter_details.insert(tk.END, "→ Continue iterating...\n\n")

        # Robot Move (show the latest simulated move from mdp.current_move)
        self.iter_details.insert(tk.END, "="*50 + "\n", 'separator')
        self.iter_details.insert(tk.END, "LATEST ROBOT MOVEMENT (simulation):\n", 'subheader')
        self.iter_details.insert(tk.END, "="*50 + "\n\n", 'separator')

        if self.mdp.current_move:
            move = self.mdp.current_move
            success = "✓" if move['success'] else "✗"
            arrow = "←" if move['action'] == "LEFT" else "→"
            self.iter_details.insert(tk.END, f"From State: {move['from_state']}\n")
            self.iter_details.insert(tk.END, f"Action: {arrow} {move['action']}\n")
            self.iter_details.insert(tk.END, f"Result: {success} {'Success' if move['success'] else 'Failed'}\n", 'highlight')
            self.iter_details.insert(tk.END, f"To State: {move['to_state']}\n")
            self.iter_details.insert(tk.END, f"Reward: {move['reward']:.1f}\n", 'value')
        else:
            if self.mdp.robot_position == 3:
                self.iter_details.insert(tk.END, "Robot at goal (no movement)\n", 'highlight')
            else:
                self.iter_details.insert(tk.END, "No simulated move yet\n", 'value')

        # Configure tags
        self.iter_details.tag_config('header', foreground='#e94560', font=('Courier', 11, 'bold'))
        self.iter_details.tag_config('separator', foreground='#4a5568')
        self.iter_details.tag_config('subheader', foreground='#06d6a0', font=('Courier', 10, 'bold'))
        self.iter_details.tag_config('state_header', foreground='#ffd700', font=('Courier', 9, 'bold'))
        self.iter_details.tag_config('action', foreground='#87ceeb')
        self.iter_details.tag_config('highlight', foreground='#ff6b9d', font=('Courier', 9, 'bold'))
        self.iter_details.tag_config('value', foreground='#98d8c8')

        # Update movement/history text area from unified history (show up to current index)
        self.update_history_text_upto(self.history_index)

    def update_display(self):
        # redraw elements
        self.draw_hallway()
        self.update_iteration_display()
        # update history label / counters
        self.update_history_label()

    # History / unified timeline helpers
    def build_unified_history_from_iterations(self):
        """Create unified_history entries corresponding to current iteration_history.
           Called when we recompute value iteration or at simulation start.
        """
        self.unified_history = []
        for it_entry in self.mdp.iteration_history:
            entry = {
                'kind': 'iter',
                'iter_index': it_entry['iteration'],
                'values': it_entry['values']
            }
            self.unified_history.append(entry)

    def append_move_to_unified_history(self, move):
        """Append a robot move (dict) to unified_history."""
        move_index = len([e for e in self.unified_history if e.get('kind') == 'move'])
        entry = {
            'kind': 'move',
            'move_index': move_index,
            'move': move
        }
        self.unified_history.append(entry)

    def update_history_text_upto(self, idx):
        """Renders the history_text to include all unified_history entries up to index idx (inclusive)."""
        if not self.unified_history:
            self.history_text.delete(1.0, tk.END)
            return

        idx = max(0, min(idx, len(self.unified_history) - 1))
        self.history_text.delete(1.0, tk.END)

        for i in range(0, idx + 1):
            entry = self.unified_history[i]
            if entry['kind'] == 'iter':
                iter_idx = entry['iter_index']
                vals = entry['values']
                # format: [ITER 0]  V = {0:0.00, 1:0.00, 2:0.00, 3:0.00}
                vals_str = ", ".join([f"{s}:{vals[s]:.2f}" for s in sorted(vals.keys())])
                line = f"[ITER {iter_idx}]  V = {{{vals_str}}}\n"
                self.history_text.insert(tk.END, line)
            else:
                mv = entry['move']
                success = "✓" if mv['success'] else "✗"
                arrow = "←" if mv['action'] == "LEFT" else "→"
                reward = f"+{mv['reward']}" if mv['reward'] > 0 else str(mv['reward'])
                move_index = entry['move_index']
                line = f"[STEP {move_index}] From {mv['from_state']} {arrow} {mv['action']} {success} → {mv['to_state']}  (R: {reward})\n"
                self.history_text.insert(tk.END, line)

        # Auto-scroll to the end of shown portion for convenience
        self.history_text.see(tk.END)

    def update_history_label(self):
        total = len(self.unified_history)
        if total == 0:
            self.history_label.config(text="0/0")
        else:
            # show 1-based index like 1/total but we'll keep 0-based display per your requested format "0/12"
            display_idx = self.history_index
            self.history_label.config(text=f"{display_idx}/{total-1}")

    # History navigation commands (unified)
    def history_first(self):
        if not self.unified_history:
            return
        self.history_index = 0
        self.sync_ui_to_history_index()

    def history_prev(self):
        if not self.unified_history:
            return
        if self.history_index > 0:
            self.history_index -= 1
            self.sync_ui_to_history_index()

    def history_next(self):
        if not self.unified_history:
            return
        if self.history_index < len(self.unified_history) - 1:
            self.history_index += 1
            self.sync_ui_to_history_index()

    def history_last(self):
        if not self.unified_history:
            return
        self.history_index = len(self.unified_history) - 1
        self.sync_ui_to_history_index()

    def sync_ui_to_history_index(self):
        """Update canvas, right panel and history text to reflect history_index."""
        # Update canvas robot position depending on entry at history_index
        if not self.unified_history:
            return

        idx = max(0, min(self.history_index, len(self.unified_history) - 1))
        entry = self.unified_history[idx]
        if entry['kind'] == 'iter':
            # show sim_start_position (or 0) as robot position during iteration snapshots
            pos = self.mdp.sim_start_position if self.mdp.sim_start_position is not None else 0
            # do not modify actual mdp.robot_position (we want simulation state preserved)
            # but for drawing we set a temporary attribute for draw_hallway to pick up (we already implemented reading unified_history)
        else:
            # move entry - show the to_state of that move
            move_idx = entry['move_index']
            if move_idx < len(self.mdp.sim_history):
                pos = self.mdp.sim_history[move_idx]['to_state']
            else:
                pos = self.mdp.robot_position

        # After updating internal state used by draw_hallway, refresh displays
        self.update_display()

    # Iteration trace navigation (legacy kept but not used in GUI)
    def first_iteration(self):
        self.mdp.display_iteration_index = 0
        self.update_display()

    def prev_iteration(self):
        idx = getattr(self.mdp, 'display_iteration_index', len(self.mdp.iteration_history)-1)
        if idx > 0:
            self.mdp.display_iteration_index = idx - 1
            self.update_display()

    def next_iteration(self):
        idx = getattr(self.mdp, 'display_iteration_index', len(self.mdp.iteration_history)-1)
        if idx < len(self.mdp.iteration_history) - 1:
            self.mdp.display_iteration_index = idx + 1
            self.update_display()

    def last_iteration(self):
        self.mdp.display_iteration_index = len(self.mdp.iteration_history) - 1
        self.update_display()

    def play_full_sequence(self, start_position):
        """Play value-iteration iterations then start robot movement automatically.
           Also build unified history (iterations first, then moves).
        """

        # Cancel if something was running
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except:
                pass
            self.after_id = None

        # Compute value iteration fully and reset robot
        self.mdp.compute_value_iteration()
        # Build unified history from iterations immediately
        self.build_unified_history_from_iterations()
        # Reset any previous sim history and start simulation
        self.mdp.start_simulation(start_position)

        # reset play counters
        self.play_index = 0                        # for value-iteration steps
        self.total_iters = len(self.mdp.iteration_history)
        self.history_index = 0
        self.total_history_len = len(self.unified_history)  # will grow as moves are appended

        # show the first iteration snapshot and update UI
        self.update_display()

        # Start playing value iteration steps (visualization), then simulate moves
        self.after_id = self.root.after(self.mdp.simulation_speed, self.play_value_iterations)

    def play_value_iterations(self):
        """Show value-iteration steps one by one, then switch to robot movement."""
        if self.play_index < len(self.mdp.iteration_history):
            # advance the history_index to show this iteration snapshot
            self.history_index = self.play_index
            self.play_index += 1
            self.update_display()
            self.after_id = self.root.after(self.mdp.simulation_speed, self.play_value_iterations)
        else:
            # Finished value-iteration visualization; start robot movement animation
            # ensure unified_history currently contains all iteration entries (it does)
            self.after_id = self.root.after(self.mdp.simulation_speed, self.step_simulation)

    # Simulation controls
    def reset_mdp(self, start_position: int):
        """Called when user clicks 'Start at X' — compute policy (if not done) and start sim."""
        # if value-iteration hasn't been computed, compute it (but typically done at init)
        if not self.mdp.iteration_history:
            self.mdp.compute_value_iteration()

        # cancel any running auto callbacks
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        # Prepare and start simulation from start_position
        self.mdp.start_simulation(start_position)
        # show the final value-iteration snapshot while simulating
        self.mdp.display_iteration_index = len(self.mdp.iteration_history) - 1
        self.update_display()
        # Start auto-play loop
        self.after_id = self.root.after(self.mdp.simulation_speed, self.step_simulation)

    def step_simulation(self):
        """Advance one simulated step and schedule next until terminal.
           Each move is appended to unified_history as it happens.
        """
        if not self.mdp.is_simulating:
            # nothing to do (maybe already at goal). ensure UI updated
            self.update_display()
            self.after_id = None
            return

        move = self.mdp.simulate_step()

        # Append move to unified_history and update counters
        if move:
            self.append_move_to_unified_history(move)
            self.total_history_len = len(self.unified_history)
            # set history_index to point to the newest appended move so UI moves along
            self.history_index = len(self.unified_history) - 1

        # Update UI immediately
        self.update_display()

        # If still simulating and not terminal, schedule next
        if self.mdp.is_simulating:
            self.after_id = self.root.after(self.mdp.simulation_speed, self.step_simulation)
        else:
            # reached terminal or stopped
            self.after_id = None
            self.update_display()

    def stop_simulation(self):
        """Immediate stop (user-initiated). Note: Stop button removed from UI but keep method for completeness."""
        self.mdp.stop_simulation()
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        self.update_display()


def main():
    root = tk.Tk()
    app = MDPGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
