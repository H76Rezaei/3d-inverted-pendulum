"""
3D Inverted Pendulum with MPC Controller
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time

from mpc_controller_casadi import NonlinearMPC


class SimplePendulum3D:
    """3D Inverted Pendulum with MPC controller"""
    
    def __init__(self):
        # System parameters
        self.M = 2.0        # Cart mass [kg]
        self.m = 1.0        # Pendulum mass [kg]
        self.L = 1.5        # Pendulum length [m]
        self.g = 9.81       # Gravity [m/s^2]
        self.b_theta = 0.1  # Damping for theta [N*m*s/rad]
        self.b_phi = 0.1    # Damping for phi [N*m*s/rad]
        
        # Control forces
        self.Fx = 0.0
        self.Fy = 0.0
        
        # State: [x_c, x_c_dot, y_c, y_c_dot, theta, theta_dot, phi, phi_dot]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, np.pi - 0.1, 0.0, 0.0, 0.0])
        
        # Simulation parameters
        self.dt = 0.01
        self.time = 0.0
        self.paused = False
        
        # MPC reference state (θ=π is upright)
        self.x_reference = np.array([0.0, 0.0, 0.0, 0.0, np.pi, 0.0, 0.0, 0.0])
        
        # Trajectory parameters
        self.target_position = np.array([0.0, 0.0])
        self.smoothing_factor = 0.12
        self.position_tolerance = 0.08
        
        # Noise and disturbance parameters
        self.noise_enabled = False
        self.noise_std = 0.0
        self.disturbance_enabled = False
        self.disturbance_magnitude = 0.0
        self.disturbance_probability = 0.0
        
        # Initialize MPC controller
        self.mpc = NonlinearMPC(
            N=20,
            dt=0.01,
            M=self.M,
            m=self.m,
            L=self.L,
            g=9.81,
            b_theta=self.b_theta,
            b_phi=self.b_phi,
            u_max=150.0,
            u_min=-150.0
        )
        
        # Performance tracking
        self.mpc_failures = 0
        self.mpc_attempts = 0
        self.last_u_opt = None

    def derivatives(self, state, u):
        """Compute derivatives matching MPC controller dynamics"""
        try:
            x, x_dot, y, y_dot, theta, theta_dot, phi, phi_dot = state
            Fx, Fy = u
            
            # Apply deadzone to control inputs
            Fx = 0.0 if abs(Fx) < 0.1 else Fx
            Fy = 0.0 if abs(Fy) < 0.1 else Fy
            
            # Trigonometric terms
            st, ct = np.sin(theta), np.cos(theta)
            sp, cp = np.sin(phi), np.cos(phi)
            
            # 4x4 mass matrix
            M_total = self.M + self.m
            mL = self.m * self.L
            mL2 = self.m * self.L**2
            
            M = np.array([
                [M_total, 0, mL * ct * cp, -mL * st * sp],
                [0, M_total, mL * ct * sp, mL * st * cp],
                [mL*ct * cp, mL*ct * sp, mL2, 0],
                [-mL*st * sp, mL*st * cp, 0, mL2 * st**2]
            ])
            
            # Right-hand side
            f = np.array([
                Fx + mL * (theta_dot**2 * st * cp + phi_dot**2 * st * cp - 2 * theta_dot * phi_dot * ct * sp),
                Fy + mL * (theta_dot**2 * st * sp + phi_dot**2 * st * sp + 2 * theta_dot * phi_dot * ct * cp),
                -self.m * self.g * self.L * st - self.b_theta * theta_dot,
                -self.m * self.g * self.L * st * sp - self.b_phi * phi_dot
            ])
            
            # Solve for accelerations
            try:
                eps = 1e-8
                M_reg = M + np.eye(4) * eps
                x_ddot, y_ddot, theta_ddot, phi_ddot = np.linalg.solve(M_reg, f)
            except np.linalg.LinAlgError:
                x_ddot, y_ddot, theta_ddot, phi_ddot = np.linalg.pinv(M) @ f
            
            # Clip accelerations
            max_accel = 50.0
            x_ddot = np.clip(x_ddot, -max_accel, max_accel)
            y_ddot = np.clip(y_ddot, -max_accel, max_accel)
            theta_ddot = np.clip(theta_ddot, -20.0, 20.0)
            phi_ddot = np.clip(phi_ddot, -20.0, 20.0)
            
            return np.array([x_dot, x_ddot, y_dot, y_ddot, 
                           theta_dot, theta_ddot, phi_dot, phi_ddot])
        
        except Exception as e:
            return np.zeros_like(state)

    def compute_control(self):
        """Compute MPC control"""
        self.mpc_attempts += 1
        
        # Get current state and reference
        x_current = self.state.copy()
        x_reference = self.x_reference.copy()
        
        # Set position reference to target
        x_reference[0] = self.target_position[0]
        x_reference[2] = self.target_position[1]
        x_reference[1] = 0.0
        x_reference[3] = 0.0
        
        # Keep angle references for stability
        x_reference[4] = np.pi
        x_reference[5] = 0.0
        x_reference[7] = 0.0
        
        # Handle angle wrapping for theta
        theta_diff = x_current[4] - x_reference[4]
        if abs(theta_diff) > np.pi:
            if theta_diff > 0:
                x_current[4] = x_reference[4] + theta_diff - 2 * np.pi
            else:
                x_current[4] = x_reference[4] + theta_diff + 2 * np.pi
        
        # Handle angle wrapping for phi
        phi_diff = x_current[6] - x_reference[6]
        if abs(phi_diff) > np.pi:
            if phi_diff > 0:
                x_current[6] = x_reference[6] + phi_diff - 2 * np.pi
            else:
                x_current[6] = x_reference[6] + phi_diff + 2 * np.pi
        
        try:
            x_reference[6] = x_current[6]
            t0 = time.time()
            u, success, info = self.mpc.solve(x_current, x_reference)
            
            if not success:
                if 'error' in info:
                    print(f"MPC warning: {info['error']}")
                u = np.array([
                    -0.1 * self.state[1],
                    -0.1 * self.state[3]
                ])
            
            # Apply low-pass filter to control inputs
            if hasattr(self, 'prev_u'):
                alpha = 0.3
                u = alpha * u + (1 - alpha) * self.prev_u
            self.prev_u = u.copy()
                    
        except Exception as e:
            print(f"Error in MPC solve: {e}")
            u = np.zeros(2)
            
        # Apply control with limits
        self.Fx = float(np.clip(u[0], -150.0, 150.0))
        self.Fy = float(np.clip(u[1], -150.0, 150.0))
        
        if not success:
            self.mpc_failures += 1
            self._pd_fallback(x_current)

    def _pd_fallback(self, x_current):
        """PD fallback controller"""
        # Handle angle wrapping for theta
        theta = x_current[4]
        theta_diff = theta - np.pi
        if abs(theta_diff) > np.pi:
            if theta_diff > 0:
                theta_err = theta_diff - 2 * np.pi
            else:
                theta_err = theta_diff + 2 * np.pi
        else:
            theta_err = theta_diff
        
        # Handle angle wrapping for phi
        phi = x_current[6]
        phi_diff = phi - np.pi
        if abs(phi_diff) > np.pi:
            if phi_diff > 0:
                phi_err = phi_diff - 2 * np.pi
            else:
                phi_err = phi_diff + 2 * np.pi
        else:
            phi_err = phi_diff
        
        # Energy terms
        E_k = 0.5 * self.m * (x_current[5]**2)
        E_p = self.m * self.g * self.L * (1 + np.cos(theta))
        E_d = 2 * self.m * self.g * self.L
        E_err = E_d - (E_k + E_p)
        
        # Energy injection
        energy_injection = 0.0
        if abs(theta_err) < 0.0005:
            energy_threshold = 0.005 * E_d
            energy_gain = 0.5
        elif abs(theta_err) < 0.01:
            energy_threshold = 0.01 * E_d
            energy_gain = 0.3
        elif abs(theta_err) < 0.05:
            energy_threshold = 0.05 * E_d
            energy_gain = 0.2
        else:
            energy_threshold = 0.1 * E_d
            energy_gain = 0.1
            
        if abs(E_err) > energy_threshold:
            energy_injection = energy_gain * E_err * np.sign(x_current[5] * np.cos(theta))

        # Angle control gains
        if abs(theta_err) < 0.0002:
            kp_theta = 1000.0
            kd_theta = 100.0
        elif abs(theta_err) < 0.001:
            kp_theta = 12000.0
            kd_theta = 600.0
            kp_phi = 3500.0
            kd_phi = 350.0
        elif abs(theta_err) < 0.01:
            kp_theta = 8000.0
            kd_theta = 400.0
            kp_phi = 2500.0
            kd_phi = 250.0
        elif abs(theta_err) < 0.03:
            kp_theta = 6000.0
            kd_theta = 300.0
            kp_phi = 2000.0
            kd_phi = 200.0
        elif abs(theta_err) < 0.1:
            kp_theta = 5000.0
            kd_theta = 250.0
            kp_phi = 1800.0
            kd_phi = 180.0
        else:
            kp_theta = 8000.0
            kd_theta = 400.0
            kp_phi = 3000.0
            kd_phi = 300.0

        # Calculate control forces
        Fx = kp_theta * theta_err - kd_theta * x_current[5] + energy_injection
        Fy = kp_phi * phi_err - kd_phi * x_current[7]
        
        # Bias correction
        if abs(theta_err) < 0.001:
            bias_correction = 0.0
            if theta_err > 0:
                bias_correction = -200.0 * theta_err
            elif theta_err < 0:
                bias_correction = -100.0 * theta_err
            Fx += bias_correction
        
        # Position control
        pos_x_err = self.target_position[0] - x_current[0]
        pos_y_err = self.target_position[1] - x_current[2]
        
        if abs(theta_err) > 0.02:
            pos_force_x = 0.0
            pos_force_y = 0.0
            damping_factor = 2.0
            Fx -= damping_factor * 40.0 * x_current[1]
            Fy -= damping_factor * 40.0 * x_current[3]
        elif abs(theta_err) > 0.002:
            kp_pos = 30.0
            kd_pos = 6.0
            max_pos_force = 15.0
            
            pos_force_x = kp_pos * pos_x_err - kd_pos * x_current[1]
            pos_force_y = kp_pos * pos_y_err - kd_pos * x_current[3]
            
            pos_force_x = np.clip(pos_force_x, -max_pos_force, max_pos_force)
            pos_force_y = np.clip(pos_force_y, -max_pos_force, max_pos_force)
            
            Fx -= 12.0 * x_current[1]
            Fy -= 12.0 * x_current[3]
        elif abs(theta_err) > 0.0005:
            kp_pos = 30.0
            kd_pos = 6.0
            max_pos_force = 12.0
            
            pos_force_x = kp_pos * pos_x_err - kd_pos * x_current[1]
            pos_force_y = kp_pos * pos_y_err - kd_pos * x_current[3]
            
            pos_force_x = np.clip(pos_force_x, -max_pos_force, max_pos_force)
            pos_force_y = np.clip(pos_force_y, -max_pos_force, max_pos_force)
            
            Fx -= 2.0 * x_current[1]
            Fy -= 2.0 * x_current[3]
        else:
            kp_pos = 15.0
            kd_pos = 3.0
            max_pos_force = 6.0
            
            pos_force_x = kp_pos * pos_x_err - kd_pos * x_current[1]
            pos_force_y = kp_pos * pos_y_err - kd_pos * x_current[3]
            
            pos_force_x = np.clip(pos_force_x, -max_pos_force, max_pos_force)
            pos_force_y = np.clip(pos_force_y, -max_pos_force, max_pos_force)
            
            Fx -= 1.0 * x_current[1]
            Fy -= 1.0 * x_current[3]
        
        # Add position forces
        Fx += pos_force_x
        Fy += pos_force_y
        
        # Apply force limits
        self.Fx = float(np.clip(Fx, -150.0, 150.0))
        self.Fy = float(np.clip(Fy, -150.0, 150.0))

    def integrate_step(self):
        """Integration step with RK4"""
        if self.paused:
            return
        
        # Compute control
        self.compute_control()

        # RK4 integration
        k1 = self.derivatives(self.state, [self.Fx, self.Fy])
        k2 = self.derivatives(self.state + 0.5 * self.dt * k1, [self.Fx, self.Fy])
        k3 = self.derivatives(self.state + 0.5 * self.dt * k2, [self.Fx, self.Fy])
        k4 = self.derivatives(self.state + self.dt * k3, [self.Fx, self.Fy])

        self.state += self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Normalize angles
        if not (np.pi - 0.1 < abs(self.state[4]) < np.pi + 0.1):
            self.state[4] = (self.state[4] + np.pi) % (2 * np.pi) - np.pi
        if not (np.pi - 0.1 < abs(self.state[6]) < np.pi + 0.1):
            self.state[6] = (self.state[6] + np.pi) % (2 * np.pi) - np.pi
            
        # Always wrap phi
        self.state[6] = ((self.state[6] + np.pi) % (2 * np.pi)) - np.pi

        # Velocity limits
        self.state[1] = np.clip(self.state[1], -5, 5)
        self.state[3] = np.clip(self.state[3], -5, 5)
        self.state[5] = np.clip(self.state[5], -10, 10)
        self.state[7] = np.clip(self.state[7], -10, 10)
        
        self.time += self.dt

    def _update_trajectory(self):
        """Update reference trajectory"""
        self.x_reference[0] = self.target_position[0]
        self.x_reference[2] = self.target_position[1]

    def _apply_noise_and_disturbances(self):
        """Apply noise and disturbances"""
        if not self.noise_enabled and not self.disturbance_enabled:
            return
            
        if self.noise_enabled:
            noise_x = np.random.normal(0, self.noise_std)
            noise_y = np.random.normal(0, self.noise_std)
            self.Fx += noise_x
            self.Fy += noise_y
            
            self.Fx = np.clip(self.Fx, -150.0, 150.0)
            self.Fy = np.clip(self.Fy, -150.0, 150.0)
        
        if self.disturbance_enabled and np.random.random() < self.disturbance_probability:
            angle = np.random.uniform(0, 2*np.pi)
            magnitude = np.random.uniform(0.5, self.disturbance_magnitude)
            disturbance_x = magnitude * np.cos(angle)
            disturbance_y = magnitude * np.sin(angle)
            
            self.state[1] += disturbance_x / self.M
            self.state[3] += disturbance_y / self.M

    def get_positions(self):
        """Get cart and pendulum positions for visualization"""
        x_c, _, y_c, _, theta, _, phi, _ = self.state
        
        cart_pos = np.array([x_c, y_c, 0])
        
        pendulum_pos = np.array([
            x_c + self.L * np.sin(theta) * np.cos(phi),
            y_c + self.L * np.sin(theta) * np.sin(phi),
            -self.L * np.cos(theta)
        ])
        
        return cart_pos, pendulum_pos

    def set_target(self, x_target, y_target):
        """Set new target position"""
        max_velocity = 0.5
        max_acceleration = 2.0

        x_target = np.clip(x_target, -2, 2)
        y_target = np.clip(y_target, -2, 2)
        
        prev_x = self.x_reference[0]
        prev_y = self.x_reference[2]
        
        self.target_position = np.array([x_target, y_target])
        
        distance = np.sqrt((x_target - prev_x)**2 + (y_target - prev_y)**2)
        
        if distance > 0.1:
            self.x_reference[0] = prev_x + self.smoothing_factor * (x_target - prev_x)
            self.x_reference[2] = prev_y + self.smoothing_factor * (y_target - prev_y)
        else:
            self.x_reference[0] = x_target
            self.x_reference[2] = y_target
        
        self.x_reference[1] = 0.0
        self.x_reference[3] = 0.0
        self.x_reference[4] = np.pi
        self.x_reference[5] = 0.0
        self.x_reference[6] = self.state[6]
        self.x_reference[7] = 0.0

    def reset(self):
        """Reset to initial state"""
        self.state = np.array([0.0, 0.0, 0.0, 0.0, np.pi - 0.1, 0.0, 0.0, 0.0])
        self.x_reference = np.array([0.0, 0.0, 0.0, 0.0, np.pi, 0.0, 0.0, 0.0])
        self.time = 0.0
        self.Fx = 0.0
        self.Fy = 0.0
        self.mpc_failures = 0
        self.mpc_attempts = 0

    def is_at_target(self, tolerance=None):
        """Check if cart is at target position"""
        if tolerance is None:
            tolerance = self.position_tolerance
        
        current_pos = np.array([self.state[0], self.state[2]])
        distance = np.linalg.norm(self.target_position - current_pos)
        return distance < tolerance

    def enable_noise(self, enabled=True, noise_std=0.1):
        """Enable or disable noise"""
        self.noise_enabled = enabled
        self.noise_std = noise_std

    def enable_disturbances(self, enabled=True, magnitude=2.0, probability=0.01):
        """Enable or disable disturbances"""
        self.disturbance_enabled = enabled
        self.disturbance_magnitude = magnitude
        self.disturbance_probability = probability

    def print_status(self):
        """Print current status"""
        theta_deg = np.degrees(self.state[4])
        cart_pos, pendulum_pos = self.get_positions()
        
        print(f"\nStatus at t={self.time:.2f}s:")
        print(f"  θ={theta_deg:.1f}° (target: 180°)")
        print(f"  Cart: ({cart_pos[0]:.2f}, {cart_pos[1]:.2f})")
        print(f"  Pendulum tip: ({pendulum_pos[0]:.2f}, {pendulum_pos[1]:.2f}, {pendulum_pos[2]:.2f})")
        print(f"  Forces: Fx={self.Fx:.1f}N, Fy={self.Fy:.1f}N")
        
        if self.mpc_attempts > 0:
            success_rate = (1 - self.mpc_failures/self.mpc_attempts) * 100
            print(f"  MPC success: {success_rate:.1f}%")


class SimpleVisualization:
    """Visualization for 3D inverted pendulum"""
    
    def __init__(self):
        self.pendulum = SimplePendulum3D()
        
        # Create figure
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle('3D Inverted Pendulum - MPC Controller', fontsize=14)
        
        # 3D view
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax3d.set_title('3D View')
        
        # Top view
        self.ax_top = self.fig.add_subplot(122)
        self.ax_top.set_title('Top View (X-Y)')
        self.ax_top.set_aspect('equal')
        
        self.setup_plots()
        
        # Animation
        self.animation = animation.FuncAnimation(
            self.fig, self.animate, interval=30, blit=False, cache_frame_data=False
        )
        
        # Key bindings
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
    
    def setup_plots(self):
        """Setup plot elements"""
        # 3D plot elements
        self.cart_3d, = self.ax3d.plot([0], [0], [0], 'bs', markersize=12, label='Cart')
        self.rod_3d, = self.ax3d.plot([0, 0], [0, 0], [0, -1.5], 'r-', linewidth=3)
        self.mass_3d, = self.ax3d.plot([0], [0], [-1.5], 'ro', markersize=10, label='Mass')
        
        self.ax3d.set_xlim([-2, 2])
        self.ax3d.set_ylim([-2, 2])
        self.ax3d.set_zlim([-2, 1])
        self.ax3d.set_xlabel('X [m]')
        self.ax3d.set_ylabel('Y [m]')
        self.ax3d.set_zlabel('Z [m]')
        
        # Top view elements
        self.cart_top, = self.ax_top.plot(0, 0, 'bs', markersize=10, label='Cart')
        self.mass_top, = self.ax_top.plot(0, 0, 'ro', markersize=8, label='Mass')
        self.target_top, = self.ax_top.plot(0, 0, 'g+', markersize=15, 
                                           markeredgewidth=3, label='Target')
        
        self.ax_top.set_xlim([-2.5, 2.5])
        self.ax_top.set_ylim([-2.5, 2.5])
        self.ax_top.set_xlabel('X [m]')
        self.ax_top.set_ylabel('Y [m]')
        self.ax_top.grid(True, alpha=0.3)
        self.ax_top.legend()
        
        # Status text
        self.status_text = self.fig.text(0.02, 0.02, '', fontsize=10, family='monospace')
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        step = 0.3
        current_target = [self.pendulum.target_position[0], self.pendulum.target_position[1]]
        
        if event.key == 'up':
            self.pendulum.set_target(current_target[0], current_target[1] + step)
        elif event.key == 'down':
            self.pendulum.set_target(current_target[0], current_target[1] - step)
        elif event.key == 'left':
            self.pendulum.set_target(current_target[0] - step, current_target[1])
        elif event.key == 'right':
            self.pendulum.set_target(current_target[0] + step, current_target[1])
        elif event.key == 'c':
            self.pendulum.set_target(0.0, 0.0)
        elif event.key == 'r':
            self.pendulum.reset()
        elif event.key == ' ':
            self.pendulum.paused = not self.pendulum.paused
            print(f"{'Paused' if self.pendulum.paused else 'Resumed'}")
        elif event.key == 's':
            self.pendulum.print_status()
        elif event.key == 'n':
            self.pendulum.enable_noise(not self.pendulum.noise_enabled)
        elif event.key == 'd':
            self.pendulum.enable_disturbances(not self.pendulum.disturbance_enabled)
        elif event.key == '1':
            self.pendulum.set_target(1.0, 0.0)
        elif event.key == '2':
            self.pendulum.set_target(-1.0, 1.0)
        elif event.key == '3':
            self.pendulum.set_target(0.0, -1.0)
        elif event.key == '0':
            self.pendulum.set_target(0.0, 0.0)
    
    def animate(self, frame):
        """Animation loop"""
        # Update simulation
        self.pendulum.integrate_step()
        
        # Get positions
        cart_pos, pendulum_pos = self.pendulum.get_positions()
        
        # Update 3D plot
        self.cart_3d.set_data_3d([cart_pos[0]], [cart_pos[1]], [cart_pos[2]])
        self.mass_3d.set_data_3d([pendulum_pos[0]], [pendulum_pos[1]], [pendulum_pos[2]])
        self.rod_3d.set_data_3d([cart_pos[0], pendulum_pos[0]], 
                               [cart_pos[1], pendulum_pos[1]], 
                               [cart_pos[2], pendulum_pos[2]])
        
        # Update top view
        self.cart_top.set_data([cart_pos[0]], [cart_pos[1]])
        self.mass_top.set_data([pendulum_pos[0]], [pendulum_pos[1]])
        self.target_top.set_data([self.pendulum.target_position[0]], 
                                [self.pendulum.target_position[1]])
        
        # Update status
        state = self.pendulum.state
        theta_deg = np.degrees(state[4])
        phi_deg = np.degrees(state[6])
        success_rate = (1 - self.pendulum.mpc_failures/max(self.pendulum.mpc_attempts, 1)) * 100
        
        target_distance = np.sqrt((state[0] - self.pendulum.target_position[0])**2 + 
                                 (state[2] - self.pendulum.target_position[1])**2)
        
        at_target = self.pendulum.is_at_target()
        target_status = "✅ AT TARGET" if at_target else f"Distance: {target_distance:.3f}m"
        
        status = f'''Time: {self.pendulum.time:.1f}s
Cart: ({state[0]:.3f}, {state[2]:.3f}) → Target: ({self.pendulum.target_position[0]:.3f}, {self.pendulum.target_position[1]:.3f})
{target_status}
θ: {theta_deg:.1f}° (target: 180°)  φ: {phi_deg:.1f}°
Forces: Fx={self.pendulum.Fx:.1f}N  Fy={self.pendulum.Fy:.1f}N
MPC Success: {success_rate:.0f}%
Noise: {'ON' if self.pendulum.noise_enabled else 'OFF'}  Dist: {'ON' if self.pendulum.disturbance_enabled else 'OFF'}'''
        
        self.status_text.set_text(status)
        
        return (self.cart_3d, self.mass_3d, self.rod_3d, 
                self.cart_top, self.mass_top, self.target_top)


def main():
    try:
        sim = SimpleVisualization()
        plt.show()
        sim.pendulum.print_status()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()