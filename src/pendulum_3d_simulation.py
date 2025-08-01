"""
3D Inverted Pendulum on Cart 
==========================================

Controls:
- Arrow Keys: Move cart target position
- R: Reset simulation  
- Space: Pause/Resume
- P: Generate plots
- C: Center cart
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time

class Pendulum3D:
    """3D Inverted Pendulum on Cart System"""
    
    def __init__(self):
        # System parameters
        self.M = 2.0        # Cart mass [kg]
        self.m = 1.0        # Pendulum mass [kg]
        self.L = 1.5        # Pendulum length [m]
        self.g = 9.81       # Gravity [m/s^2]
        self.b_theta = 3.0  # Damping for theta [N*m*s/rad] 
        self.b_phi = 3.0    # Damping for phi [N*m*s/rad]
        # Control forces
        self.Fx = 0.0       # Force in x-direction [N]
        self.Fy = 0.0       # Force in y-direction [N]
        
        # Target positions for cart
        self.target_x = 0.0
        self.target_y = 0.0
        
        # Simple position controller gains
        self.control_gain = 80.0      
        self.control_damping = 15.0   
        
        # State variables: [x_c, x_c_dot, y_c, y_c_dot, theta, theta_dot, phi, phi_dot]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, np.pi - 0.05, 0.0, 0.0, 0.0])  
        
        # Simulation parameters
        self.dt = 0.018     # Time step [s]
        self.time = 0.0     # Current time [s]
        
        self.real_start_time = time.time()
        # Animation control
        self.paused = False
        
        # Data logging
        self.time_history = []
        self.cart_x_history = []
        self.cart_y_history = []
        self.theta_history = []
        self.phi_history = []
        self.fx_history = []
        self.fy_history = []
        self.energy_history = []
        
    def equations_of_motion(self, state, Fx, Fy):
        """
        3D Inverted Pendulum Equations of Motion
        """
        x_c, x_c_dot, y_c, y_c_dot, theta, theta_dot, phi, phi_dot = state
        
        # Handle singularities properly
        epsilon = 1e-6
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        # Protect against singularities
        if abs(sin_theta) < epsilon:
            sin_theta = np.sign(sin_theta) * epsilon if sin_theta != 0 else epsilon
        
        # Mass matrix components
        M_total = self.M + self.m
        mL = self.m * self.L
        mL2 = self.m * self.L**2
        
        # Build the 4x4 mass matrix for [x_c_ddot, y_c_ddot, theta_ddot, phi_ddot]
        M = np.zeros((4, 4))
        
        # Cart dynamics
        M[0, 0] = M_total
        M[0, 2] = mL * cos_theta * cos_phi
        M[0, 3] = -mL * sin_theta * sin_phi
        
        M[1, 1] = M_total
        M[1, 2] = mL * cos_theta * sin_phi
        M[1, 3] = mL * sin_theta * cos_phi
        
        # Pendulum dynamics
        M[2, 0] = mL * cos_theta * cos_phi
        M[2, 1] = mL * cos_theta * sin_phi
        M[2, 2] = mL2
        M[2, 3] = 0
        
        M[3, 0] = -mL * sin_theta * sin_phi
        M[3, 1] = mL * sin_theta * cos_phi
        M[3, 2] = 0
        M[3, 3] = mL2 * sin_theta**2
        
        # Right-hand side vector
        RHS = np.zeros(4)
        
        # Cart X forces
        RHS[0] = (Fx - mL * (
            theta_dot**2 * sin_theta * cos_phi +
            phi_dot**2 * sin_theta * cos_phi +
            2 * theta_dot * phi_dot * cos_theta * sin_phi
        ))
        
        # Cart Y forces  
        RHS[1] = (Fy - mL * (
            theta_dot**2 * sin_theta * sin_phi +
            phi_dot**2 * sin_theta * sin_phi -
            2 * theta_dot * phi_dot * cos_theta * cos_phi
        ))
        
        # Pendulum theta torques
        RHS[2] = (-self.m * self.g * self.L * sin_theta - 
          self.b_theta * theta_dot +
          mL2 * phi_dot**2 * sin_theta * cos_theta +
          0.01 * (theta - np.pi))  
        
        # Pendulum phi torques
        RHS[3] = (-self.b_phi * phi_dot -
                  2 * mL2 * sin_theta * cos_theta * theta_dot * phi_dot)
        
        # Solve the linear system M * accelerations = RHS
        try:
            # Better regularization for stability
            M_reg = M + np.eye(4) * 1e-6
            accelerations = np.linalg.solve(M_reg, RHS)
            x_c_ddot, y_c_ddot, theta_ddot, phi_ddot = accelerations
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            x_c_ddot = Fx / M_total
            y_c_ddot = Fy / M_total
            theta_ddot = -self.g * sin_theta / self.L - self.b_theta * theta_dot / mL2
            phi_ddot = -self.b_phi * phi_dot / (mL2 * sin_theta**2 + epsilon)
        
        # Limit accelerations for numerical stability
        max_accel = 20.0  
        x_c_ddot = np.clip(x_c_ddot, -max_accel, max_accel)
        y_c_ddot = np.clip(y_c_ddot, -max_accel, max_accel)
        theta_ddot = np.clip(theta_ddot, -max_accel, max_accel)
        phi_ddot = np.clip(phi_ddot, -max_accel, max_accel)
        
        # Return state derivatives
        return np.array([x_c_dot, x_c_ddot, y_c_dot, y_c_ddot, 
                        theta_dot, theta_ddot, phi_dot, phi_ddot])
    
    def integrate_step(self):
        """Integrate one time step using RK4"""
        if self.paused:
            return
        # Check if real time has passed
        real_elapsed = time.time() - self.real_start_time
        if self.time >= real_elapsed:
            return
            
        # Position controller
        error_x = self.target_x - self.state[0]
        error_y = self.target_y - self.state[2]
        
        self.Fx = (self.control_gain * error_x - 
                  self.control_damping * self.state[1])
        self.Fy = (self.control_gain * error_y - 
                  self.control_damping * self.state[3])
        
        # Limit control forces
        max_force = 100.0  # Maximum control force [N]
        self.Fx = np.clip(self.Fx, -max_force, max_force)
        self.Fy = np.clip(self.Fy, -max_force, max_force)
        
        # RK4 integration
        k1 = self.equations_of_motion(self.state, self.Fx, self.Fy)
        k2 = self.equations_of_motion(self.state + 0.5*self.dt*k1, self.Fx, self.Fy)
        k3 = self.equations_of_motion(self.state + 0.5*self.dt*k2, self.Fx, self.Fy)
        k4 = self.equations_of_motion(self.state + self.dt*k3, self.Fx, self.Fy)
        
        # Update state
        self.state += self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # FIXED: Wrap theta to [-π, π] range for proper display
        self.state[4] = np.arctan2(np.sin(self.state[4]), np.cos(self.state[4]))
        
        # Wrap phi angle to [-pi, pi]
        self.state[6] = np.arctan2(np.sin(self.state[6]), np.cos(self.state[6]))
        
        # Limit cart position to prevent it from going off-screen
        self.state[0] = np.clip(self.state[0], -4.0, 4.0)
        self.state[2] = np.clip(self.state[2], -4.0, 4.0)
        
        # Limit velocities to prevent numerical explosion
        max_vel = 5.0
        self.state[1] = np.clip(self.state[1], -max_vel, max_vel)  # x_dot
        self.state[3] = np.clip(self.state[3], -max_vel, max_vel)  # y_dot
        self.state[5] = np.clip(self.state[5], -max_vel, max_vel)  # theta_dot
        self.state[7] = np.clip(self.state[7], -max_vel, max_vel)  # phi_dot
        
        # Safety check for numerical stability
        if not np.isfinite(self.state).all():
            print("Numerical instability detected, resetting...")
            self.reset()
            return
        
        self.time += self.dt
        
        
        # Log data for plotting
        self.time_history.append(self.time)
        self.cart_x_history.append(self.state[0])
        self.cart_y_history.append(self.state[2])
        self.theta_history.append(np.degrees(self.state[4]))
        self.phi_history.append(np.degrees(self.state[6]))
        self.fx_history.append(self.Fx)
        self.fy_history.append(self.Fy)
        
        # Calculate total energy
        kinetic_energy = 0.5 * (self.M + self.m) * (self.state[1]**2 + self.state[3]**2)
        kinetic_energy += 0.5 * self.m * self.L**2 * (self.state[5]**2 + self.state[7]**2 * np.sin(self.state[4])**2)
        potential_energy = -self.m * self.g * self.L * np.cos(self.state[4])
        total_energy = kinetic_energy + potential_energy
        self.energy_history.append(total_energy)
        
        # Limit history length for memory
        max_history = 5000
        if len(self.time_history) > max_history:
            self.time_history.pop(0)
            self.cart_x_history.pop(0)
            self.cart_y_history.pop(0)
            self.theta_history.pop(0)
            self.phi_history.pop(0)
            self.fx_history.pop(0)
            self.fy_history.pop(0)
            self.energy_history.pop(0)
    
    def get_positions(self):
        """Get current positions for visualization"""
        x_c, _, y_c, _, theta, _, phi, _ = self.state
        
        # Cart position
        cart_pos = np.array([x_c, y_c, 0])
        
        # Pendulum mass position
        pendulum_pos = np.array([
            x_c + self.L * np.sin(theta) * np.cos(phi),
            y_c + self.L * np.sin(theta) * np.sin(phi),
            -self.L * np.cos(theta)
        ])
        
        return cart_pos, pendulum_pos
    
    def reset(self):
        """Reset simulation to initial state"""
        self.state = np.array([0.0, 0.0, 0.0, 0.0, np.pi - 0.05, 0.0, 0.0, 0.0])  # Start closer to upright
        self.time = 0.0
        self.real_start_time = time.time()
        self.target_x = 0.0
        self.target_y = 0.0
        self.Fx = 0.0
        self.Fy = 0.0
        
        # Clear history
        self.time_history.clear()
        self.cart_x_history.clear()
        self.cart_y_history.clear()
        self.theta_history.clear()
        self.phi_history.clear()
        self.fx_history.clear()
        self.fy_history.clear()
        self.energy_history.clear()
    
    def generate_report_plots(self):
        """Generate plots for analysis"""
        if len(self.time_history) < 10:
            print("Not enough data for plots. Run simulation longer.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('3D Inverted Pendulum Simulation Results', fontsize=16)
        
        # Plot 1: Cart Position
        axes[0, 0].plot(self.time_history, self.cart_x_history, 'b-', label='X position')
        axes[0, 0].plot(self.time_history, self.cart_y_history, 'r-', label='Y position')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].set_ylabel('Cart Position [m]')
        axes[0, 0].set_title('Cart Position vs Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Pendulum Angles
        axes[0, 1].plot(self.time_history, self.theta_history, 'g-', label='θ (polar)')
        axes[0, 1].plot(self.time_history, self.phi_history, 'm-', label='φ (azimuthal)')
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].set_ylabel('Angle [degrees]')
        axes[0, 1].set_title('Pendulum Angles vs Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Control Forces
        axes[0, 2].plot(self.time_history, self.fx_history, 'c-', label='Fx')
        axes[0, 2].plot(self.time_history, self.fy_history, 'orange', label='Fy')
        axes[0, 2].set_xlabel('Time [s]')
        axes[0, 2].set_ylabel('Force [N]')
        axes[0, 2].set_title('Control Forces vs Time')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Phase Portrait (θ vs θ̇)
        if len(self.theta_history) > 1:
            theta_rad = np.array(self.theta_history) * np.pi / 180
            theta_dot = np.gradient(theta_rad, self.dt)
            axes[1, 0].plot(self.theta_history, theta_dot * 180/np.pi, 'g-')
            axes[1, 0].set_xlabel('θ [degrees]')
            axes[1, 0].set_ylabel('θ̇ [degrees/s]')
            axes[1, 0].set_title('Phase Portrait (θ)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Energy Conservation
        axes[1, 1].plot(self.time_history, self.energy_history, 'k-')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Total Energy [J]')
        axes[1, 1].set_title('Energy Conservation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: 3D Trajectory
        if len(self.time_history) > 1:
            x_positions = []
            y_positions = []
            z_positions = []
            
            for i in range(len(self.time_history)):
                theta_rad = self.theta_history[i] * np.pi / 180
                phi_rad = self.phi_history[i] * np.pi / 180
                x_p = self.cart_x_history[i] + self.L * np.sin(theta_rad) * np.cos(phi_rad)
                y_p = self.cart_y_history[i] + self.L * np.sin(theta_rad) * np.sin(phi_rad)
                z_p = -self.L * np.cos(theta_rad)
                x_positions.append(x_p)
                y_positions.append(y_p)
                z_positions.append(z_p)
            
            axes[1, 2].remove()
            ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
            ax_3d.plot(x_positions, y_positions, z_positions, 'r-', alpha=0.7)
            ax_3d.set_xlabel('X [m]')
            ax_3d.set_ylabel('Y [m]')
            ax_3d.set_zlabel('Z [m]')
            ax_3d.set_title('Pendulum Mass Trajectory')
        
        plt.tight_layout()
        plt.show()
        print("Report plots generated!")

class PendulumSimulation:
    """3D Pendulum Visualization and Simulation"""
    
    def __init__(self):
        self.pendulum = Pendulum3D()
        
        # Create figure and subplots
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle('3D Inverted Pendulum on Cart', fontsize=16)
        
        # 3D plot
        self.ax = self.fig.add_subplot(221, projection='3d')
        self.ax.set_title('3D View')
        
        # Top view (X-Y plane)
        self.ax_top = self.fig.add_subplot(222)
        self.ax_top.set_title('Top View (X-Y Plane)')
        self.ax_top.set_aspect('equal')
        
        # Side view (X-Z plane)
        self.ax_side1 = self.fig.add_subplot(223)
        self.ax_side1.set_title('Side View (X-Z Plane)')
        self.ax_side1.set_aspect('equal')
        
        # Side view (Y-Z plane)
        self.ax_side2 = self.fig.add_subplot(224)
        self.ax_side2.set_title('Side View (Y-Z Plane)')
        self.ax_side2.set_aspect('equal')
        
        # Initialize plot elements
        self.setup_3d_plot()
        self.setup_2d_plots()
        self.setup_controls()
        
        # Print instructions
        print("=== 3D PENDULUM SIMULATION ===")
        print("Controls:")
        print("  Arrow Keys: Move cart target position")
        print("  R: Reset simulation")
        print("  Space: Pause/Resume")
        print("  P: Generate plots")
        print("  C: Center cart")

        # Animation with normal speed
        self.animation = animation.FuncAnimation(
            self.fig, self.animate, interval=20, blit=False, cache_frame_data=False
        )
        
        # Key bindings
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
        
    def setup_3d_plot(self):
        """Setup 3D visualization elements"""
        # Cart
        self.cart_3d, = self.ax.plot([0], [0], [0], 'bs', markersize=15, label='Cart')
        
        # Pendulum rod
        self.rod_3d, = self.ax.plot([0, 0], [0, 0], [0, -1.5], 'r-', linewidth=3)
        
        # Pendulum mass
        self.mass_3d, = self.ax.plot([0], [0], [-1.5], 'ro', markersize=12, label='Mass')
        
        # Trajectory
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []
        self.traj_3d, = self.ax.plot([], [], [], 'g--', alpha=0.6, label='Trajectory')
        
        # Set limits and labels
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_zlim([-3, 1])
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.legend()
        
        # Ground plane
        xx, yy = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
        zz = np.zeros_like(xx)
        self.ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')
        
    def setup_2d_plots(self):
        """Setup 2D view plots"""
        # Top view elements
        self.cart_top, = self.ax_top.plot(0, 0, 'bs', markersize=10)
        self.mass_top, = self.ax_top.plot(0, 0, 'ro', markersize=8)
        self.rod_top, = self.ax_top.plot([0, 0], [0, 0], 'r-', linewidth=2)
        self.target_top, = self.ax_top.plot(0, 0, 'g+', markersize=15, markeredgewidth=3, label='Target')
        
        # Side view 1 elements (X-Z)
        self.cart_side1, = self.ax_side1.plot(0, 0, 'bs', markersize=10)
        self.mass_side1, = self.ax_side1.plot(0, -1.5, 'ro', markersize=8)
        self.rod_side1, = self.ax_side1.plot([0, 0], [0, -1.5], 'r-', linewidth=2)
        
        # Side view 2 elements (Y-Z)
        self.cart_side2, = self.ax_side2.plot(0, 0, 'bs', markersize=10)
        self.mass_side2, = self.ax_side2.plot(0, -1.5, 'ro', markersize=8)
        self.rod_side2, = self.ax_side2.plot([0, 0], [0, -1.5], 'r-', linewidth=2)
        
        # Set limits for 2D plots
        for ax in [self.ax_top, self.ax_side1, self.ax_side2]:
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])
            ax.grid(True, alpha=0.3)
        
        self.ax_top.set_xlabel('X [m]')
        self.ax_top.set_ylabel('Y [m]')
        self.ax_side1.set_xlabel('X [m]')
        self.ax_side1.set_ylabel('Z [m]')
        self.ax_side2.set_xlabel('Y [m]')
        self.ax_side2.set_ylabel('Z [m]')
        
    def setup_controls(self):
        """Setup interactive controls"""
        self.fig.text(0.02, 0.95, 'Controls:', fontsize=12, weight='bold')
        self.fig.text(0.02, 0.92, '• Arrow Keys: Move cart target', fontsize=9)
        self.fig.text(0.02, 0.89, '• R: Reset  Space: Pause  P: Plots  C: Center', fontsize=9)

        # Status text
        self.status_text = self.fig.text(0.02, 0.60, '', fontsize=9, family='monospace')
        
    def on_key_press(self, event):
        """Handle keyboard input"""
        step = 0.3
        
        if event.key == 'up':
            self.pendulum.target_y += step
            print(f"Target Y: {self.pendulum.target_y:.2f}m")
        elif event.key == 'down':
            self.pendulum.target_y -= step
            print(f"Target Y: {self.pendulum.target_y:.2f}m")
        elif event.key == 'left':
            self.pendulum.target_x -= step
            print(f"Target X: {self.pendulum.target_x:.2f}m")
        elif event.key == 'right':
            self.pendulum.target_x += step
            print(f"Target X: {self.pendulum.target_x:.2f}m")
        elif event.key == 'r':
            self.pendulum.reset()
            self.trajectory_x.clear()
            self.trajectory_y.clear()
            self.trajectory_z.clear()
            print("Simulation reset")
        elif event.key == ' ':
            self.pendulum.paused = not self.pendulum.paused
            print(f"Simulation {'paused' if self.pendulum.paused else 'resumed'}")
        elif event.key == 'c':
            self.pendulum.target_x = 0.0
            self.pendulum.target_y = 0.0
            print("Cart centered")
        elif event.key == 'p':
            self.pendulum.generate_report_plots()
            
        # Limit target positions
        self.pendulum.target_x = np.clip(self.pendulum.target_x, -2.5, 2.5)
        self.pendulum.target_y = np.clip(self.pendulum.target_y, -2.5, 2.5)
    
    def animate(self, frame):
        """Animation function"""
        # Integrate physics
        self.pendulum.integrate_step()
        
        # Get current positions
        cart_pos, pendulum_pos = self.pendulum.get_positions()
        
        # Update trajectory
        self.trajectory_x.append(pendulum_pos[0])
        self.trajectory_y.append(pendulum_pos[1])
        self.trajectory_z.append(pendulum_pos[2])
        
        # Limit trajectory length
        max_traj_length = 500
        if len(self.trajectory_x) > max_traj_length:
            self.trajectory_x.pop(0)
            self.trajectory_y.pop(0)
            self.trajectory_z.pop(0)
        
        # Update 3D plot
        self.cart_3d.set_data_3d([cart_pos[0]], [cart_pos[1]], [cart_pos[2]])
        self.mass_3d.set_data_3d([pendulum_pos[0]], [pendulum_pos[1]], [pendulum_pos[2]])
        self.rod_3d.set_data_3d([cart_pos[0], pendulum_pos[0]], 
                               [cart_pos[1], pendulum_pos[1]], 
                               [cart_pos[2], pendulum_pos[2]])
        self.traj_3d.set_data_3d(self.trajectory_x, self.trajectory_y, self.trajectory_z)
        
        # Update 2D plots
        # Top view
        self.cart_top.set_data([cart_pos[0]], [cart_pos[1]])
        self.mass_top.set_data([pendulum_pos[0]], [pendulum_pos[1]])
        self.rod_top.set_data([cart_pos[0], pendulum_pos[0]], [cart_pos[1], pendulum_pos[1]])
        self.target_top.set_data([self.pendulum.target_x], [self.pendulum.target_y])
        
        # Side view 1 (X-Z)
        self.cart_side1.set_data([cart_pos[0]], [cart_pos[2]])
        self.mass_side1.set_data([pendulum_pos[0]], [pendulum_pos[2]])
        self.rod_side1.set_data([cart_pos[0], pendulum_pos[0]], [cart_pos[2], pendulum_pos[2]])
        
        # Side view 2 (Y-Z)
        self.cart_side2.set_data([cart_pos[1]], [cart_pos[2]])
        self.mass_side2.set_data([pendulum_pos[1]], [pendulum_pos[2]])
        self.rod_side2.set_data([cart_pos[1], pendulum_pos[1]], [cart_pos[2], pendulum_pos[2]])
        
        # Update status text
        state = self.pendulum.state
        theta_deg = np.degrees(state[4])
        phi_deg = np.degrees(state[6])
        
        real_time = time.time() - self.pendulum.real_start_time
        status = f'''Sim: {self.pendulum.time:.1f}s | Real: {real_time:.1f}s
        Cart: ({state[0]:.2f}, {state[2]:.2f})
        Target: ({self.pendulum.target_x:.2f}, {self.pendulum.target_y:.2f})
        θ: {theta_deg:.1f}°  φ: {phi_deg:.1f}°
        Forces: Fx={self.pendulum.Fx:.1f}N  Fy={self.pendulum.Fy:.1f}N
        Damping: θ={self.pendulum.b_theta:.2f}, φ={self.pendulum.b_phi:.2f}'''
        
        self.status_text.set_text(status)
        
        return (self.cart_3d, self.mass_3d, self.rod_3d, self.traj_3d,
                self.cart_top, self.mass_top, self.rod_top, self.target_top,
                self.cart_side1, self.mass_side1, self.rod_side1,
                self.cart_side2, self.mass_side2, self.rod_side2)

def main():
    print("3D Inverted Pendulum Simulation")
    
    # Create and run simulation
    sim = PendulumSimulation()
    plt.show()

if __name__ == "__main__":
    main()