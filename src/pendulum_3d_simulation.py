import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time
from ILQR_3d_Controller import iLQR3DPendulum


class Pendulum3DSimulation:
    """3D Pendulum simulation with iLQR controller"""
    
    def __init__(self):
        # Physical parameters
        self.M = 2.0
        self.m = 1.0
        self.L = 1.5
        self.g = 9.81
        self.b_theta = 0.5
        self.b_phi = 0.5
        
        # State: [x_c, x_c_dot, y_c, y_c_dot, theta, theta_dot, phi, phi_dot]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0])  # Start at 60 degrees
        
        # Simulation parameters
        self.dt = 0.02
        self.time = 0.0
        self.paused = False
        
        # Control
        self.Fx = 0.0
        self.Fy = 0.0
        
        # iLQR controller with enhanced viscous damping
        self.ilqr = iLQR3DPendulum(T=50, dt=self.dt, M=self.M, m=self.m, L=self.L, g=self.g,
                                  b_theta=self.b_theta, b_phi=self.b_phi,
                                  b_cart=0.01)
        
        # Print controller summary
        self.ilqr.print_summary()
        
        # Control update frequency 
        self.control_update_freq = 3
        self.control_counter = 0
        self.control_valid = False
        self.control_index = 0
        
        # Performance tracking
        self.solve_times = []
        self.solve_successes = 0
        self.total_solves = 0
    
    def dynamics(self, state, u):
        """System dynamics for simulation"""
        f = self.ilqr.dynamics(state, u)
        return f
    
    def update_control(self):
        """Update control using iLQR"""
        if self.control_counter % self.control_update_freq == 0:
            start_time = time.time()
            
            # Solve iLQR
            success = self.ilqr.solve(self.state)
            
            solve_time = time.time() - start_time
            self.solve_times.append(solve_time)
            self.total_solves += 1
            
            if success:
                self.solve_successes += 1
                self.control_valid = True
                self.control_index = 0
            else:
                print("iLQR solve failed, using fallback control")
                self.control_valid = False
        
        # Get control input
        if self.control_valid and self.control_index < len(self.ilqr.u_traj):
            u = self.ilqr.u_traj[self.control_index]
            self.Fx, self.Fy = u
            self.control_index += 1
        else:
            # Improved fallback control (energy-based swing-up) - restored original working form
            theta_error = self._wrap_angle(self.state[4] - np.pi)
            phi_error = self._wrap_angle(self.state[6])
            
            # Energy-based control for swing-up
            theta = self.state[4]
            theta_dot = self.state[5]
            phi_dot = self.state[7]
            
            # Desired energy for upright position
            E_desired = self.m * self.g * self.L  # Potential energy at upright
            
            # Current energy
            E_kinetic = 0.5 * self.m * self.L**2 * (theta_dot**2 + np.sin(theta)**2 * phi_dot**2)
            E_potential = self.m * self.g * self.L * (1 - np.cos(theta))
            E_current = E_kinetic + E_potential
            
            # Energy error
            E_error = E_desired - E_current
            
            # Control gains
            k_energy = 50.0
            k_theta = 200.0
            k_phi = 200.0
            k_damp = 20.0
            
            # Energy-based control
            if abs(theta_error) > np.pi/6:  # Far from upright, use energy control
                self.Fx = k_energy * E_error * np.sign(theta_dot) * np.cos(theta) * np.cos(self.state[6])
                self.Fy = k_energy * E_error * np.sign(phi_dot) * np.sin(theta) * np.sin(self.state[6])
            else:  # Close to upright, use PD control
                self.Fx = -k_theta * theta_error - k_damp * theta_dot
                self.Fy = -k_phi * phi_error - k_damp * phi_dot
        
        # Clip controls
        self.Fx = np.clip(self.Fx, -50, 50)
        self.Fy = np.clip(self.Fy, -50, 50)
        
        self.control_counter += 1
    
    def step(self):
        """Single simulation step"""
        if self.paused:
            return
        
        # Update control
        self.update_control()
        
        # RK4 integration
        k1 = self.dynamics(self.state, [self.Fx, self.Fy])
        k2 = self.dynamics(self.state + 0.5*self.dt*k1, [self.Fx, self.Fy])
        k3 = self.dynamics(self.state + 0.5*self.dt*k2, [self.Fx, self.Fy])
        k4 = self.dynamics(self.state + self.dt*k3, [self.Fx, self.Fy])
        
        self.state += self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Angle wrapping
        self.state[4] = ((self.state[4] + np.pi) % (2*np.pi)) - np.pi
        self.state[6] = ((self.state[6] + np.pi) % (2*np.pi)) - np.pi
        
        self.time += self.dt
    
    def get_positions(self):
        """Get cart and pendulum positions"""
        x_c, _, y_c, _, theta, _, phi, _ = self.state
        
        cart_pos = np.array([x_c, y_c, 0])
        pendulum_pos = np.array([
            x_c + self.L * np.sin(theta) * np.cos(phi),
            y_c + self.L * np.sin(theta) * np.sin(phi),
            -self.L * np.cos(theta)
        ])
        
        return cart_pos, pendulum_pos
    
    def reset(self):
        """Reset simulation"""
        self.state = np.array([0.0, 0.0, 0.0, 0.0, np.pi/3, 0.0, 0.2, 0.0])
        self.time = 0.0
        self.Fx = 0.0
        self.Fy = 0.0
        self.control_counter = 0
        self.control_valid = False
        self.control_index = 0
        self.ilqr.reset_trajectory()
    
    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def print_stats(self):
        """Print performance statistics"""
        if self.total_solves > 0:
            success_rate = self.solve_successes / self.total_solves * 100
            avg_solve_time = np.mean(self.solve_times) * 1000  # ms
            print(f"\niLQR Performance:")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Average solve time: {avg_solve_time:.1f} ms")
            print(f"Total solves: {self.total_solves}")


class Visualization:
    """3D Visualization"""
    
    def __init__(self):
        self.pendulum = Pendulum3DSimulation()
        
        # Create plots
        self.fig, (self.ax3d, self.ax_top) = plt.subplots(1, 2, figsize=(14, 7))
        self.ax3d.remove()
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        
        self.fig.suptitle('3D Inverted Pendulum - iLQR Controller with Viscous Damping', fontsize=14)
        
        self.setup_plots()
        
        # Animation
        self.animation = animation.FuncAnimation(
            self.fig, self.animate, interval=30, blit=False, cache_frame_data=False)
        
        # Key bindings
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
    
    def setup_plots(self):
        """Setup plot elements"""
        # 3D plot
        self.cart_3d, = self.ax3d.plot([0], [0], [0], 'bs', markersize=12, label='Cart')
        self.rod_3d, = self.ax3d.plot([0, 0], [0, 0], [0, -1.5], 'r-', linewidth=4)
        self.mass_3d, = self.ax3d.plot([0], [0], [-1.5], 'ro', markersize=10, label='Mass')
        
        # Trajectory
        self.traj_x = []
        self.traj_y = []
        self.traj_z = []
        self.traj_3d, = self.ax3d.plot([], [], [], 'g--', alpha=0.6, label='Trajectory')
        
        self.ax3d.set_xlim([-2.5, 2.5])
        self.ax3d.set_ylim([-2.5, 2.5])
        self.ax3d.set_zlim([-2, 1])
        self.ax3d.set_xlabel('$x_c$ [m]')
        self.ax3d.set_ylabel('$y_c$ [m]')
        self.ax3d.set_zlabel('$z$ [m]')
        self.ax3d.legend()
        
        # Ground plane
        xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
        zz = np.zeros_like(xx)
        self.ax3d.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        # Top view
        self.cart_top, = self.ax_top.plot(0, 0, 'bs', markersize=10, label='Cart')
        self.mass_top, = self.ax_top.plot(0, 0, 'ro', markersize=8, label='Mass')
        self.rod_top, = self.ax_top.plot([0, 0], [0, 0], 'r-', linewidth=2)
        
        self.ax_top.set_xlim([-2.5, 2.5])
        self.ax_top.set_ylim([-2.5, 2.5])
        self.ax_top.set_xlabel('$x_c$ [m]')
        self.ax_top.set_ylabel('$y_c$ [m]')
        self.ax_top.grid(True, alpha=0.3)
        self.ax_top.set_aspect('equal')
        self.ax_top.legend()
        
        # Status text
        self.status_text = self.fig.text(0.02, 0.02, '', fontsize=9, family='monospace')
    
    def on_key_press(self, event):
        """Handle keys"""
        if event.key == 'r':
            self.pendulum.reset()
            self.traj_x.clear()
            self.traj_y.clear()
            self.traj_z.clear()
            print("Reset simulation")
        elif event.key == ' ':
            self.pendulum.paused = not self.pendulum.paused
            print(f"{'Paused' if self.pendulum.paused else 'Resumed'}")
        elif event.key == 's':
            self.pendulum.print_stats()
    
    def animate(self, frame):
        """Animation loop"""
        self.pendulum.step()
        
        # Get positions
        cart_pos, pend_pos = self.pendulum.get_positions()
        
        # Update trajectory
        self.traj_x.append(pend_pos[0])
        self.traj_y.append(pend_pos[1])
        self.traj_z.append(pend_pos[2])
        
        # Limit trajectory length
        max_traj = 500
        if len(self.traj_x) > max_traj:
            self.traj_x.pop(0)
            self.traj_y.pop(0)
            self.traj_z.pop(0)
        
        # Update 3D plot
        self.cart_3d.set_data_3d([cart_pos[0]], [cart_pos[1]], [cart_pos[2]])
        self.mass_3d.set_data_3d([pend_pos[0]], [pend_pos[1]], [pend_pos[2]])
        self.rod_3d.set_data_3d([cart_pos[0], pend_pos[0]], 
                               [cart_pos[1], pend_pos[1]],
                               [cart_pos[2], pend_pos[2]])
        self.traj_3d.set_data_3d(self.traj_x, self.traj_y, self.traj_z)
        
        # Update top view
        self.cart_top.set_data([cart_pos[0]], [cart_pos[1]])
        self.mass_top.set_data([pend_pos[0]], [pend_pos[1]])
        self.rod_top.set_data([cart_pos[0], pend_pos[0]], [cart_pos[1], pend_pos[1]])
        
        # Status
        state = self.pendulum.state
        theta_deg = np.degrees(state[4])
        phi_deg = np.degrees(state[6])
        
        success_rate = 0
        avg_solve_time = 0
        if self.pendulum.total_solves > 0:
            success_rate = self.pendulum.solve_successes / self.pendulum.total_solves * 100
            avg_solve_time = np.mean(self.pendulum.solve_times) * 1000
        
        status = f'''Time: {self.pendulum.time:.1f}s
θ: {theta_deg:.1f}° (target: 180°)
φ: {phi_deg:.1f}° (target: 0°)
Cart: ({state[0]:.2f}, {state[2]:.2f})
Forces: Fx={self.pendulum.Fx:.1f}N, Fy={self.pendulum.Fy:.1f}N
iLQR Success: {success_rate:.0f}% ({self.pendulum.total_solves} solves)
Avg solve time: {avg_solve_time:.1f}ms
Controls: r=reset, space=pause, s=stats'''
        
        self.status_text.set_text(status)
        
        return (self.cart_3d, self.mass_3d, self.rod_3d, self.traj_3d,
                self.cart_top, self.mass_top, self.rod_top)


def main():
    """Run the simulation"""
    print("Controls:")
    print("  r - Reset simulation")
    print("  space - Pause/resume")
    print("  s - Print statistics")
    viz = Visualization()
    plt.show()


if __name__ == "__main__":
    main()