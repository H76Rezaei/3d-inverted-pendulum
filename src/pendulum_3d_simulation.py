import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

# Parameters
m = 1.0    # mass (kg)
L = 1.0    # length (m) 
g = 9.81   # gravity (m/s^2)

def pendulum_3d_dynamics(state, t, u):
    """
    3D spherical pendulum dynamics
    state: [theta, theta_dot, phi, phi_dot]
    u: [tau_theta, tau_phi] - torque inputs
    """
    theta, theta_dot, phi, phi_dot = state
    tau_theta, tau_phi = u
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Theta equation (polar angle dynamics)
    theta_ddot = sin_theta * cos_theta * phi_dot**2 - (g/L) * sin_theta + tau_theta/(m*L**2)
    
    # Phi equation (azimuthal angle dynamics)
    if abs(sin_theta) < 1e-3:  # Near vertical - singularity handling
        phi_ddot = tau_phi / (m * L**2)
    else:
        phi_ddot = ((tau_phi - 2*m*L**2*sin_theta*cos_theta*theta_dot*phi_dot) / 
                    (m*L**2*sin_theta**2))
    
    return [theta_dot, theta_ddot, phi_dot, phi_ddot]

def angles_to_cartesian(theta, phi, L):
    """Convert spherical coordinates to Cartesian coordinates"""
    x = L * np.sin(theta) * np.cos(phi)
    y = L * np.sin(theta) * np.sin(phi)
    z = L * np.cos(theta)
    return x, y, z

def simulate_pendulum():
    """Simulate pendulum with specified initial conditions"""
    
    # TEST CASES - Change these values to test different behaviors:
    # Case 1: Small oscillations     [0.1, 0.0, 0.0, 0.0]
    # Case 2: 2D pendulum swing      [1.0, 0.0, 0.0, 0.0]  
    # Case 3: Conical pendulum       [0.5, 0.0, 0.0, 2.0]
    # Case 4: Complex 3D motion      [0.8, 0.5, 0.3, 1.5]
    # Case 5: Near-vertical spinning [0.1, 0.0, 0.0, 5.0]
    
    initial_state = [0.8, 0.5, 0.3, 1.5]  # [theta, theta_dot, phi, phi_dot]
    
    # Time array
    t = np.linspace(0, 8, 800)
    
    # Control input (no control - free motion)
    u = [0, 0]
    
    # Solve ODE
    solution = odeint(pendulum_3d_dynamics, initial_state, t, args=(u,))
    
    return t, solution, initial_state

# Run simulation
t, states, initial_state = simulate_pendulum()

# Convert to Cartesian coordinates for visualization
x_pos, y_pos, z_pos = [], [], []
for i in range(len(states)):
    theta = states[i, 0]
    phi = states[i, 2]
    x, y, z = angles_to_cartesian(theta, phi, L)
    x_pos.append(x)
    y_pos.append(y) 
    z_pos.append(z)

x_pos = np.array(x_pos)
y_pos = np.array(y_pos)
z_pos = np.array(z_pos)

# Create figure with layout
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

# 3D animation plot
ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_3d.set_xlim([-1.5, 1.5])
ax_3d.set_ylim([-1.5, 1.5])
ax_3d.set_zlim([-1.5, 1.5])
ax_3d.set_xlabel('X (m)')
ax_3d.set_ylabel('Y (m)')
ax_3d.set_zlabel('Z (m)')
ax_3d.set_title('3D Inverted Pendulum Simulation', fontsize=14, fontweight='bold')

# Data display panels
ax_data = fig.add_subplot(gs[0, 1])
ax_data.set_xlim([0, 1])
ax_data.set_ylim([0, 1])
ax_data.axis('off')

ax_info = fig.add_subplot(gs[1, 1])
ax_info.set_xlim([0, 1])
ax_info.set_ylim([0, 1])
ax_info.axis('off')

# 3D plot elements
line, = ax_3d.plot([], [], [], 'b-', linewidth=4, label='Pendulum Rod')
mass, = ax_3d.plot([], [], [], 'ro', markersize=12, label='Mass')
trail, = ax_3d.plot([], [], [], 'g--', alpha=0.8, linewidth=2, label='Trajectory')
base, = ax_3d.plot([0], [0], [0], 'ko', markersize=10, label='Fixed Base')

# Coordinate system axes
ax_3d.plot([0, 0.5], [0, 0], [0, 0], 'r-', linewidth=3, alpha=0.7, label='X-axis')
ax_3d.plot([0, 0], [0, 0.5], [0, 0], 'g-', linewidth=3, alpha=0.7, label='Y-axis')
ax_3d.plot([0, 0], [0, 0], [0, 0.5], 'b-', linewidth=3, alpha=0.7, label='Z-axis')

ax_3d.legend(loc='upper right')

def animate(frame):
    """Animation function for real-time visualization"""
    
    # Update 3D pendulum visualization
    line.set_data([0, x_pos[frame]], [0, y_pos[frame]])
    line.set_3d_properties([0, z_pos[frame]])
    
    mass.set_data([x_pos[frame]], [y_pos[frame]])
    mass.set_3d_properties([z_pos[frame]])
    
    # Show trajectory trail (last 150 points)
    trail_start = max(0, frame - 150)
    trail.set_data(x_pos[trail_start:frame+1], y_pos[trail_start:frame+1])
    trail.set_3d_properties(z_pos[trail_start:frame+1])
    
    # Get current state values
    theta_deg = np.degrees(states[frame, 0])
    phi_deg = np.degrees(states[frame, 2])
    theta_dot = states[frame, 1]
    phi_dot = states[frame, 3]
    current_time = t[frame]
    
    # Update data display panel
    ax_data.clear()
    ax_data.set_xlim([0, 1])
    ax_data.set_ylim([0, 1])
    ax_data.axis('off')
    
    # Position information
    ax_data.text(0.05, 0.95, 'POSITION', fontsize=14, fontweight='bold', color='blue')
    ax_data.text(0.05, 0.85, f'X = {x_pos[frame]:.3f} m', fontsize=11, fontfamily='monospace')
    ax_data.text(0.05, 0.75, f'Y = {y_pos[frame]:.3f} m', fontsize=11, fontfamily='monospace')
    ax_data.text(0.05, 0.65, f'Z = {z_pos[frame]:.3f} m', fontsize=11, fontfamily='monospace')
    
    # Angular information
    ax_data.text(0.05, 0.50, 'ANGLES', fontsize=14, fontweight='bold', color='green')
    ax_data.text(0.05, 0.40, f'θ = {theta_deg:.1f}°', fontsize=11, fontfamily='monospace')
    ax_data.text(0.05, 0.30, f'φ = {phi_deg:.1f}°', fontsize=11, fontfamily='monospace')
    
    # Velocity information
    ax_data.text(0.05, 0.15, 'VELOCITIES', fontsize=14, fontweight='bold', color='red')
    ax_data.text(0.05, 0.05, f'θ̇ = {theta_dot:.3f} rad/s', fontsize=10, fontfamily='monospace')
    ax_data.text(0.05, -0.05, f'φ̇ = {phi_dot:.3f} rad/s', fontsize=10, fontfamily='monospace')
    
    # Update info panel
    ax_info.clear()
    ax_info.set_xlim([0, 1])
    ax_info.set_ylim([0, 1])
    ax_info.axis('off')
    
    # Simulation information
    ax_info.text(0.05, 0.8, 'SIMULATION INFO', fontsize=14, fontweight='bold', color='purple')
    ax_info.text(0.05, 0.65, f'Time: {current_time:.2f} s', fontsize=12, fontweight='bold')
    ax_info.text(0.05, 0.55, f'Frame: {frame+1}/{len(t)}', fontsize=10)
    
    # Initial conditions
    ax_info.text(0.05, 0.35, 'INITIAL CONDITIONS', fontsize=12, fontweight='bold', color='orange')
    ax_info.text(0.05, 0.25, f'θ₀ = {initial_state[0]:.2f} rad', fontsize=10, fontfamily='monospace')
    ax_info.text(0.05, 0.15, f'θ̇₀ = {initial_state[1]:.2f} rad/s', fontsize=10, fontfamily='monospace')
    ax_info.text(0.05, 0.05, f'φ₀ = {initial_state[2]:.2f} rad', fontsize=10, fontfamily='monospace')
    ax_info.text(0.05, -0.05, f'φ̇₀ = {initial_state[3]:.2f} rad/s', fontsize=10, fontfamily='monospace')
    
    return line, mass, trail

# Create and run animation
anim = animation.FuncAnimation(fig, animate, frames=len(t), interval=50, blit=False, repeat=True)

plt.tight_layout()
plt.show()

# Generate angle plots for analysis
fig_plots, axes = plt.subplots(2, 2, figsize=(12, 8))

# Theta vs time
axes[0, 0].plot(t, np.degrees(states[:, 0]), 'b-', linewidth=2)
axes[0, 0].set_title('Polar Angle θ vs Time', fontweight='bold')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('θ (degrees)')
axes[0, 0].grid(True, alpha=0.3)

# Phi vs time
axes[0, 1].plot(t, np.degrees(states[:, 2]), 'g-', linewidth=2)
axes[0, 1].set_title('Azimuthal Angle φ vs Time', fontweight='bold')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('φ (degrees)')
axes[0, 1].grid(True, alpha=0.3)

# Theta_dot vs time
axes[1, 0].plot(t, states[:, 1], 'r-', linewidth=2)
axes[1, 0].set_title('Angular Velocity θ̇ vs Time', fontweight='bold')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('θ̇ (rad/s)')
axes[1, 0].grid(True, alpha=0.3)

# Phi_dot vs time
axes[1, 1].plot(t, states[:, 3], 'm-', linewidth=2)
axes[1, 1].set_title('Angular Velocity φ̇ vs Time', fontweight='bold')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('φ̇ (rad/s)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print simulation summary
print("="*60)
print("3D INVERTED PENDULUM SIMULATION SUMMARY")
print("="*60)
print(f"Initial Conditions:")
print(f"  θ₀ = {initial_state[0]:.3f} rad ({np.degrees(initial_state[0]):.1f}°)")
print(f"  θ̇₀ = {initial_state[1]:.3f} rad/s")
print(f"  φ₀ = {initial_state[2]:.3f} rad ({np.degrees(initial_state[2]):.1f}°)")
print(f"  φ̇₀ = {initial_state[3]:.3f} rad/s")
print(f"\nSimulation Parameters:")
print(f"  Mass: {m} kg")
print(f"  Length: {L} m") 
print(f"  Gravity: {g} m/s²")
print(f"  Simulation time: {t[-1]} seconds")
print(f"  Time steps: {len(t)}")
print("="*60)