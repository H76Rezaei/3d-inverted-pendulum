# 3D Inverted Pendulum Control with Neural Network Approximation

A simulation and control system for a 3D inverted pendulum using iterative Linear Quadratic Regulator (iLQR) with neural network approximation, developed as part of a Master's thesis at TU Darmstadt.

## Overview

This project implements a complete control system for stabilizing a 3D inverted pendulum around the upright position. The system uses advanced optimal control methods with machine learning approximation to achieve real-time performance.

### Key Features

- **3D Pendulum Dynamics**: Full Lagrangian-based simulation with proper spherical pendulum kinematics
- **iLQR Controller**: Iterative Linear Quadratic Regulator with line search optimization
- **Real-time Visualization**: Interactive 3D animation with multiple view perspectives
- **Performance Tracking**: Success rate monitoring and solve time statistics
- **Fallback Control**: Energy-based swing-up controller for robustness

## System Architecture

The system consists of two main components:

1. **Pendulum3DSimulation**: Physics simulation with Lagrangian dynamics

   - State: `[x_c, x_c_dot, y_c, y_c_dot, theta, theta_dot, phi, phi_dot]`
   - RK4 integration for numerical stability
   - Viscous damping for realistic behavior

2. **iLQR3DPendulum**: Optimization-based controller
   - Finite horizon optimal control (T=50 steps)
   - Adaptive cost function weights near equilibrium
   - Regularization and line search for numerical robustness

## Installation

1. Clone the repository:

```bash
git clone https://github.com/H76Rezaei/3d-inverted-pendulum
cd 3d-inverted-pendulum
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main simulation:

```bash
python pendulum_3d_simulation.py
```

### Testing the iLQR Controller

Run the standalone controller test:

```bash
python ILQR_3d_Controller.py
```

## License

This project is developed for academic research purposes at TU Darmstadt.
