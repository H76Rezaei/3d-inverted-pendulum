import numpy as np
import time

class iLQR3DPendulum:
    """
    Iterative Linear Quadratic Regulator for 3D Inverted Pendulum
    """
    
    def __init__(self, 
                 T=50,          # Horizon length (time steps) 
                 dt=0.02,       # Time step
                 M=2.0, m=1.0, L=1.5, g=9.81,
                 b_theta=0.5, b_phi=0.5,
                 b_cart=0.01,   # Cart friction
                 max_iters=100,
                 verbose=False):  
        
        # System parameters
        self.M, self.m, self.L, self.g = M, m, L, g
        self.b_theta, self.b_phi = b_theta, b_phi
        self.b_cart = b_cart
        self.dt = dt
        self.T = T
        self.max_iters = max_iters
        self.verbose = verbose
        
        # Dimensions
        self.nx = 8  # state dimension
        self.nu = 2  # control dimension
        
        
        # Version 1: Tutorial-style cost structure
        self.Q_tutorial = np.diag([10.0, 1.0, 10.0, 1.0, 100.0, 10.0, 100.0, 10.0])
        self.R_tutorial = np.diag([0.1, 0.1])
        self.Qf_tutorial = self.Q_tutorial * 5

        # Version 2: Strict near upright position
        self.Q_strict = np.diag([10.0, 1.0, 10.0, 1.0, 200.0, 20.0, 100.0, 10.0])
        self.R_strict = np.diag([0.1, 0.1])
        self.Qf_strict = self.Q_strict * 8
        
        # Current cost matrices (can switch between versions)
        self.cost_version = "strict"  # or "original", "tutorial"
        self._update_cost_matrices()
        
        # Initialize trajectory storage
        self.reset_trajectory()
        
        # Regularization parameters
        self.reg_min = 1e-6
        self.reg_max = 1e6
        self.reg_factor = 2.0
        
        # Line search parameters
        self.line_search_alphas = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        self.min_cost_improvement = 1e-4
        
    def _update_cost_matrices(self):
        """Update cost matrices based on current version"""
        if self.cost_version == "strict":
            self.Q = self.Q_strict.copy()
            self.R = self.R_strict.copy()
            self.Qf = self.Qf_strict.copy()
        else:  # tutorial
            self.Q = self.Q_tutorial.copy()
            self.R = self.R_tutorial.copy()
            self.Qf = self.Qf_tutorial.copy()
    
    def set_cost_version(self, version):
        """Switch between cost function versions"""
        if version in ["original", "tutorial", "strict"]:
            self.cost_version = version
            self._update_cost_matrices()
        else:
            raise ValueError("Cost version must be 'original', 'tutorial', or 'strict'")
    
    def reset_trajectory(self):
        """Initialize nominal trajectory"""
        self.x_traj = np.zeros((self.T + 1, self.nx))
        self.u_traj = np.zeros((self.T, self.nu))
        
        # Initialize gains and value function
        self.K = np.zeros((self.T, self.nu, self.nx))  # Feedback gains
        self.k = np.zeros((self.T, self.nu))           # Feedforward terms
        self.V = np.zeros(self.T + 1)                  # Value function
        self.Vx = np.zeros((self.T + 1, self.nx))      # Value gradient
        self.Vxx = np.zeros((self.T + 1, self.nx, self.nx))  # Value Hessian

    def dynamics(self, x, u):
        """3D pendulum dynamics with proper Lagrangian formulation"""
        x_c, x_c_dot, y_c, y_c_dot, theta, theta_dot, phi, phi_dot = x
        Fx, Fy = u
        
        # Trigonometric terms
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        
        # Prevent singularities
        if abs(st) < 1e-6:
            st = np.sign(st) * 1e-6 if st != 0 else 1e-6
        
        # Mass matrix components
        M_total = self.M + self.m
        mL = self.m * self.L
        mL2 = self.m * self.L**2
        
        # Mass matrix (4x4 for accelerations)
        M_mat = np.array([
            [M_total, 0, mL*ct*cp, -mL*st*sp],
            [0, M_total, mL*ct*sp, mL*st*cp],
            [mL*ct*cp, mL*ct*sp, mL2, 0],
            [-mL*st*sp, mL*st*cp, 0, mL2*st**2]
        ])
        
        # Right-hand side: Forces and moments
        rhs = np.array([
            # Cart x-equation: External force + coupling - friction
            Fx + mL*(theta_dot**2*st*cp + phi_dot**2*st*cp + 2*theta_dot*phi_dot*ct*sp) - self.b_cart*x_c_dot,
            # Cart y-equation: External force + coupling - friction  
            Fy + mL*(theta_dot**2*st*sp + phi_dot**2*st*sp - 2*theta_dot*phi_dot*ct*cp) - self.b_cart*y_c_dot,
            # Theta equation: Gravity + coupling - damping
            -self.m*self.g*self.L*st + mL2*st*ct*phi_dot**2 - self.b_theta*theta_dot,
            # Phi equation: Coupling - damping
            -2*mL2*st*ct*theta_dot*phi_dot - self.b_phi*phi_dot
        ])
        
        # Solve for accelerations with regularization
        try:
            M_reg = M_mat + np.eye(4) * 1e-12
            accel = np.linalg.solve(M_reg, rhs)
        except np.linalg.LinAlgError:
            accel = np.linalg.lstsq(M_mat, rhs, rcond=1e-6)[0]
        
        # Clip accelerations for numerical stability
        accel = np.clip(accel, -50, 50)
        
        # Return state derivative
        return np.array([x_c_dot, accel[0], y_c_dot, accel[1],
                        theta_dot, accel[2], phi_dot, accel[3]])

    def discrete_dynamics(self, x, u):
        """Discrete dynamics using RK4 integration"""
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 0.5*self.dt*k1, u)
        k3 = self.dynamics(x + 0.5*self.dt*k2, u)
        k4 = self.dynamics(x + self.dt*k3, u)
        
        x_next = x + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Wrap angles
        x_next[4] = self._wrap_angle(x_next[4])  # theta
        x_next[6] = self._wrap_angle(x_next[6])  # phi
        
        return x_next

    def linearize_dynamics(self, x, u, eps=1e-6):
        """Compute linearization A, B matrices using finite differences"""
        A = np.zeros((self.nx, self.nx))
        B = np.zeros((self.nx, self.nu))
        
        # Get nominal next state
        x_nom = self.discrete_dynamics(x, u)
        
        # Compute A matrix (df/dx)
        for i in range(self.nx):
            x_pert = x.copy()
            x_pert[i] += eps
            x_pert_next = self.discrete_dynamics(x_pert, u)
            A[:, i] = (x_pert_next - x_nom) / eps
        
        # Compute B matrix (df/du)
        for i in range(self.nu):
            u_pert = u.copy()
            u_pert[i] += eps
            x_pert_next = self.discrete_dynamics(x, u_pert)
            B[:, i] = (x_pert_next - x_nom) / eps
        
        return A, B

    def stage_cost(self, x, u, t):
        """Stage cost function with adaptive weights near upright position"""
        # Target state (upright pendulum at origin)
        x_target = np.array([0, 0, 0, 0, np.pi, 0, 0, 0])
        
        # State error with angle wrapping
        x_err = x - x_target
        x_err[4] = self._wrap_angle(x_err[4])  # theta error
        x_err[6] = self._wrap_angle(x_err[6])  # phi error
        
        # Adaptive cost weights - increase penalty when close to upright
        Q_adaptive = self.Q.copy()
        if self.cost_version == "strict":
            # Calculate distance from upright (theta = pi)
            theta_error = abs(x_err[4])
            # Increase theta position and velocity weights when close to upright
            if theta_error < np.pi/6:  # Within 30 degrees of upright
                # Exponential increase in penalty as we get closer
                theta_weight_factor = 1 + 10 * np.exp(-theta_error * 3)
                Q_adaptive[4, 4] *= theta_weight_factor  # theta position
                Q_adaptive[5, 5] *= theta_weight_factor  # theta velocity
        
        if t == self.T:  # Terminal cost
            Qf_adaptive = self.Qf.copy()
            if self.cost_version == "strict":
                theta_error = abs(x_err[4])
                if theta_error < np.pi/6:
                    theta_weight_factor = 1 + 20 * np.exp(-theta_error * 3)
                    Qf_adaptive[4, 4] *= theta_weight_factor
                    Qf_adaptive[5, 5] *= theta_weight_factor
            return 0.5 * x_err.T @ Qf_adaptive @ x_err
        else:  # Running cost
            return 0.5 * x_err.T @ Q_adaptive @ x_err + 0.5 * u.T @ self.R @ u

    def cost_derivatives(self, x, u, t):
        """Compute cost derivatives (lx, lu, lxx, luu, lux) with adaptive weights"""
        # Target state
        x_target = np.array([0, 0, 0, 0, np.pi, 0, 0, 0])
        
        # State error with angle wrapping
        x_err = x - x_target
        x_err[4] = self._wrap_angle(x_err[4])
        x_err[6] = self._wrap_angle(x_err[6])
        
        # Adaptive cost weights - same logic as stage_cost
        Q_adaptive = self.Q.copy()
        if self.cost_version == "strict":
            theta_error = abs(x_err[4])
            if theta_error < np.pi/6:
                theta_weight_factor = 1 + 10 * np.exp(-theta_error * 3)
                Q_adaptive[4, 4] *= theta_weight_factor
                Q_adaptive[5, 5] *= theta_weight_factor
        
        if t == self.T:  # Terminal cost
            Qf_adaptive = self.Qf.copy()
            if self.cost_version == "strict":
                theta_error = abs(x_err[4])
                if theta_error < np.pi/6:
                    theta_weight_factor = 1 + 20 * np.exp(-theta_error * 3)
                    Qf_adaptive[4, 4] *= theta_weight_factor
                    Qf_adaptive[5, 5] *= theta_weight_factor
            lx = Qf_adaptive @ x_err
            lu = np.zeros(self.nu)
            lxx = Qf_adaptive
            luu = np.zeros((self.nu, self.nu))
            lux = np.zeros((self.nu, self.nx))
        else:  # Running cost
            lx = Q_adaptive @ x_err
            lu = self.R @ u
            lxx = Q_adaptive
            luu = self.R
            lux = np.zeros((self.nu, self.nx))
            
        return lx, lu, lxx, luu, lux

    def backward_pass(self, reg=0):
        """Backward pass to compute gains and value function"""
        # Initialize value function at terminal time
        x_T = self.x_traj[self.T]
        lx, lu, lxx, luu, lux = self.cost_derivatives(x_T, np.zeros(self.nu), self.T)
        
        self.V[self.T] = self.stage_cost(x_T, np.zeros(self.nu), self.T)
        self.Vx[self.T] = lx
        self.Vxx[self.T] = lxx
        
        # Backward recursion
        for t in range(self.T - 1, -1, -1):
            x_t = self.x_traj[t]
            u_t = self.u_traj[t]
            
            # Get dynamics derivatives
            A, B = self.linearize_dynamics(x_t, u_t)
            
            # Get cost derivatives
            lx, lu, lxx, luu, lux = self.cost_derivatives(x_t, u_t, t)
            
            # Q-function derivatives (Bellman equation)
            Qx = lx + A.T @ self.Vx[t+1]
            Qu = lu + B.T @ self.Vx[t+1]
            Qxx = lxx + A.T @ self.Vxx[t+1] @ A
            Quu = luu + B.T @ self.Vxx[t+1] @ B
            Qux = lux + B.T @ self.Vxx[t+1] @ A
            
            # Regularization
            Quu_reg = Quu + reg * np.eye(self.nu)
            
            # Check if Quu is positive definite
            try:
                # Cholesky decomposition to check positive definiteness
                L = np.linalg.cholesky(Quu_reg)
                
                # Compute gains
                self.k[t] = -np.linalg.solve(Quu_reg, Qu)
                self.K[t] = -np.linalg.solve(Quu_reg, Qux)
                
            except np.linalg.LinAlgError:
                # Not positive definite, increase regularization
                if self.verbose:
                    print(f"  Backward pass failed at t={t}, Quu not PD")
                return False
            
            # Update value function
            self.V[t] = (self.stage_cost(x_t, u_t, t) + 
                        self.k[t].T @ Qu + 
                        0.5 * self.k[t].T @ Quu @ self.k[t])
            
            self.Vx[t] = (Qx + self.K[t].T @ Quu @ self.k[t] + 
                         self.K[t].T @ Qu + Qux.T @ self.k[t])
            
            self.Vxx[t] = (Qxx + self.K[t].T @ Quu @ self.K[t] + 
                          self.K[t].T @ Qux + Qux.T @ self.K[t])
            
            # Ensure Vxx stays symmetric
            self.Vxx[t] = 0.5 * (self.Vxx[t] + self.Vxx[t].T)
        
        return True

    def forward_pass(self, x0, alpha=1.0):
        """Forward pass with line search parameter alpha"""
        x_new = np.zeros((self.T + 1, self.nx))
        u_new = np.zeros((self.T, self.nu))
        
        x_new[0] = x0
        cost_new = 0
        
        for t in range(self.T):
            # State deviation
            dx = x_new[t] - self.x_traj[t]
            
            # Control update with line search
            u_new[t] = self.u_traj[t] + alpha * self.k[t] + self.K[t] @ dx
            
            # Clip control for physical limits
            u_new[t] = np.clip(u_new[t], -100, 100)
            
            # Add stage cost
            cost_new += self.stage_cost(x_new[t], u_new[t], t)
            
            # Forward dynamics
            x_new[t+1] = self.discrete_dynamics(x_new[t], u_new[t])
        
        # Add terminal cost
        cost_new += self.stage_cost(x_new[self.T], np.zeros(self.nu), self.T)
        
        return x_new, u_new, cost_new

    def line_search(self, x0, cost_old):
        """Perform line search to find suitable step size"""
        for alpha in self.line_search_alphas:
            x_new, u_new, cost_new = self.forward_pass(x0, alpha)
            
            if cost_new < cost_old:
                return x_new, u_new, cost_new, alpha
        
        # If all line search fails, return smallest alpha
        return self.forward_pass(x0, self.line_search_alphas[-1]) + (self.line_search_alphas[-1],)

    def solve(self, x0, x_init=None, u_init=None):
        """Main iLQR solver"""
        if self.verbose:
            print(f"iLQR solve started with cost version: {self.cost_version}")
        
        # Initialize trajectory
        if x_init is not None:
            self.x_traj = x_init.copy()
        else:
            # Initialize with straight line to target
            x_target = np.array([0, 0, 0, 0, np.pi, 0, 0, 0])
            for t in range(self.T + 1):
                alpha = t / self.T
                self.x_traj[t] = (1 - alpha) * x0 + alpha * x_target
        
        if u_init is not None:
            self.u_traj = u_init.copy()
        else:
            # Initialize with zero controls
            self.u_traj = np.zeros((self.T, self.nu))
        
        # Forward simulate initial trajectory for consistency
        self._forward_simulate_trajectory(x0)
        
        # Calculate initial cost
        cost_old = self._calculate_trajectory_cost()
        
        if self.verbose:
            print(f"Initial cost: {cost_old:.6f}")
        
        reg = self.reg_min
        
        for iteration in range(self.max_iters):
            start_time = time.time()
            
            # Backward pass with regularization loop
            backward_success = False
            reg_iterations = 0
            
            while not backward_success and reg < self.reg_max:
                backward_success = self.backward_pass(reg)
                if not backward_success:
                    reg *= self.reg_factor
                    reg_iterations += 1
                    if reg_iterations > 10:  # Prevent infinite loop
                        break
            
            if not backward_success:
                if self.verbose:
                    print(f"  Backward pass failed, regularization too high: {reg}")
                break
            
            # Forward pass with line search
            x_new, u_new, cost_new, alpha = self.line_search(x0, cost_old)
            
            # Check for improvement
            cost_improvement = cost_old - cost_new
            
            if cost_improvement > 0:
                # Accept the step
                self.x_traj = x_new
                self.u_traj = u_new
                cost_old = cost_new
                
                # Reduce regularization on successful step
                reg = max(reg / self.reg_factor, self.reg_min)
                
                solve_time = (time.time() - start_time) * 1000  # ms
                
                if self.verbose:
                    print(f"  Iter {iteration+1:2d}: Cost = {cost_new:.6f}, "
                          f"ΔCost = {cost_improvement:.2e}, α = {alpha:.3f}, "
                          f"reg = {reg:.2e}, time = {solve_time:.1f}ms")
                
                # Check convergence
                if cost_improvement < self.min_cost_improvement:
                    if self.verbose:
                        print(f"  Converged after {iteration + 1} iterations")
                    return True
            else:
                if self.verbose:
                    print(f"  Iter {iteration+1:2d}: No improvement found")
                break
        
        return iteration < self.max_iters - 1

    def _forward_simulate_trajectory(self, x0):
        """Forward simulate trajectory for consistency"""
        self.x_traj[0] = x0
        for t in range(self.T):
            self.x_traj[t+1] = self.discrete_dynamics(self.x_traj[t], self.u_traj[t])

    def _calculate_trajectory_cost(self):
        """Calculate total cost of current trajectory"""
        cost = 0
        for t in range(self.T):
            cost += self.stage_cost(self.x_traj[t], self.u_traj[t], t)
        cost += self.stage_cost(self.x_traj[self.T], np.zeros(self.nu), self.T)
        return cost

    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def get_control(self, x, t=0):
        """Get control for current state (for real-time use)"""
        if t >= self.T:
            return np.zeros(self.nu)
        
        # State deviation from nominal trajectory
        dx = x - self.x_traj[t]
        
        # Apply feedback control law
        u = self.u_traj[t] + self.K[t] @ dx
        
        return np.clip(u, -100, 100)

    def print_summary(self):
        """Print summary of solver settings"""
        print(f"\niLQR Controller Summary:")
        print(f"  Horizon: {self.T} steps")
        print(f"  Time step: {self.dt} s")
        print(f"  Cost version: {self.cost_version}")
        print(f"  Max iterations: {self.max_iters}")
        print(f"  Regularization: [{self.reg_min:.1e}, {self.reg_max:.1e}]")
        print(f"  Line search alphas: {self.line_search_alphas}")


# Example usage and testing
if __name__ == "__main__":
    # Create controller
    ilqr = iLQR3DPendulum(T=30, dt=0.02, verbose=True)
    
    # Print summary
    ilqr.print_summary()
    
    # Test initial state (pendulum hanging down)
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0])
    
    print(f"\nTesting with cost version: {ilqr.cost_version}")
    start_time = time.time()
    success = ilqr.solve(x0)
    solve_time = time.time() - start_time
    
    if success:
        final_cost = ilqr._calculate_trajectory_cost()
        print(f"\nSolver succeeded in {solve_time:.2f}s")
        print(f"Final cost: {final_cost:.6f}")
        
        # Test control at initial state
        u_test = ilqr.get_control(x0)
        print(f"Control at x0: Fx={u_test[0]:.2f}N, Fy={u_test[1]:.2f}N")
        
        # Test with original cost function
        print(f"\nSwitching to original cost function...")
        ilqr.set_cost_version("original")
        start_time = time.time()
        success_orig = ilqr.solve(x0)
        solve_time_orig = time.time() - start_time
        
        if success_orig:
            final_cost_orig = ilqr._calculate_trajectory_cost()
            print(f"Original cost solver succeeded in {solve_time_orig:.2f}s")
            print(f"Final cost (original): {final_cost_orig:.6f}")
        else:
            print("Original cost solver failed")
            
    else:
        print(f"\nSolver failed after {solve_time:.2f}s")