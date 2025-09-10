import numpy as np


class iLQR3DPendulum:
    """
    Iterative Linear Quadratic Regulator for 3D Inverted Pendulum
    """
    
    def __init__(self, 
                 T=50,          # Horizon length (time steps) 
                 dt=0.02,       # Time step
                 M=2.0, m=1.0, L=1.5, g=9.81,
                 b_theta=0.5, b_phi=0.5,
                 b_cart=0.01, b_cubic=0.001,  # Additional viscous damping parameters (conservative)
                 max_iters=100):  
        
        # System parameters
        self.M, self.m, self.L, self.g = M, m, L, g
        self.b_theta, self.b_phi = b_theta, b_phi
        self.b_cart, self.b_cubic = b_cart, b_cubic  # Viscous damping coefficients
        self.dt = dt
        self.T = T
        self.max_iters = max_iters
        
        # Dimensions
        self.nx = 8  # state dimension
        self.nu = 2  # control dimension
        
        # Cost matrices 
        self.Q = np.diag([1.0, 0.1, 1.0, 0.1, 50.0, 5.0, 50.0, 5.0])
        self.R = np.diag([0.01, 0.01])
        self.Qf = self.Q * 10
        
        # Initialize trajectory storage
        self.reset_trajectory()
        
        # Regularization (more conservative for stability)
        self.reg_min = 1e-6
        self.reg_max = 1e10
        self.reg_factor = 2.0
        
    def reset_trajectory(self):
        """Initialize nominal trajectory"""
        self.x_traj = np.zeros((self.T + 1, self.nx))
        self.u_traj = np.zeros((self.T, self.nu))
        
    def _dynamics_only(self, x, u):
        """3D pendulum dynamics - returns only f"""
        x_c, x_c_dot, y_c, y_c_dot, theta, theta_dot, phi, phi_dot = x
        Fx, Fy = u
        
        # Trigonometric terms
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        
        # Avoid singularities
        if abs(st) < 1e-6:
            st = 1e-6 * np.sign(st) if st != 0 else 1e-6
        
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
        
        # Right-hand side with enhanced viscous damping (smooth cubic damping)
        rhs = np.array([
            # Cart x-equation with viscous damping:
            Fx + mL*(theta_dot**2*st*cp + phi_dot**2*st*cp - 2*theta_dot*phi_dot*ct*sp) - self.b_cart*x_c_dot,
            # Cart y-equation with viscous damping: 
            Fy + mL*(theta_dot**2*st*sp + phi_dot**2*st*sp + 2*theta_dot*phi_dot*ct*cp) - self.b_cart*y_c_dot,
            # Theta equation with enhanced viscous damping (smooth cubic): 
            -self.m*self.g*self.L*st - self.b_theta*theta_dot - self.b_cubic*theta_dot*theta_dot**2,
            # Phi equation with enhanced viscous damping (smooth cubic): 
            -self.b_phi*phi_dot - self.b_cubic*phi_dot*phi_dot**2 - 2*mL2*st*ct*theta_dot*phi_dot + mL*st*cp*(Fx/M_total) + mL*st*sp*(Fy/M_total)
        ])
        
        # Solve for accelerations with regularization
        try:
            M_reg = M_mat + np.eye(4) * 1e-12  # Minimal regularization
            accel = np.linalg.solve(M_reg, rhs)
        except np.linalg.LinAlgError:
            accel = np.linalg.lstsq(M_mat, rhs, rcond=1e-6)[0]
        
        # Clip accelerations to prevent numerical issues
        accel = np.clip(accel, -50, 50)
        
        # State derivative
        f = np.array([x_c_dot, accel[0], y_c_dot, accel[1],
                     theta_dot, accel[2], phi_dot, accel[3]])
        
        return f

    def dynamics(self, x, u):
        """3D pendulum dynamics - returns (f, fx, fu)"""
        # Get the dynamics without Jacobians
        f = self._dynamics_only(x, u)
        
        # Compute Jacobians numerically for robustness
        fx = self._numerical_jacobian_x(x, u)
        fu = self._numerical_jacobian_u(x, u)
        
        return f, fx, fu
    
    def _numerical_jacobian_x(self, x, u, eps=1e-6):
        """Numerical Jacobian w.r.t. state"""
        fx = np.zeros((self.nx, self.nx))
        f0 = self._dynamics_only(x, u)
        
        for i in range(self.nx):
            x_pert = x.copy()
            x_pert[i] += eps
            f_pert = self._dynamics_only(x_pert, u)
            fx[:, i] = (f_pert - f0) / eps
            
        # Clip Jacobian to prevent numerical issues
        fx = np.clip(fx, -1000, 1000)
        return fx
    
    def _numerical_jacobian_u(self, x, u, eps=1e-6):
        """Numerical Jacobian w.r.t. control"""
        fu = np.zeros((self.nx, self.nu))
        f0 = self._dynamics_only(x, u)
        
        for i in range(self.nu):
            u_pert = u.copy()
            u_pert[i] += eps
            f_pert = self._dynamics_only(x, u_pert)
            fu[:, i] = (f_pert - f0) / eps
            
        # Clip Jacobian to prevent numerical issues
        fu = np.clip(fu, -1000, 1000)
        return fu
    
    def cost(self, x, u, t):
        """Stage cost"""
        if t == self.T:  # Terminal cost
            # Error to target (upright pendulum at origin)
            x_target = np.array([0, 0, 0, 0, np.pi, 0, 0, 0])
            error = x - x_target
            # Handle angle wrapping for theta and phi
            error[4] = self._wrap_angle(error[4])
            error[6] = self._wrap_angle(error[6])
            
            return 0.5 * error.T @ self.Qf @ error
        else:  # Running cost
            # Error to target
            x_target = np.array([0, 0, 0, 0, np.pi, 0, 0, 0])
            error = x - x_target
            # Handle angle wrapping
            error[4] = self._wrap_angle(error[4])
            error[6] = self._wrap_angle(error[6])
            
            return 0.5 * error.T @ self.Q @ error + 0.5 * u.T @ self.R @ u
    
    def cost_derivatives(self, x, u, t):
        """Cost derivatives (lx, lu, lxx, luu, lux)"""
        if t == self.T:  # Terminal cost
            x_target = np.array([0, 0, 0, 0, np.pi, 0, 0, 0])
            error = x - x_target
            error[4] = self._wrap_angle(error[4])
            error[6] = self._wrap_angle(error[6])
            
            lx = self.Qf @ error
            lu = np.zeros(self.nu)
            lxx = self.Qf
            luu = np.zeros((self.nu, self.nu))
            lux = np.zeros((self.nu, self.nx))
        else:  # Running cost
            x_target = np.array([0, 0, 0, 0, np.pi, 0, 0, 0])
            error = x - x_target
            error[4] = self._wrap_angle(error[4])
            error[6] = self._wrap_angle(error[6])
            
            lx = self.Q @ error
            lu = self.R @ u
            lxx = self.Q
            luu = self.R
            lux = np.zeros((self.nu, self.nx))
            
        return lx, lu, lxx, luu, lux
    
    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def forward_pass(self, x0, alpha=1.0):
        """Forward pass with line search"""
        x_new = np.zeros((self.T + 1, self.nx))
        u_new = np.zeros((self.T, self.nu))
        
        x_new[0] = x0
        cost_new = 0
        
        for t in range(self.T):
            # Control update
            dx = x_new[t] - self.x_traj[t]
            u_new[t] = self.u_traj[t] + alpha * self.k[t] + self.K[t] @ dx
            
            # Clip control
            u_new[t] = np.clip(u_new[t], -50, 50)
            
            # Add cost
            cost_new += self.cost(x_new[t], u_new[t], t)
            
            # Forward dynamics (discrete)
            f, _, _ = self.dynamics(x_new[t], u_new[t])
            x_new[t+1] = x_new[t] + self.dt * f
            
            # Wrap angles
            x_new[t+1, 4] = self._wrap_angle(x_new[t+1, 4])
            x_new[t+1, 6] = self._wrap_angle(x_new[t+1, 6])
        
        # Terminal cost
        cost_new += self.cost(x_new[self.T], np.zeros(self.nu), self.T)
        
        return x_new, u_new, cost_new
    
    def backward_pass(self, reg=0):
        """Backward pass"""
        # Initialize value function at terminal time
        lx, lu, lxx, luu, lux = self.cost_derivatives(self.x_traj[self.T], np.zeros(self.nu), self.T)
        Vx = lx
        Vxx = lxx
        
        self.k = np.zeros((self.T, self.nu))
        self.K = np.zeros((self.T, self.nu, self.nx))
        
        for t in range(self.T - 1, -1, -1):
            x_t = self.x_traj[t]
            u_t = self.u_traj[t]
            
            # Get dynamics and cost derivatives
            f, fx, fu = self.dynamics(x_t, u_t)
            lx, lu, lxx, luu, lux = self.cost_derivatives(x_t, u_t, t)
            
            # Discrete-time derivatives (first-order approximation)
            A = np.eye(self.nx) + self.dt * fx
            B = self.dt * fu
            
            # Q-function derivatives
            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A
            
            # Regularization
            Quu_reg = Quu + reg * np.eye(self.nu)
            
            # Check if Quu is positive definite
            try:
                L = np.linalg.cholesky(Quu_reg)
                # Control law
                self.k[t] = -np.linalg.solve(Quu_reg, Qu)
                self.K[t] = -np.linalg.solve(Quu_reg, Qux)
            except np.linalg.LinAlgError:
                # If not PD, increase regularization
                return False
            
            # Value function update - Keep original working form
            Vx = Qx + self.K[t].T @ Quu @ self.k[t] + self.K[t].T @ Qu + Qux.T @ self.k[t]
            Vxx = Qxx + self.K[t].T @ Quu @ self.K[t] + self.K[t].T @ Qux + Qux.T @ self.K[t]
            
            # Ensure Vxx stays symmetric
            Vxx = 0.5 * (Vxx + Vxx.T)
        
        return True
    
    def solve(self, x0, x_init=None, u_init=None, verbose=False):
        """Main iLQR solver"""
        # Initialize trajectory
        if x_init is not None:
            self.x_traj = x_init.copy()
        else:
            # Smart initialization:
            x_target = np.array([0, 0, 0, 0, np.pi, 0, 0, 0])
            
            # Check if we're far from target 
            theta_error = abs(self._wrap_angle(x0[4] - np.pi))
            phi_error = abs(self._wrap_angle(x0[6]))
            
            if theta_error > np.pi/4 or phi_error > np.pi/4:  # Far from target
                # Create a more aggressive initial trajectory
                for t in range(self.T + 1):
                    alpha = t / self.T
                    # Use a more aggressive interpolation for swing-up
                    self.x_traj[t] = (1 - alpha) * x0 + alpha * x_target
                    # Add some energy to help with swing-up
                    if t < self.T//2:
                        self.x_traj[t, 5] = 2.0 * (1 - 2*t/self.T)  # Initial angular velocity
            else:
                # Close to target, use gentle interpolation
                for t in range(self.T + 1):
                    alpha = t / self.T
                    self.x_traj[t] = (1 - alpha) * x0 + alpha * x_target
        
        if u_init is not None:
            self.u_traj = u_init.copy()
        else:
            self.u_traj = np.zeros((self.T, self.nu))
        
        # Forward simulate initial trajectory to get consistent dynamics
        x_sim = np.zeros((self.T + 1, self.nx))
        x_sim[0] = x0
        for t in range(self.T):
            f, _, _ = self.dynamics(x_sim[t], self.u_traj[t])
            x_sim[t+1] = x_sim[t] + self.dt * f
            x_sim[t+1, 4] = self._wrap_angle(x_sim[t+1, 4])
            x_sim[t+1, 6] = self._wrap_angle(x_sim[t+1, 6])
        self.x_traj = x_sim
        
        # Initial cost
        cost_old = sum(self.cost(self.x_traj[t], self.u_traj[t], t) for t in range(self.T))
        cost_old += self.cost(self.x_traj[self.T], np.zeros(self.nu), self.T)
        
        reg = self.reg_min
        
        for iteration in range(self.max_iters):
            # Backward pass
            improved = False
            while not improved and reg < self.reg_max:
                if self.backward_pass(reg):
                    improved = True
                else:
                    reg *= self.reg_factor
                    if verbose:
                        print(f"Increasing regularization to {reg}")
            
            if not improved:
                if verbose:
                    print("Failed to improve - regularization too high")
                break
            
            # Forward pass with line search
            alpha = 1.0
            for _ in range(10):  # Line search iterations
                x_new, u_new, cost_new = self.forward_pass(x0, alpha)
                
                if cost_new < cost_old:
                    # Accept step
                    self.x_traj = x_new
                    self.u_traj = u_new
                    cost_improvement = cost_old - cost_new
                    cost_old = cost_new
                    
                    if verbose:
                        print(f"Iter {iteration}: Cost = {cost_new:.4f}, Improvement = {cost_improvement:.6f}")
                    
                    # Check convergence - more lenient for difficult cases
                    if cost_improvement < 1e-3:  # Relaxed convergence criterion
                        if verbose:
                            print(f"Converged after {iteration + 1} iterations")
                        return True
                    
                    # Reduce regularization on successful step
                    reg = max(reg / self.reg_factor, self.reg_min)
                    break
                else:
                    # Reduce step size
                    alpha *= 0.5
                    if alpha < 1e-4:
                        if verbose:
                            print("Line search failed")
                        break
            else:
                if verbose:
                    print("Line search failed")
                break
        
        return iteration < self.max_iters - 1
