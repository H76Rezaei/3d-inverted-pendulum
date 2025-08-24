# mpc_controller_casadi.py
from time import time
import numpy as np
import casadi as ca
import os
import sys

class NonlinearMPC:
    """
    Nonlinear MPC for the 3D inverted pendulum using CasADi + Ipopt.
    - States (nx=8): [x_c, x_c_dot, y_c, y_c_dot, theta, theta_dot, phi, phi_dot]
    - Controls (nu=2): [Fx, Fy]
    - Discretization: RK4 with given dt
    """
    def __init__(self,
                 N=20,                   # Prediction horizon
                 dt=0.01,                # Time step
                 M=2.0, m=1.0, L=1.5, g=9.81,  # Physical parameters
                 b_theta=0.1, b_phi=0.1,  # Damping coefficients
                 u_max=150.0,            # Control limits
                 u_min=-150.0,
                 Q=None, R=None, P=None):
        # Horizon and timing
        self.N = int(N)
        self.dt = float(dt)

        # Physical parameters
        self.M = float(M)
        self.m = float(m)
        self.L = float(L)
        self.g = float(g)
        self.b_theta = float(b_theta)
        self.b_phi = float(b_phi)

        # Dimensions
        self.nx = 8
        self.nu = 2

        # Cost matrices
        if Q is None:
            Q = np.diag([
                1000.0,     # x_c position
                50.0,       # x_c velocity
                1000.0,     # y_c position
                50.0,       # y_c velocity
                200000.0,   # theta deviation
                20000.0,    # theta_dot
                1000.0,     # phi deviation
                100.0       # phi_dot
            ])
        if R is None:
            R = 0.01 * np.eye(self.nu)
        if P is None:
            P = Q * 20.0

        self.Q = ca.DM(Q)
        self.R = ca.DM(R)
        self.P = ca.DM(P)

        # Control bounds
        self.u_min = float(u_min)
        self.u_max = float(u_max)

        # Solver options
        s_opts = {
            'ipopt.max_iter': 200,
            'ipopt.tol': 1e-6,
            'ipopt.constr_viol_tol': 1e-6,
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'ipopt.output_file': '',
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_init': 1e-2,
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.limited_memory_max_history': 10,
            'ipopt.nlp_scaling_method': 'gradient-based',
            'ipopt.linear_solver': 'mumps',
            'ipopt.max_cpu_time': 0.05,
            'ipopt.fast_step_computation': 'yes',
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.adaptive_mu_globalization': 'kkt-error',
            'ipopt.bound_relax_factor': 0.0,
            'print_time': 0
        }
        self.ipopt_opts = s_opts

        # Build symbolic dynamics and NLP
        self._build_symbolic_model()
        self._build_nlp()

        # Storage for warm-start
        self.prev_sol = None
        self.prev_decision = None

    def _continuous_dynamics(self, x, u):
        """Continuous dynamics for 3D inverted pendulum"""
        M = self.M; m = self.m; L = self.L; g = self.g
        b_theta = self.b_theta; b_phi = self.b_phi

        x_c = x[0]; x_c_dot = x[1]
        y_c = x[2]; y_c_dot = x[3]
        theta = x[4]; theta_dot = x[5]
        phi = x[6]; phi_dot = x[7]

        Fx = u[0]; Fy = u[1]

        sin_theta = ca.sin(theta)
        cos_theta = ca.cos(theta)
        sin_phi = ca.sin(phi)
        cos_phi = ca.cos(phi)

        # 4x4 mass matrix
        M_total = M + m
        mL = m * L
        mL2 = m * L**2
        
        MM = ca.SX.zeros(4, 4)
        MM[0, 0] = M_total
        MM[0, 1] = 0
        MM[0, 2] = mL * cos_theta * cos_phi
        MM[0, 3] = -mL * sin_theta * sin_phi

        MM[1, 0] = 0
        MM[1, 1] = M_total
        MM[1, 2] = mL * cos_theta * sin_phi
        MM[1, 3] = mL * sin_theta * cos_phi

        MM[2, 0] = mL *cos_theta * cos_phi
        MM[2, 1] = mL*cos_theta * sin_phi
        MM[2, 2] = mL2
        MM[2, 3] = 0

        MM[3, 0] = -mL*sin_theta * sin_phi
        MM[3, 1] = mL*sin_theta * cos_phi
        MM[3, 2] = 0
        MM[3, 3] = mL2 * sin_theta**2

        # Right-hand side
        RHS = ca.SX.zeros(4, 1)
        RHS[0] = Fx + mL * (theta_dot**2 * sin_theta * cos_phi + phi_dot**2 * sin_theta * cos_phi - 2 * theta_dot * phi_dot * cos_theta * sin_phi)
        RHS[1] = Fy + mL * (theta_dot**2 * sin_theta * sin_phi + phi_dot**2 * sin_theta * sin_phi + 2 * theta_dot * phi_dot * cos_theta * cos_phi)
        RHS[2] = -m * g * L * sin_theta - b_theta * theta_dot
        RHS[3] = -m * g * L * sin_theta * sin_phi - b_phi * phi_dot

        # Solve for accelerations
        reg = 1e-8
        MM_reg = MM + ca.DM.eye(4) * reg
        accel = ca.solve(MM_reg, RHS)

        x_c_ddot = accel[0]
        y_c_ddot = accel[1]
        theta_ddot = accel[2]
        phi_ddot = accel[3]

        # State derivative
        x_dot = ca.vertcat(
            x_c_dot,
            x_c_ddot,
            y_c_dot,
            y_c_ddot,
            theta_dot,
            theta_ddot,
            phi_dot,
            phi_ddot
        )
        return x_dot

    def _rk4_step(self, x, u):
        """RK4 integration step"""
        f = self._continuous_dynamics
        dt = self.dt
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def _build_symbolic_model(self):
        """Build symbolic model for CasADi"""
        self.X = ca.SX.sym('X', self.nx, self.N + 1)
        self.U = ca.SX.sym('U', self.nu, self.N)
        self.X0 = ca.SX.sym('X0', self.nx)
        self.Xref = ca.SX.sym('Xref', self.nx)

    def _build_nlp(self):
        """Build nonlinear programming problem"""
        X = self.X; U = self.U; X0 = self.X0; Xref = self.Xref
        N = self.N

        cost = 0
        constraints = []

        # Initial condition constraint
        constraints.append(X[:, 0] - X0)

        # Build dynamics constraints and cost
        for k in range(N):
            dx = X[:, k] - Xref
            theta_error = ca.fabs(X[4, k] - np.pi)
        
            # Adaptive position weight based on pendulum stability
            position_weight = ca.if_else(
                theta_error > 0.05,
                0.01,
                1.0
            )
            
            # State deviation cost
            cost += ca.mtimes([dx.T, self.Q, dx])
            
            # Control cost
            cost += ca.mtimes([U[:, k].T, self.R, U[:, k]])
            
            # Additional theta penalty
            cost += 20000.0 * (theta_error**2)
            
            # Angular velocity penalty
            cost += 100.0 * (X[5, k]**2 + X[7, k]**2)

            # Dynamics constraint via RK4
            xk = X[:, k]
            uk = U[:, k]
            x_next = self._rk4_step(xk, uk)
            constraints.append(X[:, k + 1] - x_next)

        # Terminal cost
        dx_term = X[:, N] - Xref
        cost += 50.0 * ca.mtimes([dx_term.T, self.P, dx_term])
        cost += 20.0 * (X[5, N]**2 + X[7, N]**2)

        # Flatten decision variables
        opt_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1)
        )

        g = ca.vertcat(*constraints)
        p = ca.vertcat(X0, Xref)

        nlp = {'x': opt_vars, 'f': cost, 'g': g, 'p': p}

        # Create solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, self.ipopt_opts)

        # Build bounds
        self._build_bounds()

        # Helper dimensions
        self.n_vars = int(opt_vars.size1())
        self.n_constraints = int(g.size1())

    def _build_bounds(self):
        """Build variable and constraint bounds"""
        # State bounds
        x_min = [-5.0, -10.0, -5.0, -10.0, -2*np.pi, -20.0, -2*np.pi, -20.0]
        x_max = [5.0, 10.0, 5.0, 10.0, 2*np.pi, 20.0, 2*np.pi, 20.0]

        lbx = []
        ubx = []
        for k in range(self.N + 1):
            lbx.extend(x_min)
            ubx.extend(x_max)
        for k in range(self.N):
            lbx.extend([self.u_min, self.u_min])
            ubx.extend([self.u_max, self.u_max])

        self.lbx = ca.DM(lbx)
        self.ubx = ca.DM(ubx)

        # Constraint bounds (equality constraints)
        self.lbg = ca.DM.zeros(self.nx * (self.N + 1))
        self.ubg = ca.DM.zeros(self.nx * (self.N + 1))

    def solve(self, x0, x_ref, warm_start=True):
        """
        Solve MPC given current state and reference.
        Returns: (u0, success_flag, info dict)
        """
        try:
            # Format inputs
            x0 = np.array(x0, dtype=float).flatten()
            x_ref = np.array(x_ref, dtype=float).flatten()
            
            if len(x0) != self.nx or len(x_ref) != self.nx:
                raise ValueError(f"State dimensions mismatch: got {len(x0)}, {len(x_ref)}, expected {self.nx}")

            # Parameter vector
            p = np.concatenate([x0, x_ref])

            # Build cold initial guess
            def make_cold_guess():
                x_guess = np.zeros((self.nx, self.N + 1))
                for k in range(self.N + 1):
                    alpha = k / self.N if self.N > 0 else 0
                    x_guess[:, k] = (1 - alpha) * x0 + alpha * x_ref
                u_guess = np.zeros((self.nu, self.N))
                return np.concatenate([x_guess.flatten(order='F'), u_guess.flatten(order='F')])

            # Try warm-started guess if available
            tried_guesses = []
            if warm_start and self.prev_decision is not None:
                try:
                    x_prev = self.prev_decision[:self.nx * (self.N + 1)].reshape((self.nx, self.N + 1), order='F')
                    u_prev = self.prev_decision[self.nx * (self.N + 1):].reshape((self.nu, self.N), order='F')
                    x_guess = np.zeros((self.nx, self.N + 1))
                    u_guess = np.zeros((self.nu, self.N))
                    x_guess[:, 0] = x0
                    if self.N > 1:
                        x_guess[:, 1:] = x_prev[:, 1:]
                    u_guess = np.roll(u_prev, -1, axis=1)
                    x0_guess = np.concatenate([x_guess.flatten(order='F'), u_guess.flatten(order='F')])
                    tried_guesses.append(x0_guess)
                except Exception:
                    pass

            # Always prepare cold guess
            cold_guess = make_cold_guess()
            tried_guesses.append(cold_guess)

            # Try noisy guess to escape local issues
            noise = 1e-3 * np.random.randn(*cold_guess.shape)
            tried_guesses.append(np.clip(cold_guess + noise, -1e6, 1e6))

            solve_info = None
            sol = None
            
            # Try each guess until we get a usable solution
            for guess in tried_guesses:
                args = {
                    'x0': guess,
                    'lbx': self.lbx,
                    'ubx': self.ubx,
                    'lbg': self.lbg,
                    'ubg': self.ubg,
                    'p': p
                }
                try:
                    t0 = time()
                    # Suppress solver output
                    with open(os.devnull, 'w') as devnull:
                        old_stdout = sys.stdout
                        sys.stdout = devnull
                        try:
                            sol = self.solver(**args)
                        finally:
                            sys.stdout = old_stdout
                    solve_time = time() - t0
                except Exception as e:
                    solve_info = {'error': f'solver call exception: {e}'}
                    sol = None

                if sol is None:
                    continue

                solver_stats = self.solver.stats()
                success = bool(solver_stats.get('success', False))
                return_status = solver_stats.get('return_status', 'unknown')

                acceptable_failures = [
                    'Maximum_Iterations_Exceeded',
                    'Acceptable_Level',
                    'User_Requested_Stop',
                    'Maximum_CpuTime_Exceeded'
                ]

                if success or return_status in acceptable_failures:
                    w_opt = np.array(sol['x']).flatten()
                    if not np.isfinite(w_opt).all():
                        continue

                    # Cache for warm start
                    self.prev_decision = w_opt.copy()

                    # Extract control
                    nx_tot = self.nx * (self.N + 1)
                    U_flat = w_opt[nx_tot:nx_tot + self.nu * self.N]
                    U0 = U_flat[:self.nu]
                    U0 = np.clip(U0, self.u_min, self.u_max)

                    info = {
                        'status': 'ok' if success else 'acceptable',
                        'solve_time': solve_time,
                        'return_status': return_status
                    }
                    return np.array(U0).astype(float), True, info

                solve_info = {
                    'status': 'failed',
                    'return_status': return_status,
                    'solver_stats': solver_stats
                }

            # Fallback to previous control if available
            if self.prev_decision is not None:
                try:
                    nx_tot = self.nx * (self.N + 1)
                    U_flat = self.prev_decision[nx_tot:nx_tot + self.nu * self.N]
                    U0 = U_flat[:self.nu]
                    U0 = np.clip(U0, self.u_min, self.u_max)
                    return np.array(U0).astype(float), False, {'error': 'All solver attempts failed - using previous control fallback', 'solve_info': solve_info}
                except Exception:
                    pass

            # Final failure
            return np.zeros(self.nu), False, {'error': 'All solver attempts failed', 'solve_info': solve_info}

        except Exception as e:
            return np.zeros(self.nu), False, {'error': str(e)}

    def get_last_trajectory(self):
        """Get predicted trajectory from last solution"""
        if self.prev_decision is None:
            return None
        w = self.prev_decision
        X_flat = w[:self.nx * (self.N + 1)]
        X_mat = X_flat.reshape((self.nx, self.N + 1), order='F')
        return X_mat
