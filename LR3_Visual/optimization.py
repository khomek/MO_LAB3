import numpy as np

class Optimization:
    def __init__(self):
        self.history = []

    def func(self, x):
        return 4 * x[0]**2 + 0.5 * x[0] * x[1] + 2 * x[1]**2

    def grad(self, x):
        return np.array([8 * x[0] + 0.5 * x[1], 0.5 * x[0] + 4 * x[1]])

    def norma(self, vector):
        return np.sqrt(vector[0]**2 + vector[1]**2)

    def norma1(self, a, b):
        return np.sqrt(a**2 + b**2)

    def find_t(self, x, t, new_grad):
        x1 = np.array([x[0] - t * new_grad[0], x[1] - t * new_grad[1]])
        return self.func(x1)

    def bisection_method(self, a, b, e, x, new_grad, max_iterations=100):  # Added max_iterations
        if not (a < b and e > 0):
            print("Invalid input to bisection_method")
            return None # or raise an Exception

        k = 0
        xk = (a + b) / 2
        length = abs(b - a)

        while length > e and k < max_iterations:
            y = a + length / 4
            z = b - length / 4
            fy = self.find_t(x, y, new_grad)
            fz = self.find_t(x, z, new_grad)
            fxk = self.find_t(x, xk, new_grad)

            if fy < fxk:
                b = xk
                xk = y
            else:
                if fz < fxk:
                    a = xk
                    xk = z
                else:
                    a = y
                    b = z

            length = abs(b - a)
            k += 1

        if k == max_iterations:
            print("Bisection method did not converge")
            return None # or raise an Exception
        else:
            return xk

    def is_positive_definite(self, matrix):
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    def newton_method(self, x0, eps1, eps2, M):
        x = x0.copy()
        k = 0
        prev_x = None
        prev_f = None
        convergence_flag = False
        self.history = [x.copy()]

        hessian = np.array([[8, 0.5], [0.5, 4]])

        while k < M:
            try:
                current_grad = self.grad(x)
                grad_norm = self.norma(current_grad)

                if grad_norm <= eps1:
                    return x, k + 1

                try:
                    hessian_inv = np.linalg.inv(hessian)
                    is_pd = self.is_positive_definite(hessian_inv)
                except np.linalg.LinAlgError:
                    is_pd = False

                if is_pd:
                    d_k = -hessian_inv @ current_grad
                    t_k = 1.0
                else:
                    d_k = -current_grad
                    t_k = self.bisection_method(0.01, 1.0, 0.001, x, current_grad)  # Pass current_grad

                    if t_k is None: #Bisection method failed
                        return x, k + 1, "Bisection method failed to converge."

                prev_x = x.copy()
                prev_f = self.func(prev_x)

                x = x + t_k * d_k
                self.history.append(x.copy())

                x_diff = self.norma(x - prev_x)
                f_diff = abs(self.func(x) - prev_f)

                if x_diff < eps2 and f_diff < eps2:
                    if convergence_flag:
                        return x, k + 1
                    convergence_flag = True
                else:
                    convergence_flag = False

                k += 1

            except Exception as e:
                print(f"Error during optimization: {str(e)}")
                return x, k + 1, f"Error: {str(e)}"

        return x, k + 1