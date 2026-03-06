### Hamiltonian Neural Network Reference
> https://doi.org/10.48550/arXiv.1906.01563

Phase space $z = (q, p) ∈ R^{2n}$ (canonical coordinates).
Hamiltonian $H : R^{2n} → R$ (energy).

Canonical symplectic matrix
$J = [ 0 I_n
-I_n 0 ]$.

Hamilton's equations $ż = X_H(z) = J ∇H(z)$.
Component form: $q̇ = ∂H/∂p, ṗ = −∂H/∂q$.

#### HNN principles:
- Parameterize $H_θ(z)$ with a neural network
- $L_{\text{HNN}} = ||\frac{\partial H_{\theta}}{\partial p}-\frac{\partial q}{\partial t}||_2 + ||\frac{\partial H_{\theta}}{\partial q}+\frac{\partial p}{\partial t}||_2$
- autodifferentiation used to compute $(\partial q/\partial t,\partial p/\partial t)$ from predicted $H_{\theta}$

#### Paper implementation details:
- three layer MLP; 200 hidden units; tanh activations; trained for 2000 gradient steps
- $10^{-3}$ learning rate, Adam optimizer used
- MSE compares model to ground truth
- results of HNN $(\partial q/\partial t,\partial p/\partial t)$ used in integration with 4th order Runge-Kutta algorithm in SciPy