# Regression Analysis

## **Option 1 — Supervised Learning Setup**

### **Inputs**

- \( \Theta_1(t) \)
- \( \Theta_2(t) \)
- \( \dot{\Theta}\_1(t) \)
- \( \dot{\Theta}\_2(t) \)

### **Outputs**

- \( \Theta_1(t + \Delta t) \)
- \( \Theta_2(t + \Delta t) \)
- \( \dot{\Theta}\_1(t + \Delta t) \)
- \( \dot{\Theta}\_2(t + \Delta t) \)

---

## **Model Type**

### **Polynomial Regression**

- Suitable for **very short‑term prediction**
- Captures local nonlinear behavior
- Unstable for long‑term forecasting due to polynomial growth

### **Steps**

- 1, a linear regression
- 2, quadratic
- 3+, n-degree polynomical

Mutiple Regression, in matrix-form:

$$
X =
\begin{pmatrix}
1 & x_{11} & x_{12} & x_{13} & \cdots & x_{1p}\\
1 & x_{21} & x_{22} & x_{23} & \cdots & x_{2p}\\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots\\
1 & x_{n1} & x_{n2} & x_{n3} & \cdots & x_{np}
\end{pmatrix}
$$

$$
\beta =
\begin{pmatrix}
\beta_0 \\ \beta_1 \\ \vdots \\ \beta_p
\end{pmatrix}
$$

$$
y =
\begin{pmatrix}
y_0 \\ y_1 \\ \vdots \\ y_n
\end{pmatrix}, y =X\beta + \epsilon
$$

$$
\min_\beta ||y-X\beta||^2 = \min_\beta[(y-X\beta)^{\text{T}}(y-X\beta)] \\
\beta =
(X^{\text{T}}X)^{-1}X^{\text{T}}y
$$
