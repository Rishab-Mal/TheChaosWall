export interface PendulumState {
  theta1: number;
  theta2: number;
  omega1: number;
  omega2: number;
}

const G = 9.81;

function derivatives(
  s: PendulumState,
  l1: number,
  l2: number,
  m1: number,
  m2: number
): PendulumState {
  const { theta1, theta2, omega1, omega2 } = s;
  const delta = theta1 - theta2;
  const cosD = Math.cos(delta);
  const sinD = Math.sin(delta);
  const denom = 2 * m1 + m2 - m2 * Math.cos(2 * delta);

  const alpha1 =
    (-G * (2 * m1 + m2) * Math.sin(theta1)
      - m2 * G * Math.sin(theta1 - 2 * theta2)
      - 2 * sinD * m2 * (omega2 * omega2 * l2 + omega1 * omega1 * l1 * cosD))
    / (l1 * denom);

  const alpha2 =
    (2 * sinD
      * (omega1 * omega1 * l1 * (m1 + m2)
        + G * (m1 + m2) * Math.cos(theta1)
        + omega2 * omega2 * l2 * m2 * cosD))
    / (l2 * denom);

  return { theta1: omega1, theta2: omega2, omega1: alpha1, omega2: alpha2 };
}

function add(a: PendulumState, b: PendulumState): PendulumState {
  return {
    theta1: a.theta1 + b.theta1,
    theta2: a.theta2 + b.theta2,
    omega1: a.omega1 + b.omega1,
    omega2: a.omega2 + b.omega2,
  };
}

function scale(s: PendulumState, k: number): PendulumState {
  return {
    theta1: s.theta1 * k,
    theta2: s.theta2 * k,
    omega1: s.omega1 * k,
    omega2: s.omega2 * k,
  };
}

export function rk4Step(
  s: PendulumState,
  dt: number,
  l1 = 1.0,
  l2 = 1.0,
  m1 = 1.0,
  m2 = 1.0
): PendulumState {
  const k1 = derivatives(s, l1, l2, m1, m2);
  const k2 = derivatives(add(s, scale(k1, dt / 2)), l1, l2, m1, m2);
  const k3 = derivatives(add(s, scale(k2, dt / 2)), l1, l2, m1, m2);
  const k4 = derivatives(add(s, scale(k3, dt)), l1, l2, m1, m2);
  return add(s, scale(add(add(k1, scale(k2, 2)), add(scale(k3, 2), k4)), dt / 6));
}

export function getPositions(
  state: PendulumState,
  ox: number,
  oy: number,
  armScale: number
) {
  const x1 = ox + armScale * Math.sin(state.theta1);
  const y1 = oy + armScale * Math.cos(state.theta1);
  const x2 = x1 + armScale * Math.sin(state.theta2);
  const y2 = y1 + armScale * Math.cos(state.theta2);
  return { x1, y1, x2, y2 };
}
