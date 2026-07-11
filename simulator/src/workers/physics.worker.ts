// Physics Web Worker — mass-aware Runge-Kutta 4 integrator for the Planar Three-Body Problem.
// Gravitational equations with arbitrary masses m1, m2, m3 (G=1, normalized units).

interface PhysicsInput {
  x2: number; z2: number;
  x3: number; z3: number;   // pre-computed from COM constraint on main thread
  tMax: number; dt: number;
  m1: number; m2: number; m3: number;
}

// Derivative of state y w.r.t. time.
// State layout: [x1, z1, x2, z2, x3, z3, vx1, vz1, vx2, vz2, vx3, vz3]
// Gravity: a_i = G * Σ_j≠i  m_j * (r_j - r_i) / |r_j - r_i|^3   (G = 1)
function computeDerivatives(y: Float32Array, m1: number, m2: number, m3: number): Float32Array {
  const dydt = new Float32Array(12);

  // Position derivatives = velocities
  dydt[0] = y[6];  dydt[1] = y[7];
  dydt[2] = y[8];  dydt[3] = y[9];
  dydt[4] = y[10]; dydt[5] = y[11];

  const x1 = y[0], z1 = y[1];
  const x2 = y[2], z2 = y[3];
  const x3 = y[4], z3 = y[5];

  // Direction vectors r_i - r_j  (note: r12 = r1 - r2)
  const dx12 = x1 - x2, dz12 = z1 - z2;
  const dx13 = x1 - x3, dz13 = z1 - z3;
  const dx23 = x2 - x3, dz23 = z2 - z3;

  // Softened distances to prevent singularity
  const eps = 1e-6;
  const d12 = Math.sqrt(dx12 * dx12 + dz12 * dz12) + eps;
  const d13 = Math.sqrt(dx13 * dx13 + dz13 * dz13) + eps;
  const d23 = Math.sqrt(dx23 * dx23 + dz23 * dz23) + eps;

  const ic12 = 1.0 / (d12 * d12 * d12);
  const ic13 = 1.0 / (d13 * d13 * d13);
  const ic23 = 1.0 / (d23 * d23 * d23);

  // a1 = -m2*(r12/d12^3) - m3*(r13/d13^3)
  dydt[6]  = -m2 * dx12 * ic12 - m3 * dx13 * ic13;
  dydt[7]  = -m2 * dz12 * ic12 - m3 * dz13 * ic13;
  // a2 = +m1*(r12/d12^3) - m3*(r23/d23^3)
  dydt[8]  =  m1 * dx12 * ic12 - m3 * dx23 * ic23;
  dydt[9]  =  m1 * dz12 * ic12 - m3 * dz23 * ic23;
  // a3 = +m1*(r13/d13^3) + m2*(r23/d23^3)
  dydt[10] =  m1 * dx13 * ic13 + m2 * dx23 * ic23;
  dydt[11] =  m1 * dz13 * ic13 + m2 * dz23 * ic23;

  return dydt;
}

function rk4Step(y: Float32Array, h: number, m1: number, m2: number, m3: number): Float32Array {
  const k1 = computeDerivatives(y, m1, m2, m3);

  const t1 = new Float32Array(12);
  for (let i = 0; i < 12; i++) t1[i] = y[i] + (h / 2) * k1[i];
  const k2 = computeDerivatives(t1, m1, m2, m3);

  const t2 = new Float32Array(12);
  for (let i = 0; i < 12; i++) t2[i] = y[i] + (h / 2) * k2[i];
  const k3 = computeDerivatives(t2, m1, m2, m3);

  const t3 = new Float32Array(12);
  for (let i = 0; i < 12; i++) t3[i] = y[i] + h * k3[i];
  const k4 = computeDerivatives(t3, m1, m2, m3);

  const yNext = new Float32Array(12);
  for (let i = 0; i < 12; i++) {
    yNext[i] = y[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  }
  return yNext;
}

self.onmessage = (event: MessageEvent<PhysicsInput>) => {
  const startTime = performance.now();
  const { x2, z2, x3, z3, tMax, dt, m1, m2, m3 } = event.data;

  // Initial state: positions + zero velocities
  let currentState = new Float32Array(12);
  currentState[0] = 1.0; currentState[1] = 0.0;  // Body 1
  currentState[2] = x2;  currentState[3] = z2;   // Body 2
  currentState[4] = x3;  currentState[5] = z3;   // Body 3 (COM-constrained)

  const numSteps = Math.round(tMax / dt) + 1;
  const trajectory = new Float32Array(numSteps * 12);
  trajectory.set(currentState, 0);

  // 20 sub-steps per output step for high-precision integration
  const subSteps = 20;
  const h = dt / subSteps;

  for (let step = 1; step < numSteps; step++) {
    let s: any = currentState;
    for (let sub = 0; sub < subSteps; sub++) {
      s = rk4Step(s, h, m1, m2, m3);
    }
    currentState = s;
    trajectory.set(currentState, step * 12);
  }

  const duration = performance.now() - startTime;
  (self as any).postMessage({ trajectory, duration, numSteps }, [trajectory.buffer]);
};
