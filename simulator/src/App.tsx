import { useState, useEffect, useRef, useMemo } from 'react';
import { OrbitCanvas } from './components/OrbitCanvas';
import {
  Play, Pause, RotateCcw, Activity, Cpu, Sliders,
  Info, Sparkles, TrendingUp, X, Gauge,
} from 'lucide-react';

// PINN model hard constraint: trained only for t ∈ [0, 10]
const PINN_T_MAX = 10.0;
const DT = 0.0390625; // 256 intervals per 10s = 257 steps

export default function App() {
  // ── Body 2 initial coordinates (Body 1 fixed, Body 3 dependent) ──
  // Default values set to a widely separated, highly stable non-colliding orbit
  const [x2, setX2] = useState<number>(-0.8);
  const [z2, setZ2] = useState<number>(0.8);

  // ── Masses — only physics uses these; PINN always uses m=1 ──
  const [m1, setM1] = useState<number>(1.0);
  const [m2, setM2] = useState<number>(1.0);
  const [m3, setM3] = useState<number>(1.0);

  // ── Extended simulation time for physics (PINN capped at 10s internally) ──
  const [tMax, setTMax] = useState<number>(30);

  // ── Trajectory buffers ──
  const [physicsTrajectory, setPhysicsTrajectory] = useState<Float32Array | null>(null);
  const [aiTrajectory, setAiTrajectory] = useState<Float32Array | null>(null);

  // ── Telemetry ──
  const [physicsLatency, setPhysicsLatency] = useState<number>(0);
  const [aiLatency, setAiLatency] = useState<number>(0);
  const [aiError, setAiError] = useState<string | null>(null);

  // ── Playback ──
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [timeIndex, setTimeIndex] = useState<number>(0);
  const [speed, setSpeed] = useState<number>(1);

  // ── Visibility toggles ──
  const [showPhysics, setShowPhysics] = useState<boolean>(true);
  const [showAI, setShowAI] = useState<boolean>(true);
  const [showExplanation, setShowExplanation] = useState<boolean>(false);

  // ── Worker refs ──
  const physicsWorkerRef = useRef<Worker | null>(null);
  const aiWorkerRef = useRef<Worker | null>(null);

  // ── Derived constants ──
  const totalSteps = useMemo(() => Math.round(tMax / DT) + 1, [tMax]);
  const aiTotalSteps = useMemo(() => Math.round(PINN_T_MAX / DT) + 1, []);

  // ── Mass-aware Body 3 initial position from center-of-mass constraint ──
  // m1·r1 + m2·r2 + m3·r3 = 0  ⟹  r3 = -(m1·r1 + m2·r2) / m3
  const x3Init = useMemo(() => -(m1 * 1.0 + m2 * x2) / m3, [m1, m2, m3, x2]);
  const z3Init = useMemo(() => -(m2 * z2) / m3, [m2, m3, z2]);

  // ── Worker initialisation ──
  useEffect(() => {
    const physicsWorker = new Worker(
      new URL('./workers/physics.worker.ts', import.meta.url), { type: 'module' }
    );
    const aiWorker = new Worker(
      new URL('./workers/ai.worker.ts', import.meta.url), { type: 'module' }
    );
    physicsWorkerRef.current = physicsWorker;
    aiWorkerRef.current = aiWorker;

    // Send initialization parameters (absolute paths derived from Vite's base path) to AI worker
    const base = import.meta.env.BASE_URL;
    aiWorker.postMessage({
      type: 'init',
      modelUrl: new URL(`${base}pinn_model_2d_v2.onnx`, window.location.origin).href,
      modelDataUrl: new URL(`${base}pinn_model_2d_v2.onnx.data`, window.location.origin).href,
    });

    physicsWorker.onmessage = (e: MessageEvent) => {
      setPhysicsTrajectory(e.data.trajectory);
      setPhysicsLatency(e.data.duration);
      setTimeIndex(0);
    };
    aiWorker.onmessage = (e: MessageEvent) => {
      if (e.data.success) {
        setAiTrajectory(e.data.trajectory);
        setAiLatency(e.data.duration);
        setAiError(null);
      } else {
        setAiError(e.data.error);
        setAiLatency(e.data.duration);
      }
    };

    triggerSimulations(x2, z2, m1, m2, m3, tMax);

    return () => { physicsWorker.terminate(); aiWorker.terminate(); };
  }, []);

  // ── Re-run when tMax changes ──
  useEffect(() => {
    setTimeIndex(0);
    triggerSimulations(x2, z2, m1, m2, m3, tMax);
  }, [tMax]);

  // ── Dispatch to both workers ──
  const triggerSimulations = (
    cx2: number, cz2: number,
    cm1: number, cm2: number, cm3: number,
    ctMax: number
  ) => {
    const cx3 = -(cm1 * 1.0 + cm2 * cx2) / cm3;
    const cz3 = -(cm2 * cz2) / cm3;

    physicsWorkerRef.current?.postMessage({
      x2: cx2, z2: cz2, x3: cx3, z3: cz3,
      tMax: ctMax, dt: DT,
      m1: cm1, m2: cm2, m3: cm3,
    });
    aiWorkerRef.current?.postMessage({
      x2: cx2, z2: cz2,
      tMax: PINN_T_MAX, // AI ignores this and always uses 10.0, but send for clarity
      dt: DT,
    });
  };

  const handlePositionChange = (newX: number, newZ: number) => {
    const clamp = (v: number) => Math.min(Math.max(v, -1.5), 1.5);
    const cx = clamp(newX);
    const cz = clamp(newZ);

    // Compute proposed position of Body 3 based on current masses
    const cx3 = -(m1 * 1.0 + m2 * cx) / m3;
    const cz3 = -(m2 * cz) / m3;

    // Verify distance safety (prevent initial overlap or immediate singularity collision)
    const d12Sq = (cx - 1.0) ** 2 + cz ** 2;
    const d23Sq = (cx - cx3) ** 2 + (cz - cz3) ** 2;
    const d13Sq = (cx3 - 1.0) ** 2 + cz3 ** 2;

    const MIN_DIST = 0.15; // Safe minimum distance to avoid singularity
    const MIN_DIST_SQ = MIN_DIST * MIN_DIST;

    if (d12Sq < MIN_DIST_SQ || d23Sq < MIN_DIST_SQ || d13Sq < MIN_DIST_SQ) {
      // Too close! Ignore change to prevent mathematical explosion
      return;
    }

    setX2(cx); setZ2(cz);
    triggerSimulations(cx, cz, m1, m2, m3, tMax);
  };

  // ── Mass change handler ──
  const handleMassChange = (nm1: number, nm2: number, nm3: number) => {
    // Compute proposed position of Body 3 based on new masses
    const cx3 = -(nm1 * 1.0 + nm2 * x2) / nm3;
    const cz3 = -(nm2 * z2) / nm3;

    // Verify distance safety
    const d12Sq = (x2 - 1.0) ** 2 + z2 ** 2;
    const d23Sq = (x2 - cx3) ** 2 + (z2 - cz3) ** 2;
    const d13Sq = (cx3 - 1.0) ** 2 + cz3 ** 2;

    const MIN_DIST = 0.15;
    const MIN_DIST_SQ = MIN_DIST * MIN_DIST;

    if (d12Sq < MIN_DIST_SQ || d23Sq < MIN_DIST_SQ || d13Sq < MIN_DIST_SQ) {
      // Too close! Ignore mass change to prevent mathematical explosion
      return;
    }

    setM1(nm1); setM2(nm2); setM3(nm3);
    triggerSimulations(x2, z2, nm1, nm2, nm3, tMax);
  };

  // ── Animation loop ──
  useEffect(() => {
    if (!isPlaying) return;
    let last = performance.now();
    let acc = 0;
    const ref = { id: 0 };

    const tick = (now: number) => {
      acc += ((now - last) / 1000) * speed;
      last = now;
      if (acc >= DT) {
        const steps = Math.floor(acc / DT);
        acc %= DT;
        setTimeIndex(prev => {
          const next = prev + steps;
          return next >= totalSteps ? 0 : next;
        });
      }
      ref.id = requestAnimationFrame(tick);
    };
    ref.id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(ref.id);
  }, [isPlaying, speed, totalSteps]);

  // ── Divergence metrics (compared only within PINN range) ──
  const metrics = useMemo(() => {
    if (!physicsTrajectory || !aiTrajectory) {
      return { mse: 0, divergence: 0, pct: 0 };
    }
    const compareSteps = Math.min(aiTotalSteps, totalSteps, aiTrajectory.length / 12);
    let totalSq = 0;
    for (let s = 0; s < compareSteps; s++) {
      const idx = s * 12;
      for (let c = 0; c < 6; c++) {
        const d = physicsTrajectory[idx + c] - aiTrajectory[idx + c];
        totalSq += d * d;
      }
    }
    const mse = totalSq / (compareSteps * 6);

    const cIdx = Math.min(timeIndex, (aiTrajectory.length / 12) - 1) * 12;
    let instSq = 0;
    for (let c = 0; c < 6; c++) {
      const d = physicsTrajectory[cIdx + c] - aiTrajectory[cIdx + c];
      instSq += d * d;
    }
    const divergence = Math.sqrt(instSq / 6);
    const pct = Math.min((divergence / 0.5) * 100, 100);
    return { mse, divergence, pct };
  }, [physicsTrajectory, aiTrajectory, timeIndex, aiTotalSteps, totalSteps]);

  const formattedTime = (timeIndex * DT).toFixed(2);
  const isPinnRange = timeIndex < aiTotalSteps;

  return (
    <div className="w-screen h-screen relative overflow-hidden bg-[#07080d]">

      {/* ══ 3D Canvas ══ */}
      <div className="absolute inset-0 z-0">
        <OrbitCanvas
          physicsTrajectory={physicsTrajectory}
          aiTrajectory={aiTrajectory}
          timeIndex={timeIndex}
          aiTotalSteps={aiTotalSteps}
          x2Init={x2} z2Init={z2}
          x3Init={x3Init} z3Init={z3Init}
          onDrag={handlePositionChange}
          showPhysics={showPhysics}
          showAI={showAI}
        />
      </div>

      {/* ══ Top bar ══ */}
      <header className="absolute top-0 inset-x-0 p-4 z-10 flex flex-col md:flex-row justify-between items-center bg-gradient-to-b from-[#07080d]/85 to-transparent pointer-events-none">
        <div className="pointer-events-auto">
          <div className="flex items-center space-x-2">
            <Activity className="h-6 w-6 text-cyan-400 animate-pulse" />
            <h1 className="font-orbitron font-extrabold tracking-wider text-lg md:text-xl bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-white to-pink-500">
              PINN THREE-BODY CHAOS VECTOR
            </h1>
          </div>
          <p className="text-[11px] text-gray-400 mt-0.5 uppercase tracking-widest font-mono">
            ONNX WASM (t≤10s) &nbsp;·&nbsp; RK4 Physics (t≤{tMax}s) &nbsp;·&nbsp; m1={m1.toFixed(1)} m2={m2.toFixed(1)} m3={m3.toFixed(1)}
          </p>
        </div>
        <div className="mt-3 md:mt-0 pointer-events-auto">
          <button onClick={() => setShowExplanation(true)}
            className="flex items-center space-x-2 px-3 py-1.5 rounded-lg border border-gray-700 bg-gray-900/60 hover:bg-gray-800 text-xs font-semibold uppercase text-gray-300 transition">
            <Info className="h-4 w-4 text-cyan-400" /><span>Architecture</span>
          </button>
        </div>
      </header>

      {/* ══ LEFT SIDEBAR — Initial State & Mass Controls ══ */}
      <section className="absolute top-24 left-4 w-72 max-h-[calc(100vh-13rem)] overflow-y-auto z-10 glass-panel rounded-xl p-4 flex flex-col space-y-4">

        {/* Header */}
        <div className="flex items-center space-x-2">
          <Sliders className="h-4 w-4 text-cyan-400" />
          <h2 className="font-orbitron font-semibold text-sm tracking-wider uppercase">Initial State</h2>
        </div>

        <p className="text-[11px] text-gray-400 leading-relaxed -mt-2">
          Drag the <span className="text-amber-400 font-semibold">amber ring</span> (Body 2) on the canvas, or use the sliders below.
        </p>

        {/* Body 1 */}
        <div className="p-2.5 rounded-lg bg-cyan-950/20 border border-cyan-800/25 font-mono text-xs">
          <div className="flex justify-between text-cyan-400 font-semibold mb-1">
            <span>BODY 1 <span className="text-cyan-300/50">(fixed)</span></span>
            <span className="text-cyan-300/70">m₁ = {m1.toFixed(1)}</span>
          </div>
          <div className="flex justify-between text-gray-400">
            <span>X: <strong className="text-gray-200">1.000</strong></span>
            <span>Z: <strong className="text-gray-200">0.000</strong></span>
          </div>
        </div>

        {/* Body 2 — interactive sliders */}
        <div className="p-2.5 rounded-lg bg-amber-950/20 border border-amber-800/25 font-mono text-xs space-y-2">
          <div className="flex justify-between text-amber-400 font-semibold">
            <span>BODY 2 <span className="text-amber-300/50">(drag handle)</span></span>
            <span className="text-amber-300/70">m₂ = {m2.toFixed(1)}</span>
          </div>
          {[
            { label: 'X_init', val: x2, setter: (v: number) => handlePositionChange(v, z2) },
            { label: 'Z_init', val: z2, setter: (v: number) => handlePositionChange(x2, v) },
          ].map(({ label, val, setter }) => (
            <div key={label}>
              <div className="flex justify-between text-gray-400 mb-1">
                <span>{label}</span><span className="text-amber-300">{val.toFixed(3)}</span>
              </div>
              <input type="range" min="-1.5" max="1.5" step="0.005" value={val}
                onChange={e => setter(parseFloat(e.target.value))}
                className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-amber-400" />
            </div>
          ))}
        </div>

        {/* Body 3 — read-only, COM-derived */}
        <div className="p-2.5 rounded-lg bg-emerald-950/20 border border-emerald-800/25 font-mono text-xs">
          <div className="flex justify-between text-emerald-400 font-semibold mb-1">
            <span>BODY 3 <span className="text-emerald-300/50">(COM constraint)</span></span>
            <span className="text-emerald-300/70">m₃ = {m3.toFixed(1)}</span>
          </div>
          <div className="flex justify-between text-gray-400">
            <span>X: <strong className="text-gray-200">{x3Init.toFixed(3)}</strong></span>
            <span>Z: <strong className="text-gray-200">{z3Init.toFixed(3)}</strong></span>
          </div>
          <div className="text-[10px] text-gray-500 mt-1.5 font-sans italic">
            r₃ = −(m₁·r₁ + m₂·r₂) / m₃
          </div>
        </div>

        {/* ── Mass Section ── */}
        <div className="pt-3 border-t border-gray-800/60 space-y-3 font-mono text-xs">
          <div className="flex items-center justify-between">
            <span className="text-violet-400 font-semibold font-orbitron uppercase text-[11px] tracking-wide">
              Masses
            </span>
            <span className="text-[9px] text-violet-400/50 italic">physics only · PINN uses m=1</span>
          </div>
          {[
            { label: 'Body 1  m₁', val: m1, onChange: (v: number) => handleMassChange(v, m2, m3), accent: 'accent-cyan-500' },
            { label: 'Body 2  m₂', val: m2, onChange: (v: number) => handleMassChange(m1, v, m3), accent: 'accent-amber-400' },
            { label: 'Body 3  m₃', val: m3, onChange: (v: number) => handleMassChange(m1, m2, v), accent: 'accent-emerald-400' },
          ].map(({ label, val, onChange, accent }) => (
            <div key={label}>
              <div className="flex justify-between text-gray-400 mb-1">
                <span>{label}</span><span className="text-violet-300">{val.toFixed(1)}</span>
              </div>
              <input type="range" min="0.1" max="5.0" step="0.1" value={val}
                onChange={e => onChange(parseFloat(e.target.value))}
                className={`w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer ${accent}`} />
            </div>
          ))}
        </div>

        {/* ── Trail toggles ── */}
        <div className="pt-3 border-t border-gray-800/60 space-y-2 text-xs">
          {[
            { label: '🔵 Physics trail (RK4)', checked: showPhysics, set: setShowPhysics },
            { label: '🩷 PINN trail (t ≤ 10s)', checked: showAI, set: setShowAI },
          ].map(({ label, checked, set }) => (
            <label key={label} className="flex items-center justify-between text-gray-400 cursor-pointer">
              <span>{label}</span>
              <input type="checkbox" checked={checked} onChange={e => set(e.target.checked)}
                className="w-4 h-4 rounded bg-gray-900 border-gray-700" />
            </label>
          ))}
        </div>
      </section>

      {/* ══ RIGHT SIDEBAR wrapper ══ */}
      <div className="absolute top-24 right-4 w-72 z-10 flex flex-col space-y-4 pointer-events-none">
        {/* Telemetry Panel */}
        <section className="pointer-events-auto glass-panel rounded-xl p-4 flex flex-col space-y-4 max-h-[calc(100vh-21rem)] overflow-y-auto">
          <div className="flex items-center space-x-2">
            <Cpu className="h-4 w-4 text-pink-400" />
            <h2 className="font-orbitron font-semibold text-sm tracking-wider uppercase">Worker Telemetry</h2>
          </div>

          {/* Thread latency cards */}
          <div className="grid grid-cols-2 gap-3 font-mono text-xs">
            <div className="p-3 rounded-lg border border-cyan-800/30 bg-cyan-950/15">
              <div className="text-[10px] text-cyan-400 uppercase font-semibold">RK4 Physics</div>
              <div className="text-xl font-bold text-gray-200 mt-1">
                {physicsLatency.toFixed(0)}<span className="text-xs font-normal text-gray-400"> ms</span>
              </div>
              <div className="text-[9px] text-gray-500 mt-0.5">{tMax}s / {totalSteps} steps</div>
            </div>
            <div className="p-3 rounded-lg border border-pink-800/30 bg-pink-950/15">
              <div className="text-[10px] text-pink-400 uppercase font-semibold">ONNX WASM</div>
              <div className="text-xl font-bold text-gray-200 mt-1">
                {aiLatency.toFixed(0)}<span className="text-xs font-normal text-gray-400"> ms</span>
              </div>
              <div className="text-[9px] text-gray-500 mt-0.5">10s / {aiTotalSteps} steps</div>
            </div>
          </div>

          {aiError && (
            <div className="p-2 bg-red-950/30 border border-red-900/40 rounded-lg text-[11px] text-red-400 font-mono break-words">
              <strong>WASM ERROR:</strong> {aiError}
            </div>
          )}

          {!isPinnRange && showAI && (
            <div className="p-2 bg-amber-950/20 border border-amber-800/30 rounded-lg text-[11px] text-amber-400 font-mono">
              ⚠ t={formattedTime}s &gt; PINN range. Physics trail only.
            </div>
          )}

          {/* Divergence */}
          <div className="pt-2 border-t border-gray-800 space-y-3">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-rose-400" />
              <h3 className="font-orbitron text-xs font-semibold uppercase">Chaos Divergence</h3>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between font-mono text-[10px] text-gray-400">
                <span>PINN STABILITY</span>
                <span className={isPinnRange
                  ? (metrics.divergence > 0.3 ? 'text-red-400 font-bold'
                    : metrics.divergence > 0.1 ? 'text-amber-400' : 'text-emerald-400')
                  : 'text-gray-600'}>
                  {isPinnRange
                    ? (metrics.divergence > 0.3 ? 'CHAOTIC' : metrics.divergence > 0.1 ? 'DRIFTING' : 'ALIGNED')
                    : 'BEYOND PINN'}
                </span>
              </div>
              <div className="w-full h-2.5 bg-gray-900 rounded-full overflow-hidden border border-gray-800">
                <div className={`h-full transition-all duration-150 ${metrics.divergence > 0.3 ? 'bg-gradient-to-r from-amber-500 to-rose-600' : metrics.divergence > 0.1 ? 'bg-amber-400' : 'bg-emerald-400'}`}
                  style={{ width: `${isPinnRange ? metrics.pct : 0}%` }} />
              </div>
            </div>

            <div className="space-y-1.5 font-mono text-xs">
              {[
                { label: 'Time', val: `${formattedTime}s / ${tMax}s` },
                { label: 'Instant Drift', val: isPinnRange ? `${metrics.divergence.toFixed(4)} RMSE` : '—', cls: 'text-rose-400' },
                { label: 'Path MSE (≤10s)', val: metrics.mse.toExponential(3), cls: 'text-pink-400' },
              ].map(({ label, val, cls }) => (
                <div key={label} className="flex justify-between py-0.5 border-b border-gray-800/30">
                  <span className="text-gray-400">{label}:</span>
                  <span className={cls ?? 'text-gray-200'}>{val}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Simulation Duration */}
          <div className="pt-2 border-t border-gray-800">
            <div className="flex items-center space-x-2 mb-2">
              <Gauge className="h-4 w-4 text-indigo-400" />
              <span className="font-orbitron text-[11px] uppercase font-semibold text-indigo-400">Simulation Duration</span>
            </div>
            <div className="font-mono text-xs space-y-1">
              <div className="flex justify-between text-gray-400">
                <span>Physics runs for</span>
                <span className="text-indigo-300 font-semibold">{tMax}s</span>
              </div>
              <input type="range" min="10" max="100" step="5" value={tMax}
                onChange={e => setTMax(parseInt(e.target.value))}
                className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-indigo-500" />
              <div className="flex justify-between text-[9px] text-gray-600">
                <span>10s</span><span>55s</span><span>100s</span>
              </div>
              <div className="text-[9px] text-gray-500 italic mt-1">
                PINN always capped at 10s (training range)
              </div>
            </div>
          </div>
        </section>

        {/* Standalone Developer Profile Card */}
        <section className="pointer-events-auto glass-panel rounded-xl p-4 flex flex-col space-y-2">
          <div className="text-[10px] uppercase font-semibold text-cyan-400 font-orbitron tracking-wider">
            Developer
          </div>
          <div className="flex flex-col space-y-2">
            <div className="text-sm font-bold text-gray-200">Pranjalya Tiwari</div>
            <div className="flex space-x-2 pt-1">
              <a href="https://github.com/Pranjalya" target="_blank" rel="noopener noreferrer"
                className="flex-1 flex items-center justify-center space-x-1.5 px-2.5 py-1.5 rounded-lg border border-gray-800 bg-gray-900/60 hover:bg-gray-800 text-[10px] font-semibold text-gray-300 hover:text-white transition">
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg>
                <span>GitHub</span>
              </a>
              <a href="https://linkedin.com/in/pranjalya-tiwari" target="_blank" rel="noopener noreferrer"
                className="flex-1 flex items-center justify-center space-x-1.5 px-2.5 py-1.5 rounded-lg border border-gray-800 bg-gray-900/60 hover:bg-gray-800 text-[10px] font-semibold text-gray-300 hover:text-white transition">
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"/><rect width="4" height="12" x="2" y="9" rx="1"/><circle cx="4" cy="4" r="2"/></svg>
                <span>LinkedIn</span>
              </a>
            </div>
          </div>
        </section>
      </div>

      {/* ══ Bottom Playback Bar ══ */}
      <footer className="absolute bottom-4 inset-x-0 z-10 flex justify-center px-4 pointer-events-none">
        <div className="pointer-events-auto glass-panel rounded-xl px-5 py-3 flex flex-col md:flex-row items-center space-y-3 md:space-y-0 md:space-x-5 w-full max-w-2xl glow-blue">

          {/* Play/Pause & Reset */}
          <div className="flex items-center space-x-3 flex-shrink-0">
            <button onClick={() => setIsPlaying(p => !p)}
              className="p-2.5 rounded-full bg-cyan-600 hover:bg-cyan-500 text-white transition">
              {isPlaying
                ? <Pause className="h-4 w-4 fill-current" />
                : <Play className="h-4 w-4 fill-current" />}
            </button>
            <button onClick={() => { setTimeIndex(0); setIsPlaying(false); }}
              className="p-2 rounded-lg border border-gray-700 bg-gray-900/60 hover:bg-gray-800 text-gray-300 transition">
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>

          {/* Timeline scrubber */}
          <div className="flex-1 w-full flex items-center space-x-2.5">
            <span className="font-mono text-[11px] text-cyan-400 w-12 text-right flex-shrink-0">{formattedTime}s</span>
            <div className="relative flex-1">
              <input type="range" min="0" max={totalSteps - 1} step="1" value={timeIndex}
                onChange={e => { setTimeIndex(parseInt(e.target.value)); setIsPlaying(false); }}
                className="w-full h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-cyan-400" />
              {/* PINN end marker */}
              {tMax > PINN_T_MAX && (
                <div
                  className="absolute top-1/2 -translate-y-1/2 pointer-events-none flex flex-col items-center"
                  style={{ left: `${(aiTotalSteps / totalSteps) * 100}%` }}>
                  <div className="w-0.5 h-3 bg-pink-500/70 rounded-full" />
                </div>
              )}
            </div>
            <span className="font-mono text-[11px] text-gray-400 w-12 flex-shrink-0">{tMax}s</span>
          </div>

          {/* Speed selector */}
          <div className="flex items-center space-x-1 border border-gray-800 bg-gray-950/40 rounded-lg p-1 flex-shrink-0">
            {([0.5, 1, 2, 4] as const).map(s => (
              <button key={s} onClick={() => setSpeed(s)}
                className={`px-2 py-1 text-[10px] font-mono rounded font-semibold transition ${speed === s ? 'bg-cyan-950 border border-cyan-800 text-cyan-400' : 'text-gray-400 hover:text-gray-200'}`}>
                {s}×
              </button>
            ))}
          </div>
        </div>
      </footer>

      {/* ══ Explanation Modal ══ */}
      {showExplanation && (
        <div className="absolute inset-0 bg-[#07080d]/80 backdrop-blur-sm z-30 flex items-center justify-center p-4">
          <div className="glass-panel w-full max-w-xl rounded-2xl p-6 relative max-h-[85vh] overflow-y-auto">
            <button onClick={() => setShowExplanation(false)}
              className="absolute top-4 right-4 p-1 rounded-lg border border-gray-700 bg-gray-900/60 hover:bg-gray-800 text-gray-400 transition">
              <X className="h-5 w-5" />
            </button>
            <div className="flex items-center space-x-2 mb-4 pb-3 border-b border-gray-800">
              <Sparkles className="h-5 w-5 text-cyan-400" />
              <h2 className="font-orbitron font-extrabold text-base text-gray-100 uppercase tracking-wide">Architecture</h2>
            </div>
            <div className="space-y-3 text-[12px] text-gray-300 leading-relaxed">
              <div className="p-3 bg-cyan-950/20 border border-cyan-800/20 rounded-xl">
                <h3 className="font-orbitron text-cyan-300 text-sm font-semibold mb-1">Planar Three-Body Problem</h3>
                <p>Three equal-mass bodies exerting mutual gravitational attraction in 2D. Chaotic: even nanometer changes in initial position yield totally different long-term orbits.</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-cyan-900/10 border border-cyan-800/25 rounded-xl">
                  <h4 className="font-orbitron text-cyan-400 text-xs font-semibold mb-1">RK4 Physics Worker</h4>
                  <p className="text-gray-400 text-[11px]">Sub-stepped (20×) 4th-order Runge-Kutta. Supports variable masses m₁, m₂, m₃ and any duration. Runs in an isolated background thread.</p>
                </div>
                <div className="p-3 bg-pink-900/10 border border-pink-800/25 rounded-xl">
                  <h4 className="font-orbitron text-pink-400 text-xs font-semibold mb-1">ONNX PINN Worker</h4>
                  <p className="text-gray-400 text-[11px]">Physics-Informed Neural Net via onnxruntime-web WASM. Trained for equal masses (m=1) and t ∈ [0, 10s]. Mass/duration changes don't affect the PINN trail.</p>
                </div>
              </div>
              <div className="p-3 border border-gray-800 rounded-xl bg-gray-950/25">
                <h4 className="font-orbitron text-gray-200 text-xs font-semibold mb-1">Center-of-Mass Constraint</h4>
                <p className="text-gray-400 text-[11px]">Body 1 is fixed at (1, 0). Body 3 is placed so the system's center of mass stays at the origin:<br />
                  <span className="text-gray-200 font-mono">r₃ = −(m₁·r₁ + m₂·r₂) / m₃</span><br/>
                  Changing any mass will automatically reposition Body 3.
                </p>
              </div>

              {/* Developer Profile Card */}
              <div className="p-3 border border-gray-800 rounded-xl bg-gray-950/45 flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-3 sm:space-y-0">
                <div>
                  <h4 className="font-orbitron text-cyan-400 text-[10px] font-semibold uppercase tracking-wider">Developer</h4>
                  <p className="text-gray-200 text-xs font-bold mt-0.5">Pranjalya Tiwari</p>
                  <p className="text-gray-400 text-[10px] italic">Physics Engine & AI Architect</p>
                </div>
                <div className="flex space-x-2">
                  <a href="https://github.com/Pranjalya" target="_blank" rel="noopener noreferrer"
                    className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg border border-gray-800 bg-gray-900/60 hover:bg-gray-800 text-[11px] font-semibold text-gray-300 hover:text-white transition">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg>
                    <span>GitHub</span>
                  </a>
                  <a href="https://linkedin.com/in/pranjalya-tiwari" target="_blank" rel="noopener noreferrer"
                    className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg border border-gray-800 bg-gray-900/60 hover:bg-gray-800 text-[11px] font-semibold text-gray-300 hover:text-white transition">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"/><rect width="4" height="12" x="2" y="9" rx="1"/><circle cx="4" cy="4" r="2"/></svg>
                    <span>LinkedIn</span>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
