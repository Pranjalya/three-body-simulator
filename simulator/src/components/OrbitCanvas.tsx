import { useState, useMemo } from 'react';
import { Canvas, type ThreeEvent } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';

// Extract [x, 0, z] world positions from the flat trajectory buffer for one body
function getTrajectoryPoints(
  trajectory: Float32Array | null,
  bodyIdx: number,
  clampToSteps?: number        // optional cap (for AI trails limited to 10s)
): [number, number, number][] {
  if (!trajectory || trajectory.length === 0) return [];
  const total = trajectory.length / 12;
  const limit = clampToSteps !== undefined ? Math.min(clampToSteps, total) : total;
  const points: [number, number, number][] = [];
  for (let i = 0; i < limit; i++) {
    const idx = i * 12 + bodyIdx * 2;
    points.push([trajectory[idx], 0, trajectory[idx + 1]]);
  }
  return points;
}

// ── Trail line rendered as a Three.js primitive to avoid JSX <line> SVG collision ──
interface TrailLineProps {
  points: [number, number, number][];
  color: string;
  opacity?: number;
}

const TrailLine = ({ points, color, opacity = 0.35 }: TrailLineProps) => {
  const line = useMemo(() => {
    if (points.length < 2) return null;
    const pts = points.map(([x, y, z]) => new THREE.Vector3(x, y, z));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
    return new THREE.Line(geo, mat);
  }, [points, color, opacity]);

  if (!line) return null;
  return <primitive object={line} />;
};

// ── Fading particle tail behind the moving body head ──
interface FadingTrailProps {
  points: [number, number, number][];
  currentIndex: number;
  color: string;
  maxTailLength?: number;
}

const FadingTrail = ({ points, currentIndex, color, maxTailLength = 24 }: FadingTrailProps) => {
  const particles = useMemo(() => {
    if (!points.length) return [];
    const tail: { pos: [number, number, number]; scale: number; opacity: number }[] = [];
    for (let i = 0; i < maxTailLength; i++) {
      const idx = currentIndex - i;
      if (idx < 0) break;
      const pos = points[idx];
      if (!pos) continue;
      const decay = (maxTailLength - i) / maxTailLength;
      tail.push({ pos, scale: 0.04 * decay, opacity: 0.85 * decay * decay });
    }
    return tail;
  }, [points, currentIndex, maxTailLength]);

  return (
    <group>
      {particles.map((p, i) => (
        <mesh key={i} position={p.pos} scale={[p.scale, p.scale, p.scale]}>
          <sphereGeometry args={[1, 8, 8]} />
          <meshBasicMaterial color={color} transparent opacity={p.opacity} />
        </mesh>
      ))}
    </group>
  );
};

// ── Main scene content ──
interface SceneContentProps {
  physicsTrajectory: Float32Array | null;
  aiTrajectory: Float32Array | null;
  timeIndex: number;
  aiTotalSteps: number;
  x2Init: number; z2Init: number;
  x3Init: number; z3Init: number;
  onDrag: (x: number, z: number) => void;
  showPhysics: boolean;
  showAI: boolean;
  isDragging: boolean;
  setIsDragging: (v: boolean) => void;
}

const SceneContent = ({
  physicsTrajectory, aiTrajectory,
  timeIndex, aiTotalSteps,
  x2Init, z2Init, x3Init, z3Init,
  onDrag, showPhysics, showAI,
  isDragging, setIsDragging,
}: SceneContentProps) => {
  // Physics: full trajectory
  const phys1 = useMemo(() => getTrajectoryPoints(physicsTrajectory, 0), [physicsTrajectory]);
  const phys2 = useMemo(() => getTrajectoryPoints(physicsTrajectory, 1), [physicsTrajectory]);
  const phys3 = useMemo(() => getTrajectoryPoints(physicsTrajectory, 2), [physicsTrajectory]);

  // AI: trajectory capped at aiTotalSteps (PINN range t ≤ 10s)
  const ai1 = useMemo(() => getTrajectoryPoints(aiTrajectory, 0, aiTotalSteps), [aiTrajectory, aiTotalSteps]);
  const ai2 = useMemo(() => getTrajectoryPoints(aiTrajectory, 1, aiTotalSteps), [aiTrajectory, aiTotalSteps]);
  const ai3 = useMemo(() => getTrajectoryPoints(aiTrajectory, 2, aiTotalSteps), [aiTrajectory, aiTotalSteps]);

  // Current physics head — always valid
  const pp1 = phys1[timeIndex] ?? phys1[phys1.length - 1] ?? [1.0, 0, 0.0] as [number, number, number];
  const pp2 = phys2[timeIndex] ?? phys2[phys2.length - 1] ?? [x2Init, 0, z2Init] as [number, number, number];
  const pp3 = phys3[timeIndex] ?? phys3[phys3.length - 1] ?? [x3Init, 0, z3Init] as [number, number, number];

  // Current AI head — freeze at last known frame when physics timeline exceeds PINN range
  const safeAIIdx = Math.min(timeIndex, ai1.length - 1);
  const ap1 = ai1[safeAIIdx] ?? [1.0, 0, 0.0] as [number, number, number];
  const ap2 = ai2[safeAIIdx] ?? [x2Init, 0, z2Init] as [number, number, number];
  const ap3 = ai3[safeAIIdx] ?? [x3Init, 0, z3Init] as [number, number, number];

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    if (isDragging) {
      e.stopPropagation();
      onDrag(e.point.x, e.point.z);
    }
  };
  const handlePointerUp = (e: ThreeEvent<PointerEvent>) => {
    if (isDragging) {
      e.stopPropagation();
      setIsDragging(false);
      document.body.style.cursor = 'auto';
    }
  };

  return (
    <>
      <ambientLight intensity={0.35} />
      <directionalLight position={[8, 10, 5]} intensity={0.9} />
      <Stars radius={100} depth={50} count={4000} factor={4} saturation={0.4} fade speed={0.8} />
      <gridHelper args={[20, 40, '#1a2030', '#0d1218']} position={[0, -0.01, 0]} />

      {/* Origin crosshair */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.018, 12, 12]} />
        <meshBasicMaterial color="#374151" transparent opacity={0.6} />
      </mesh>

      {/* ── Initial position markers ── */}

      {/* Body 1 — fixed, neon blue ring */}
      <mesh position={[1.0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.07, 0.09, 32]} />
        <meshBasicMaterial color="#22d3ee" transparent opacity={0.45} side={THREE.DoubleSide} />
      </mesh>

      {/* Body 2 — draggable, amber ring */}
      <group position={[x2Init, 0, z2Init]}>
        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          onPointerDown={(e) => {
            e.stopPropagation();
            setIsDragging(true);
            document.body.style.cursor = 'grabbing';
          }}
          onPointerOver={() => { if (!isDragging) document.body.style.cursor = 'grab'; }}
          onPointerOut={() => { if (!isDragging) document.body.style.cursor = 'auto'; }}
        >
          <ringGeometry args={[0.07, 0.12, 32]} />
          <meshBasicMaterial color={isDragging ? '#f59e0b' : '#fcd34d'} transparent opacity={0.85} side={THREE.DoubleSide} />
        </mesh>
        {/* Inner dot */}
        <mesh position={[0, 0.003, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0, 0.04, 16]} />
          <meshBasicMaterial color="#fcd34d" transparent opacity={0.35} />
        </mesh>
      </group>

      {/* Body 3 — COM-dependent, emerald ring */}
      <mesh position={[x3Init, 0, z3Init]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.07, 0.09, 32]} />
        <meshBasicMaterial color="#34d399" transparent opacity={0.45} side={THREE.DoubleSide} />
      </mesh>

      {/* ── Full trajectory lines ── */}
      {showPhysics && (
        <>
          <TrailLine points={phys1} color="#06b6d4" opacity={0.3} />
          <TrailLine points={phys2} color="#06b6d4" opacity={0.3} />
          <TrailLine points={phys3} color="#06b6d4" opacity={0.3} />
        </>
      )}
      {showAI && (
        <>
          <TrailLine points={ai1} color="#ec4899" opacity={0.3} />
          <TrailLine points={ai2} color="#ec4899" opacity={0.3} />
          <TrailLine points={ai3} color="#ec4899" opacity={0.3} />
        </>
      )}

      {/* ── Moving heads + fading tails ── */}
      {showPhysics && (
        <>
          <FadingTrail points={phys1} currentIndex={timeIndex} color="#00e5ff" />
          <FadingTrail points={phys2} currentIndex={timeIndex} color="#00e5ff" />
          <FadingTrail points={phys3} currentIndex={timeIndex} color="#00e5ff" />
          {([pp1, pp2, pp3] as [number, number, number][]).map((pos, i) => (
            <mesh key={`phys-${i}`} position={pos}>
              <sphereGeometry args={[0.07, 32, 32]} />
              <meshBasicMaterial color="#22d3ee" />
            </mesh>
          ))}
        </>
      )}
      {showAI && (
        <>
          <FadingTrail points={ai1} currentIndex={safeAIIdx} color="#ff2d78" />
          <FadingTrail points={ai2} currentIndex={safeAIIdx} color="#ff2d78" />
          <FadingTrail points={ai3} currentIndex={safeAIIdx} color="#ff2d78" />
          {([ap1, ap2, ap3] as [number, number, number][]).map((pos, i) => (
            <mesh key={`ai-${i}`} position={pos}>
              <sphereGeometry args={[0.055, 32, 32]} />
              <meshBasicMaterial color="#f43f5e" />
            </mesh>
          ))}
        </>
      )}

      {/* ── Invisible drag capture plane at Y=0 ── */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} visible={false}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerOut={handlePointerUp}>
        <planeGeometry args={[200, 200]} />
      </mesh>
    </>
  );
};

// ── Canvas wrapper (exported) ──
interface OrbitCanvasProps {
  physicsTrajectory: Float32Array | null;
  aiTrajectory: Float32Array | null;
  timeIndex: number;
  aiTotalSteps: number;
  x2Init: number; z2Init: number;
  x3Init: number; z3Init: number;
  onDrag: (x: number, z: number) => void;
  showPhysics: boolean;
  showAI: boolean;
}

export const OrbitCanvas = (props: OrbitCanvasProps) => {
  const [isDragging, setIsDragging] = useState(false);

  return (
    <div className="w-full h-full touch-none select-none">
      <Canvas camera={{ position: [0, 5.5, 5.5], fov: 48 }} gl={{ antialias: true }}>
        <SceneContent {...props} isDragging={isDragging} setIsDragging={setIsDragging} />

        <OrbitControls
          enableRotate={!isDragging}
          enablePan={true}
          maxPolarAngle={Math.PI / 2 - 0.04}
          minDistance={1.5}
          maxDistance={30}
        />

        <EffectComposer>
          <Bloom luminanceThreshold={0.04} luminanceSmoothing={0.85} height={480} intensity={2.0} />
        </EffectComposer>
      </Canvas>
    </div>
  );
};
