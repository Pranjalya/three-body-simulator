// AI Inference Web Worker for PINN Model
// Runs local inference via onnxruntime-web in CPU WebAssembly.

import * as ort from 'onnxruntime-web';

interface AIInput {
  x2: number;
  z2: number;
  tMax: number;
  dt: number;
}

let session: ort.InferenceSession | null = null;
let isInitializing = false;
const initQueue: (() => void)[] = [];

// Initialize the ONNX session
async function initSession(): Promise<ort.InferenceSession> {
  if (session) return session;

  if (isInitializing) {
    return new Promise((resolve) => {
      initQueue.push(() => resolve(session!));
    });
  }

  isInitializing = true;

  try {
    // IMPORTANT: wasmPaths CDN version MUST match the installed onnxruntime-web npm version exactly.
    // Installed: onnxruntime-web@1.27.0 → CDN path set to matching version.
    // We disable numThreads to avoid the multi-threaded JSEP variant (ort-wasm-simd-threaded.jsep.mjs)
    // which requires SharedArrayBuffer cross-origin isolation headers not available in dev.
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
    ort.env.wasm.numThreads = 1; // Use single-threaded wasm — avoids JSEP/SharedArrayBuffer requirement

    // Create inference session
    // The ONNX model is copied to the public/ folder of Vite, so it is served at root '/pinn_model_2d_v2.onnx'
    // The external weights file (.onnx.data) MUST be passed explicitly — the WASM backend cannot
    // auto-resolve sibling data files over HTTP the way the native runtime can.
    session = await ort.InferenceSession.create('/pinn_model_2d_v2.onnx', {
      executionProviders: ['wasm'],
      externalData: [
        {
          // 'path' must match the name stored inside the .onnx protobuf (usually just the filename)
          path: 'pinn_model_2d_v2.onnx.data',
          // 'data' is the URL from which ort-web will fetch the binary weights blob
          data: '/pinn_model_2d_v2.onnx.data',
        },
      ],
    });

    isInitializing = false;
    // Process queued requests
    while (initQueue.length > 0) {
      const resolveFn = initQueue.shift();
      if (resolveFn) resolveFn();
    }

    return session;
  } catch (error) {
    isInitializing = false;
    console.error('Failed to initialize ONNX Runtime Web session:', error);
    throw error;
  }
}

// ── Queue serialization mechanism to prevent overlapping InferenceSession.run calls ──
// Calling activeSession.run concurrently on a single WASM session context triggers "Session mismatch".
let isProcessing = false;
let nextRequest: AIInput | null = null;

self.onmessage = (event: MessageEvent<AIInput>) => {
  nextRequest = event.data;
  processQueue();
};

async function processQueue() {
  if (isProcessing || !nextRequest) return;

  isProcessing = true;
  const currentRequest = nextRequest;
  nextRequest = null; // Clear queue so subsequent messages register as the next request

  const startTime = performance.now();
  const { x2, z2, dt } = currentRequest;
  const tMax = 10.0; // PINN model hard capped at 10.0s

  try {
    const activeSession = await initSession();

    const numIntervals = Math.round(tMax / dt);
    const numSteps = numIntervals + 1;

    // Construct input data: batch_size x 7
    // Input features: [x1, z1, x2, z2, x3, z3, t]
    // where x1=1.0, z1=0.0 are fixed initial positions of Body 1
    // x3=-1.0-x2, z3=-z2 are dependent initial positions of Body 3
    const inputData = new Float32Array(numSteps * 7);

    for (let step = 0; step < numSteps; step++) {
      const t = step * dt;
      const idx = step * 7;

      inputData[idx + 0] = 1.0;          // x1 initial
      inputData[idx + 1] = 0.0;          // z1 initial
      inputData[idx + 2] = x2;           // x2 initial (interactive)
      inputData[idx + 3] = z2;           // z2 initial (interactive)
      inputData[idx + 4] = -1.0 - x2;    // x3 initial (dependent)
      inputData[idx + 5] = -z2;          // z3 initial (dependent)
      inputData[idx + 6] = t;            // target time t
    }

    // Create tensor object
    const inputTensor = new ort.Tensor('float32', inputData, [numSteps, 7]);

    // Get input and output names dynamically to support minor model schema updates
    const inputName = activeSession.inputNames[0] || 'input';
    const outputName = activeSession.outputNames[0] || 'output';

    // Run inference (fully serialized - no concurrent overlap)
    const feeds = { [inputName]: inputTensor };
    const outputs = await activeSession.run(feeds);
    const outputTensor = outputs[outputName];

    if (!outputTensor) {
      throw new Error(`Output tensor "${outputName}" was not found in inference results.`);
    }

    // Copy data to a fresh Float32Array to safely transfer its buffer back to main thread.
    const rawOutputData = outputTensor.data as Float32Array;
    const trajectory = new Float32Array(rawOutputData);

    const duration = performance.now() - startTime;

    // Cast self to any to bypass window vs worker postMessage typescript typing overloads
    (self as any).postMessage(
      {
        trajectory,
        duration,
        numSteps,
        success: true,
      },
      [trajectory.buffer]
    );
  } catch (error: any) {
    const duration = performance.now() - startTime;
    (self as any).postMessage({
      success: false,
      error: error?.message || String(error),
      duration,
    });
  } finally {
    isProcessing = false;
    // If a new request came in while we were busy, process it on the next tick
    if (nextRequest) {
      setTimeout(processQueue, 0);
    }
  }
}
