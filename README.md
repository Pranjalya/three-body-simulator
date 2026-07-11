# Three-Body Problem Simulator: PINN vs Classical Integration

Welcome! This is an interactive, high-performance WebGL simulator designed to visualize and compare predictions of a **Physics-Informed Neural Network (PINN)** against a traditional **Runge-Kutta 4th-Order (RK4)** numerical integrator for the planar chaotic Three-Body Problem. 

Everything runs entirely on the client-side CPU (Edge Architecture) using isolated multi-threading to guarantee lag-free 60fps rendering in the browser.

---

## 🌌 The Planar Three-Body Problem
The three-body problem is one of the most famous problems in celestial mechanics. Unlike the two-body problem (which has a neat closed-form solution), the motion of three mutually attracting bodies is highly chaotic and has no general analytical solution. 

For this simulator, I focused on a specific configuration:
* **Initial State**: All three bodies start from rest (zero velocity) and fall under mutual gravitational pull.
* **Degrees of Freedom**: Body 1 is fixed at $(1, 0)$, Body 2's starting position is controlled interactively, and Body 3 is derived using a strict Center of Mass (COM) constraint:
  $$\mathbf{r}_3 = -\frac{m_1\mathbf{r}_1 + m_2\mathbf{r}_2}{m_3}$$
This ensures the net momentum of the system remains zero and the center of mass stays fixed at the origin. Even with these constraints, the resulting trajectories are incredibly chaotic.

---

## 🛠️ What I Did (The Implementation Pipeline)

### 1. Dataset Generation (`/model_training/pinn/generate_data.py`)
To train a data-driven model, I generated high-precision ground truth trajectories:
* **Integrator**: Used SciPy's high-precision adaptive step-size **DOP853** (8th-order Runge-Kutta) integrator.
* **Filter**: Simulating three bodies starting from rest often results in immediate collisions where the gravity terms blow up. I filtered out any singular/colliding trajectories (retaining ~80% of generated configurations).
* **Dataset**: Created a training set consisting of 20,000+ simulation trajectories evaluated on the interval $t \in [0, 10s]$.

### 2. PINN Model Training (`/model_training/pinn/train.py`)
I implemented a **Physics-Informed Neural Network (PINN)** in PyTorch to predict positions over time, adopting the network specifications and training formulation described in the paper:
> **"Advancing Solutions for the Three-Body Problem Through Physics-Informed Neural Networks"** by *Manuel Santos Pereira, Luís Tripa, Nélson Lima, Francisco Caldas, Cláudia Soares (75th International Astronautical Congress, 2024)*.
> *See [research paper PDF](https://arxiv.org/abs/2503.04585) for details.*

* **Architecture**: A deep Residual Network (ResNet) containing 10 hidden layers of 128 units each, using Sinusoidal or GELU activations to model smooth second derivatives.
* **Physics-Informed Loss**: Along with the standard data loss (MSE of predicted vs. true coordinates), I integrated the physical laws of gravity directly into the training loop by evaluating the system's differential equations (PDEs) at collocation points:
  $$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{physics}}$$
  where the physics loss penalizes deviations from $\mathbf{F} = m\mathbf{a}$.
* **Temporal Capping**: The model was trained specifically on $t \in [0, 10s]$. Feeding the network values beyond 10s represents out-of-distribution extrapolation, which causes the AI orbits to visibly diverge from physical reality—a classic visualization of chaos!

### 3. Model Serialization & Export (`/model_training/pinn/inference.py`)
* **ONNX Export**: Serialized the trained PyTorch checkpoint into ONNX format (`pinn_model_2d_v2.onnx`).
* **External Weights Allocation**: Due to the size of the network, weights are split into an external data block (`pinn_model_2d_v2.onnx.data`). This keeps the ONNX schema modular and prevents memory allocation bottlenecks inside WebAssembly.

### 4. Interactive 3D Web Simulator (`/simulator`)
I built a premium, responsive WebGL interface from scratch:
* **Graphics**: Built with **React Three Fiber (R3F)** and **Three.js** inside a responsive Tailwind CSS grid.
* **Multi-Threaded Architecture**:
  * **`physics.worker.ts`**: Runs a high-precision sub-stepped (20×) RK4 integrator in a background worker, allowing the simulation duration to extend up to 100 seconds without locking the UI thread.
  * **`ai.worker.ts`**: Runs the ONNX model using **ONNX Runtime Web (Wasm)**. 
* **Serialized Queue Safety**: Concurrent execution of WASM models on a single session context causes memory corruption (`Session mismatch` errors). I solved this by building a serialized request queue inside the AI worker. Stale/intermediate coordinates are dropped, and only the latest coordinates are processed, resolving all race conditions.
* **UI Controls**:
  * Drag and drop the amber body (Body 2) on the WebGL canvas to change the initial state.
  * Adjust masses ($m_1, m_2, m_3$) to watch the center-of-mass reposition Body 3.
  * Adjust simulation duration ($10s - 100s$) and track real-time telemetry (physics latency, AI inference latency, and RMSE divergence).

---

## 🚀 Getting Started

### Prerequisites
* **Node.js**: v20 or later
* **Python**: 3.10+ (with PyTorch, ONNX, and SciPy)

### 1. Running the Web Simulator
Navigate to the simulator folder, install dependencies, and start the development server:
```bash
cd simulator
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser to interact with the system.

### 2. Training the Model
If you want to train your own version of the PINN model:
```bash
cd model_training/pinn
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run dataset generation
python3 generate_data.py --num_simulations 1000 --output_path three_body_data.npz

# Run model training
python3 train.py --data_path three_body_data.npz --epochs 200 --model_path best_model.pt

# Export the trained model to ONNX
python3 inference.py --model_path best_model.pt --export_onnx pinn_model_2d_v2.onnx
```

---

**Developed by Pranjalya Tiwari**  
* GitHub: [github.com/Pranjalya](https://github.com/Pranjalya)
