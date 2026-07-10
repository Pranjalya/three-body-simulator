# Operational Steps: PINN Three-Body Simulator

This document lists all the actions you need to take to generate data, train models, perform hyperparameter tuning, and validate the results.

---

## 0. Setup Environment

Before running any script, make sure your virtual environment is active and dependencies are correct:
```bash
# From model_training/pinn/ directory
source .venv/bin/activate
```

> [!NOTE]
> The scripts automatically detect and utilize **NVIDIA GPUs (CUDA)** and **Apple Silicon GPUs (MPS on M1/M2/M3)** for hardware acceleration. If neither is available, it defaults to CPU execution.

---

## 1. Step 1: Data Generation

We use numerical integration (`DOP853`) to generate valid planar 3-body trajectories with equal masses. 

To generate a dataset, run `generate_data.py`:
```bash
python3 generate_data.py --num_simulations 100 --output_path three_body_data.npz --cores 4
```

### Parameters for `generate_data.py`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--num_simulations` | `int` | `100` | Number of successful (non-colliding) trajectories to generate. For full paper replication, set to `30000`. |
| `--output_path` | `str` | `three_body_data.npz` | Output filepath to save the compressed NPZ file. |
| `--t_max` | `float` | `10.0` | Total time duration of each simulation (in normalized time units). |
| `--dt` | `float` | `0.0390625` | Timestep size for evaluation (results in $10 / 0.0390625 = 256$ intervals, or $257$ points). |
| `--cores` | `int` | *All CPUs* | Number of CPU cores to utilize for parallel trajectory integration. |
| `--seed` | `int` | `42` | Random seed for initial condition generation (ensures reproducibility). |

---

## 2. Step 2: Training the Model

You can train either a Standard feedforward DNN (baseline) or the Physics-Informed Neural Network (PINN) with ResNet connections.

### A. Train a ResNet PINN (Recommended)
```bash
python3 train.py --data_path three_body_data.npz --model_type resnet --use_pinn true --save_path best_pinn_model.pt
```

### B. Train a Baseline DNN (No Physics Constraints)
```bash
python3 train.py --data_path three_body_data.npz --model_type standard --use_pinn false --save_path best_dnn_model.pt
```

### Parameters for `train.py`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data_path` | `str` | `three_body_data.npz` | Path to the NPZ file generated in Step 1. |
| `--model_type` | `str` | `resnet` | Model architecture choice: `resnet` (skip connections) or `standard` (feedforward). |
| `--depth` | `int` | `12` | Total number of layers. Note: For `resnet`, depth-2 must be even. |
| `--width` | `int` | `256` | Hidden dimension (number of units per hidden layer). |
| `--activation` | `str` | `relu` | Nonlinear activation unit. Choices: `relu`, `gelu`, `tanh`, `leaky_relu`. |
| `--epochs` | `int` | `500` | Maximum number of training epochs. |
| `--batch_size` | `int` | `5000` | Size of mini-batches for stochastic gradient descent. |
| `--lr` | `float` | `7.5e-4` | Initial learning rate. |
| `--weight_decay` | `float` | `1e-5` | L2 weight regularization factor. |
| `--clip_grad` | `float` | `1.0` | Max gradient norm value for clipping (helps prevent exploding gradients). |
| `--use_pinn` | `str` | `true` | Enable physics-informed loss term (`true` or `false`). |
| `--alpha_init` | `float` | `0.001` | Starting weight for the physics-informed loss term $\alpha$. |
| `--alpha_final` | `float` | `0.75` | Ending weight for the physics-informed loss term $\alpha$ at last epoch. |
| `--save_path` | `str` | `best_model.pt` | Output path to save the best validation checkpoint. |
| `--seed` | `int` | `42` | Random seed for initialization. |

---

## 3. Step 3: TensorBoard Visualization

Training automatically logs progress metrics (data loss, physics loss, learning rate, and alpha weight) to TensorBoard.

To start TensorBoard and view the logs in your browser:
```bash
tensorboard --logdir runs/
```
Then, open your browser and navigate to `http://localhost:6006/` to inspect real-time plots of train and validation losses.

---

## 4. Step 4: Hyperparameter Tuning

To find the best network setup, you can experiment by adjusting the following parameters:

1. **Varying Capacity (Width & Depth)**:
   - For a lighter model (faster training but lower capacity):
     `--width 128 --depth 6`
   - For a heavier model (highest capacity, slower training):
     `--width 512 --depth 12`

2. **Varying Activation Units**:
   - The paper shows `relu` works best. However, you can try `gelu` or `tanh` which are popular in PINNs to get smooth gradients:
     `--activation gelu` or `--activation tanh`

3. **Tuning Physics weight ($\alpha$)**:
   - If the physics loss is too weak and the model doesn't conserve energy, increase the final weight:
     `--alpha_final 1.5`
   - If the model is failing to learn the data coordinates because physics dominates too early, decrease it:
     `--alpha_final 0.25`

---

## 5. Step 5: Model Validation & Visualization

To evaluate a trained checkpoint on a validation trajectory and generate a comparison plot, execute:
```bash
python3 -c "import sys; sys.path.append('.'); from utils import evaluate_and_plot; evaluate_and_plot('best_pinn_model.pt', 'three_body_data.npz', sim_idx=0, save_dir='plots')"
```

### Outputs of Validation
- **Console Log**: Prints the relative Mean Absolute Error (MAE) and the energy conservation deviation for both the true integration and the model's predictions.
- **Plot**: Saves a plot at `plots/sim_{sim_idx}_comparison.png` containing:
  1. **Trajectory Plot**: The orbits of all three bodies in the X-Z plane (ground truth vs. prediction).
  2. **Energy Conservation Curve**: Total system energy, potential energy, and kinetic energy over time. An ideal prediction preserves flat total energy over time.
