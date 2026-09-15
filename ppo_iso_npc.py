!pip install onnx onnxscript
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import math
import copy
import random
import json
import csv

# ==========================================
# 0. CỐ ĐỊNH SEED (REPRODUCIBILITY)
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 1. CẤU HÌNH (HYPERPARAMETERS)
# ==========================================
class Config:
    seed = 42
    eval_seed = 12345  # Seed riêng, CỐ ĐỊNH cho môi trường Evaluation
    obs_dim = 6
    num_actions = 10  # 0: Idle | 1-8: Di chuyển 8 hướng | 9: Tấn công

    # --- Giới hạn bản đồ (mô phỏng biên map trong Unity) ---
    map_bound = 8.0        # NPC bị giới hạn trong khung [-8, 8] x [-8, 8]
    wall_penalty = -0.05 

    # --- Reward shaping (potential-based, không phá vỡ optimal policy) ---
    shaping_coef = 0.05 

    # --- Reward combat ---
    hit_reward = 2.0
    kill_reward = 20.0

    # --- PPO core hyperparameters ---
    lr = 3e-4
    anneal_lr = True       # Linear LR decay
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.005
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = 0.02       # Ngưỡng Early Stop cho KL Divergence

    total_steps = 1000000   # Tổng số step môi trường
    rollout_length = 2048
    ppo_epochs = 4
    mini_batch_size = 128

    eval_interval = 5       # Mỗi 5 Update sẽ chạy Evaluation
    eval_episodes = 20      # Đánh giá trên 20 ván

    log_csv_path = "training_log.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 2. MÔI TRƯỜNG GIẢ LẬP ISOMETRIC 2D - 8 HƯỚNG
# ==========================================
class IsoTacticalEnv:
    """
    Mô phỏng combat 1v1 trên lưới 2D (logic isometric, di chuyển 8 hướng).
    Việc hiển thị isometric chỉ là render-space ở Unity, logic toạ độ
    (dx, dy) ở đây vẫn là hệ trục vuông thông thường nên không cần đổi.

    ACTION SPACE (PHẢI khớp với enum hướng đi bên Unity / C#):
        0 = Idle
        1 = Move North       (dx= 0   , dy=+1)
        2 = Move North-East  (dx=+0.7 , dy=+0.7)
        3 = Move East        (dx=+1   , dy= 0)
        4 = Move South-East  (dx=+0.7 , dy=-0.7)
        5 = Move South       (dx= 0   , dy=-1)
        6 = Move South-West  (dx=-0.7 , dy=-0.7)
        7 = Move West        (dx=-1   , dy= 0)
        8 = Move North-West  (dx=-0.7 , dy=+0.7)
        9 = Attack

    OBSERVATION (6 chiều, float32):
        [agent_hp, dx_to_enemy/10, dy_to_enemy/10, dist/10, enemy_hp, in_range_flag]

    rng: numpy.random.Generator dùng riêng cho môi trường này (thay vì state
    global np.random). Cho phép tạo môi trường Evaluation với seed CỐ ĐỊNH,
    độc lập với chuỗi random đang dùng để Train -> Eval Return không bị
    nhiễu do tình huống (vị trí địch, máu địch...) thay đổi giữa các lần eval.
    """

    def __init__(self, map_bound=8.0, wall_penalty=-0.05, shaping_coef=0.05,
                 hit_reward=1.0, kill_reward=10.0, rng=None):
        self.map_bound = map_bound
        self.wall_penalty = wall_penalty
        self.shaping_coef = shaping_coef
        self.hit_reward = hit_reward
        self.kill_reward = kill_reward
        self.rng = rng if rng is not None else np.random.default_rng()
        self.dir_map = [
            (0, 0), (0, 1), (0.7, 0.7), (1, 0), (0.7, -0.7),
            (0, -1), (-0.7, -0.7), (-1, 0), (-0.7, 0.7)
        ]
        self.reset()

    def reset(self):
        self.step_count = 0
        self.agent_hp = 1.0
        self.agent_pos = np.array([0.0, 0.0])

        self.enemy_hp = self.rng.uniform(0.2, 1.0)
        angle = self.rng.uniform(0, 2 * math.pi)
        dist = self.rng.uniform(2.0, 5.0)
        self.enemy_pos = np.array([math.cos(angle) * dist, math.sin(angle) * dist])

        return self._get_obs()

    def step(self, action):
        self.step_count += 1
        reward = -0.01  # Phạt thời gian
        done = False

        dist_before = np.linalg.norm(self.enemy_pos - self.agent_pos)

        # --- AGENT DI CHUYỂN (8 HƯỚNG) ---
        if 1 <= action <= 8:
            dx, dy = self.dir_map[action]
            new_pos = self.agent_pos + np.array([dx, dy]) * 0.5
            clipped_pos = np.clip(new_pos, -self.map_bound, self.map_bound)
            if not np.allclose(new_pos, clipped_pos):
                reward += self.wall_penalty  # Va vào biên bản đồ
            self.agent_pos = clipped_pos

        # Cập nhật khoảng cách sau khi agent di chuyển
        dist_to_enemy = np.linalg.norm(self.enemy_pos - self.agent_pos)

        # --- AGENT TẤN CÔNG ---
        if action == 9:
            if dist_to_enemy <= 1.5:
                self.enemy_hp -= 0.3
                reward += self.hit_reward
                if self.enemy_hp <= 0:
                    reward += self.kill_reward
                    done = True
            else:
                reward -= 0.1

        # --- ENEMY HÀNH ĐỘNG CƠ BẢN ---
        if not done:
            if self.rng.random() < 0.3:
                move_dir = (self.enemy_pos - self.agent_pos) if self.enemy_hp < 0.4 else self.rng.standard_normal(2)
                move_dir = move_dir / (np.linalg.norm(move_dir) + 1e-8)
                new_enemy_pos = self.enemy_pos + move_dir * 0.3
                self.enemy_pos = np.clip(new_enemy_pos, -self.map_bound, self.map_bound)

                dist_to_enemy = np.linalg.norm(self.enemy_pos - self.agent_pos)

            if dist_to_enemy <= 2.0 and self.step_count % 2 == 0:
                self.agent_hp -= 0.2
                reward -= 0.5

        # --- REWARD SHAPING: khuyến khích tiến gần địch ---
        reward += self.shaping_coef * (dist_before - dist_to_enemy)

        # --- KIỂM TRA ĐIỀU KIỆN KẾT THÚC ---
        if self.agent_hp <= 0:
            reward -= 5.0
            done = True

        if self.step_count >= 100:
            done = True

        return self._get_obs(), reward, done

    def _get_obs(self):
        dx = self.enemy_pos[0] - self.agent_pos[0]
        dy = self.enemy_pos[1] - self.agent_pos[1]
        dist = np.linalg.norm([dx, dy])

        return np.array([
            self.agent_hp, dx / 10.0, dy / 10.0, dist / 10.0,
            self.enemy_hp, 1.0 if dist <= 2.0 else 0.0
        ], dtype=np.float32)


# ==========================================
# 3. MẠNG ACTOR-CRITIC & EVALUATION
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ProActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(cfg.obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(cfg.obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, cfg.num_actions), std=0.01),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


def evaluate(model, cfg, env_kwargs=None):
    """
    Đánh giá mô hình một cách Deterministic (argmax).

    Dùng RNG riêng với seed CỐ ĐỊNH (cfg.eval_seed) cho môi trường eval,
    tạo mới mỗi lần gọi -> mọi lần evaluate() trong suốt quá trình train
    đều chạy trên đúng cfg.eval_episodes tình huống giống nhau. Nhờ vậy
    đường Eval Return chỉ phản ánh sự thay đổi của POLICY, không bị nhiễu
    bởi việc tình huống random khác nhau giữa các lần eval.
    """
    eval_kwargs = dict(env_kwargs or {})
    eval_kwargs["rng"] = np.random.default_rng(cfg.eval_seed)
    env = IsoTacticalEnv(**eval_kwargs)
    model.eval()
    device = next(model.parameters()).device
    total_rewards = []

    with torch.no_grad():
        for _ in range(cfg.eval_episodes):
            obs = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
                logits, _ = model(obs_tensor)
                action = logits.argmax(dim=-1).item()  # KHÔNG sampling
                obs, reward, done = env.step(action)
                ep_reward += reward
            total_rewards.append(ep_reward)

    model.train()
    return float(np.mean(total_rewards))


# ==========================================
# 4. THUẬT TOÁN PPO CHÍNH
# ==========================================
def train():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    env_kwargs = dict(
        map_bound=cfg.map_bound,
        wall_penalty=cfg.wall_penalty,
        shaping_coef=cfg.shaping_coef,
        hit_reward=cfg.hit_reward,
        kill_reward=cfg.kill_reward,
    )

    env = IsoTacticalEnv(rng=np.random.default_rng(cfg.seed), **env_kwargs)
    model = ProActorCritic(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, eps=1e-5)

    num_updates = cfg.total_steps // cfg.rollout_length
    global_step = 0

    best_eval_return = -float("inf")
    best_state_dict = None

    # Giá trị log mặc định -> tránh tham chiếu biến chưa định nghĩa
    # khi vòng lặp PPO bị Early-Stop ngay từ minibatch đầu tiên.
    last_approx_kl = 0.0
    last_entropy = 0.0
    last_actor_loss = 0.0
    last_critic_loss = 0.0
    last_clipfrac = 0.0
    last_explained_var = 0.0

    print(f"=== BẮT ĐẦU HUẤN LUYỆN PPO (Total Steps: {cfg.total_steps}, Device: {device}) ===")
    obs = env.reset()

    # --- LOGGING RA CSV (để vẽ chart Eval Return / KL / LR sau khi train) ---
    log_file = open(cfg.log_csv_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "update", "global_step", "lr", "approx_kl", "clipfrac",
        "entropy", "actor_loss", "critic_loss", "explained_variance", "eval_return"
    ])

    for update in range(1, num_updates + 1):
        # --- Linear LR Annealing (chi tiết chuẩn của PPO) ---
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.lr

        b_obs, b_actions, b_logprobs, b_rewards, b_values, b_dones = [], [], [], [], [], []

        # --- THU THẬP DỮ LIỆU (ROLLOUT) ---
        for step in range(cfg.rollout_length):
            global_step += 1
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, value = model(obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, done = env.step(action.item())

            b_obs.append(obs)
            b_actions.append(action.item())
            b_logprobs.append(log_prob.item())
            b_rewards.append(reward)
            b_values.append(value.item())
            b_dones.append(done)

            obs = next_obs if not done else env.reset()

        # --- TÍNH LỢI THẾ (GAE) ---
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
            _, next_value = model(obs_tensor)
            next_value = next_value.item()

            returns, advantages = [], []
            gae = 0
            values = b_values + [next_value]

            for step in reversed(range(len(b_rewards))):
                delta = b_rewards[step] + cfg.gamma * values[step + 1] * (1 - int(b_dones[step])) - values[step]
                gae = delta + cfg.gamma * cfg.gae_lambda * (1 - int(b_dones[step])) * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[step])

            returns = torch.tensor(returns, dtype=torch.float32, device=device)
            advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        b_obs = torch.tensor(np.array(b_obs), dtype=torch.float32, device=device)
        b_actions = torch.tensor(b_actions, dtype=torch.long, device=device)
        b_logprobs = torch.tensor(b_logprobs, dtype=torch.float32, device=device)
        b_values = torch.tensor(b_values, dtype=torch.float32, device=device)

        # --- EXPLAINED VARIANCE: Critic dự đoán "returns" tốt tới đâu ---
        # (tính 1 lần/update, dùng values THU THẬP LÚC ROLLOUT, trước khi update)
        # ~1.0 -> Critic gần như hoàn hảo | ~0 -> Critic vô dụng (như đoán mean)
        # âm    -> Critic còn TỆ HƠN cả việc đoán trung bình
        y_pred = b_values.cpu().numpy()
        y_true = returns.cpu().numpy()
        var_y = np.var(y_true)
        last_explained_var = 1.0 - np.var(y_true - y_pred) / (var_y + 1e-8)

        # --- TỐI ƯU HÓA (PPO UPDATE) ---
        b_inds = np.arange(cfg.rollout_length)

        for epoch in range(cfg.ppo_epochs):
            np.random.shuffle(b_inds)
            continue_training = True

            for start in range(0, cfg.rollout_length, cfg.mini_batch_size):
                end = start + cfg.mini_batch_size
                mb_inds = b_inds[start:end]

                logits, values = model(b_obs[mb_inds])
                values = values.squeeze(-1)

                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(b_actions[mb_inds])
                entropy = dist.entropy().mean()

                logratio = new_logprobs - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # TÍNH KL DIVERGENCE & CLIP FRACTION (để log/early-stop)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()

                last_approx_kl = approx_kl
                last_entropy = entropy.item()
                last_clipfrac = clipfrac

                if approx_kl > cfg.target_kl:
                    continue_training = False
                    break  # Dừng Epoch sớm để bảo vệ Policy

                mb_advantages = advantages[mb_inds]
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                v_loss_unclipped = (values - returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(values - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef)
                v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = actor_loss - cfg.ent_coef * entropy + cfg.vf_coef * critic_loss
                last_actor_loss = actor_loss.item()
                last_critic_loss = critic_loss.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if not continue_training:
                break  # Phá vòng lặp Epoch ngoài cùng

        # --- EVALUATION & LƯU BEST MODEL ---
        current_lr = optimizer.param_groups[0]["lr"]
        eval_return_str = ""

        if update % cfg.eval_interval == 0:
            eval_return = evaluate(model, cfg, env_kwargs)
            eval_return_str = f"{eval_return:.4f}"

            if eval_return > best_eval_return:
                best_eval_return = eval_return
                best_state_dict = copy.deepcopy(model.state_dict())
                saved_mark = "(*)"
            else:
                saved_mark = ""

            print(f"Upd {update:03d} | Step: {global_step:06d} | Eval Ret: {eval_return:6.2f} {saved_mark} | "
                  f"KL: {last_approx_kl:.5f} | ClipFrac: {last_clipfrac:.3f} | "
                  f"Ent: {last_entropy:.3f} | Act Loss: {last_actor_loss:.4f} | "
                  f"Critic Loss: {last_critic_loss:.4f} | ExplVar: {last_explained_var:.3f} | LR: {current_lr:.2e}")

        # Ghi log mỗi update: KL/LR/Entropy/ClipFrac/Loss/ExplainedVariance có ở mọi update,
        # Eval Return chỉ có giá trị vào những update tới kỳ eval (còn lại để trống)
        log_writer.writerow([
            update, global_step, f"{current_lr:.8e}", f"{last_approx_kl:.6f}",
            f"{last_clipfrac:.6f}", f"{last_entropy:.6f}", f"{last_actor_loss:.6f}",
            f"{last_critic_loss:.6f}", f"{last_explained_var:.6f}", eval_return_str
        ])
        log_file.flush()

    log_file.close()
    print(f"[LOG] Đã lưu log training (KL/LR/Entropy/Eval Return...) vào: {cfg.log_csv_path}")

    print("=== HUẤN LUYỆN HOÀN TẤT ===")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Đã phục hồi Best Model (Eval Return: {best_eval_return:.2f}) để chuẩn bị xuất ONNX.")

    return model, cfg


# ==========================================
# 5. XUẤT FILE ONNX (DÙNG CHO UNITY)
# ==========================================
def export_to_onnx(model, cfg):
    model = model.to("cpu")
    model.eval()
    dummy_input = torch.randn(1, cfg.obs_dim)
    dynamic_axes = {'obs': {0: 'batch_size'}, 'logits': {0: 'batch_size'}, 'value': {0: 'batch_size'}}
    onnx_path = "npc_tactical_iso2d_best.onnx"

    export_kwargs = dict(
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['obs'], output_names=['logits', 'value'], dynamic_axes=dynamic_axes
    )

    with torch.no_grad():
        try:
            # dynamo=False -> dùng exporter cũ (TorchScript-based), tạo ra
            # đồ thị ONNX đơn giản, tương thích tốt hơn với Unity Sentis.
            # Cần cài: pip install onnx
            torch.onnx.export(model, dummy_input, onnx_path, dynamo=False, **export_kwargs)
        except TypeError:
            # Một số bản torch cũ không có tham số 'dynamo' -> bỏ qua
            torch.onnx.export(model, dummy_input, onnx_path, **export_kwargs)
    print(f"\n[ONNX] Thành công! Đã xuất Best Model ra file: {onnx_path}")

    # Lưu thêm checkpoint PyTorch (debug / train tiếp nếu cần)
    torch.save(model.state_dict(), "npc_tactical_iso2d_best.pt")
    print("[PyTorch] Đã lưu checkpoint: npc_tactical_iso2d_best.pt")

    # Ghi action mapping ra JSON để code C# bên Unity đối chiếu cho khớp
    action_map = {
        0: "Idle", 1: "Move_N", 2: "Move_NE", 3: "Move_E", 4: "Move_SE",
        5: "Move_S", 6: "Move_SW", 7: "Move_W", 8: "Move_NW", 9: "Attack"
    }
    with open("action_mapping.json", "w", encoding="utf-8") as f:
        json.dump(action_map, f, ensure_ascii=False, indent=2)
    print("[Unity] Đã lưu action_mapping.json để đối chiếu hành động trong C#.")


# ==========================================
# 6. VẼ BIỂU ĐỒ TRAINING (cho báo cáo)
# ==========================================
def plot_training_curves(csv_path="training_log.csv", output_path="training_curves.png", target_kl=0.02):
    """
    Đọc file CSV log (do train() ghi ra) và vẽ 8 biểu đồ:
    Eval Return, Explained Variance, Approx KL, Learning Rate,
    Entropy, Actor Loss, Critic Loss, Clip Fraction.
    Lưu kết quả thành 1 ảnh PNG để chèn vào báo cáo.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # không cần GUI, chỉ xuất file ảnh
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] Chưa cài matplotlib -> bỏ qua vẽ biểu đồ. Cài bằng: pip install matplotlib")
        return

    updates, lrs, kls, clipfracs = [], [], [], []
    entropies, actor_losses, critic_losses, explained_vars = [], [], [], []
    eval_updates, eval_returns = [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = int(row["update"])
            updates.append(u)
            lrs.append(float(row["lr"]))
            kls.append(float(row["approx_kl"]))
            clipfracs.append(float(row["clipfrac"]))
            entropies.append(float(row["entropy"]))
            actor_losses.append(float(row["actor_loss"]))
            critic_losses.append(float(row["critic_loss"]))
            # Clip để biểu đồ không bị "vỡ" thang đo do giá trị âm rất lớn
            # ở vài update đầu (khi var(returns) gần 0).
            explained_vars.append(np.clip(float(row["explained_variance"]), -1.0, 1.0))
            if row["eval_return"] != "":
                eval_updates.append(u)
                eval_returns.append(float(row["eval_return"]))

    fig, axes = plt.subplots(2, 4, figsize=(19, 8))

    axes[0, 0].plot(eval_updates, eval_returns, marker="o", color="tab:blue")
    axes[0, 0].set_title("Eval Return")
    axes[0, 0].set_xlabel("Update")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(updates, explained_vars, color="tab:cyan")
    axes[0, 1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].axhline(1.0, color="green", linestyle=":", linewidth=1, label="lý tưởng")
    axes[0, 1].set_title("Explained Variance (Critic)")
    axes[0, 1].set_xlabel("Update")
    axes[0, 1].set_ylim(-1.05, 1.05)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(updates, kls, color="tab:orange", label="approx_kl")
    axes[0, 2].axhline(target_kl, color="red", linestyle="--", linewidth=1, label="target_kl")
    axes[0, 2].set_title("Approx KL Divergence")
    axes[0, 2].set_xlabel("Update")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    axes[0, 3].plot(updates, lrs, color="tab:green")
    axes[0, 3].set_title("Learning Rate")
    axes[0, 3].set_xlabel("Update")
    axes[0, 3].grid(True, alpha=0.3)

    axes[1, 0].plot(updates, entropies, color="tab:purple")
    axes[1, 0].set_title("Policy Entropy")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(updates, actor_losses, color="tab:red")
    axes[1, 1].set_title("Actor Loss")
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(updates, critic_losses, color="tab:olive")
    axes[1, 2].set_title("Critic Loss")
    axes[1, 2].set_xlabel("Update")
    axes[1, 2].grid(True, alpha=0.3)

    axes[1, 3].plot(updates, clipfracs, color="tab:brown")
    axes[1, 3].set_title("Clip Fraction")
    axes[1, 3].set_xlabel("Update")
    axes[1, 3].grid(True, alpha=0.3)

    fig.suptitle("PPO Training Curves - NPC Isometric 2D (8 hướng)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Đã lưu biểu đồ training vào: {output_path}")


# ==========================================
# 7. RUN
# ==========================================
if __name__ == "__main__":
    trained_model, config = train()
    export_to_onnx(trained_model, config)
    plot_training_curves(config.log_csv_path, target_kl=config.target_kl)
