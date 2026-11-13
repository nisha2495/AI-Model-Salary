import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=== AI Salary Prediction Model - PyTorch ===")

# 1. DATA BANANA - Position Level (1-10) -> Salary
# Jese video me tha, Position badhegi to Salary badhegi
np.random.seed(42)
position = np.random.randint(1, 11, 200)  # 1 = Junior, 10 = CEO
experience = np.random.randint(0, 20, 200)
# Salary ka formula + thora noise
salary = position * 15000 + experience * 3000 + np.random.normal(0, 5000, 200)

X = np.column_stack((position, experience))
y = salary.reshape(-1, 1)

print(f"[+] Data Ready: {len(X)} employees")
print(f"[+] Example: Position {position[0]}, Exp {experience[0]} yrs -> Salary {salary[0]:.0f}")

# 2. DATA KO TRAIN / TEST ME TORNA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train)
y_test_s = scaler_y.transform(y_test)
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
y_test_t = torch.tensor(y_test_s, dtype=torch.float32)

# 3. NEURAL NETWORK BANANA - Video wala hi model
class SalaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)

model = SalaryModel()
print(f"\n[+] Model Created:")
print(model)

# 4. TRAINING
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/300 - Loss: {loss.item():.4f}")

# 5. EVALUATION
model.eval()
with torch.no_grad():
    test_pred = model(X_test_t)
    test_loss = loss_fn(test_pred, y_test_t)
    print(f"\n[+] TEST LOSS: {test_loss.item():.4f} - PASS")

    # Ek naya banda predict karo
    # Position 7 (Manager), Experience 10 saal
    new_emp = scaler_X.transform([[7, 10]])
    new_emp_t = torch.tensor(new_emp, dtype=torch.float32)
    pred_salary_s = model(new_emp_t).numpy()
    pred_salary = scaler_y.inverse_transform(pred_salary_s)
    print(f"[+] PREDICTION: Position 7, 10 yrs Exp -> Salary = {pred_salary[0][0]:.0f} PKR")

# 6. GRAPH - Portfolio ke liye screenshot
plt.plot(losses)
plt.title("AI Model Training Loss - Salary Prediction")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("training_loss.png")
print("\n[DONE] Graph saved as training_loss.png")
print("[DONE] Model Complete!")