import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from dotenv import load_dotenv

from data_setup import load_and_preprocess_data
from models import ComplexOverfit, ComplexRegularized, get_simple_model

# Load environment variables (e.g., WANDB_API_KEY)
load_dotenv()

# Global list to store results for final comparison
comparison_results = []
criterion = nn.MSELoss()

def evaluate(model, X_data, y_data):
    model.eval()
    with torch.no_grad():
        preds = model(X_data)
        loss = criterion(preds, y_data)
    return loss.item()

def log_model_result(name, train_loss, val_loss, test_loss, description):
    comparison_results.append([name, train_loss, val_loss, test_loss, description])
    # Also log as a metric for this step
    wandb.log({
        f"{name}_Train_Loss": train_loss,
        f"{name}_Val_Loss": val_loss,
        f"{name}_Test_Loss": test_loss
    })
    print(f"\n--- {name} Results ---")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val Loss:   {val_loss:.6f}")
    print(f"Test Loss:  {test_loss:.6f}")

def l1_loss(model, l1_lambda):
    l1 = sum(p.abs().sum() for p in model.parameters())
    return l1_lambda * l1

def main():
    # 1. Login to W&B
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("WANDB_API_KEY not found in environment. Manual login may be required.")
        wandb.login()

    run = wandb.init(
        project="productivity-prediction",
        name="4_Model_Comparison_Split",
        config={"dataset": "Smartphone_Usage_Productivity"}
    )

    # 2. Load Data
    print("Loading and preprocessing data...")
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Smartphone_Usage_Productivity_Dataset_50000.csv')
    (X_train, y_train), (X_val, y_val), (X_test, y_test), \
    train_loader, val_loader, test_loader, input_dim = load_and_preprocess_data(dataset_path)

    print(f"Train Shape: {X_train.shape}")
    print(f"Val Shape:   {X_val.shape}")
    print(f"Test Shape:  {X_test.shape}")

    # ==========================================
    # 3. Model 1: Single Sample Training
    # ==========================================
    # Overfit completely on 1 sample.
    X_single = X_train[0:1]
    y_single = y_train[0:1]

    model_1 = get_simple_model(input_dim)
    optimizer_1 = optim.Adam(model_1.parameters(), lr=0.1)

    print("\nTraining Model 1 (1 Sample)...")
    for _ in range(200):
        optimizer_1.zero_grad()
        loss = criterion(model_1(X_single), y_single)
        loss.backward()
        optimizer_1.step()

    loss_train = evaluate(model_1, X_single, y_single) 
    loss_val = evaluate(model_1, X_val, y_val)
    loss_test = evaluate(model_1, X_test, y_test)
    log_model_result("Model_1_OneSample", loss_train, loss_val, loss_test, "Overfit on 1 Sample")

    # ==========================================
    # 4. Model 2: 5 Samples Training
    # ==========================================
    # Overfit the simple model on 5 samples
    indices = torch.arange(5)
    X_five = X_train[indices]
    y_five = y_train[indices]

    model_2 = get_simple_model(input_dim)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=0.05)

    print("\nTraining Model 2 (5 Samples)...")
    for _ in range(500):
        optimizer_2.zero_grad()
        loss = criterion(model_2(X_five), y_five)
        loss.backward()
        optimizer_2.step()

    loss_train = evaluate(model_2, X_five, y_five) # Loss on the 5 samples it saw
    loss_val = evaluate(model_2, X_val, y_val)
    loss_test = evaluate(model_2, X_test, y_test)
    log_model_result("Model_2_FiveSamples", loss_train, loss_val, loss_test, "Overfit on 5 Samples")

    # ==========================================
    # 5. Model 3: Complex Model (Overfit Whole Data)
    # ==========================================
    model_3 = ComplexOverfit(input_dim)
    optimizer_3 = optim.Adam(model_3.parameters(), lr=0.001)

    print("\nTraining Model 3 (Complex Overfit)...")
    for epoch in tqdm(range(100)):
        for bx, by in train_loader:
            optimizer_3.zero_grad()
            loss = criterion(model_3(bx), by)
            loss.backward()
            optimizer_3.step()

    loss_train = evaluate(model_3, X_train, y_train)
    loss_val = evaluate(model_3, X_val, y_val)
    loss_test = evaluate(model_3, X_test, y_test)
    log_model_result("Model_3_ComplexOverfit", loss_train, loss_val, loss_test, "Complex/No Reg (Overfitting)")

    # ==========================================
    # 6. Model 4: Complex Model + Regularization (Target Best)
    # ==========================================
    model_4 = ComplexRegularized(input_dim)
    optimizer_4 = optim.Adam(model_4.parameters(), lr=0.001, weight_decay=1e-4)

    print("\nTraining Model 4 (Regularized)...")
    for epoch in tqdm(range(15)):
        for bx, by in train_loader:
            optimizer_4.zero_grad()
            outputs = model_4(bx)
            loss = criterion(outputs, by) + l1_loss(model_4, 1e-5)
            loss.backward()
            optimizer_4.step()

    loss_train = evaluate(model_4, X_train, y_train)
    loss_val = evaluate(model_4, X_val, y_val)
    loss_test = evaluate(model_4, X_test, y_test)
    log_model_result("Model_4_Regularized", loss_train, loss_val, loss_test, "Regularized (Target Best)")

    # Save this best model
    checkpoints_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    torch.save(model_4.state_dict(), os.path.join(checkpoints_dir, "best_model_regularized.pth"))

    # ==========================================
    # 7. Final Comparison & WandB Logging
    # ==========================================
    columns = ["Model", "Train Loss (Specific)", "Val Loss", "Test Loss", "Description"]
    wandb_table = wandb.Table(data=comparison_results, columns=columns)
    wandb.log({"Final Comparison Table": wandb_table})

    data_for_plot = [[r[0], r[3]] for r in comparison_results]
    plot_table = wandb.Table(data=data_for_plot, columns=["Model", "Test Loss (MSE)"])
    wandb.log({
        "Test Loss Comparison": wandb.plot.bar(
            plot_table, "Model", "Test Loss (MSE)", title="Final Test Performance"
        )
    })

    print("\nExperiment Complete. Check WandB Dashboard.")
    wandb.finish()

if __name__ == "__main__":
    main()
