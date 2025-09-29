import os, time, json, math, random
import torch
import torch.nn as nn
import torch.optim as optim
from sympy import trunc
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from torchvision.models import resnet18, ResNet18_Weights

seed = 2
# epoch = 200
num_epochs = 200

# ---- 0. 可复现设置 ----
def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # True更稳但慢；False 允许CUDNN自动调优
    torch.backends.cudnn.benchmark = True

set_seed(seed)

# ---- 1. 设备 ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
# if device.type == "cuda":
#     print("GPU:", torch.cuda.get_device_name(0))

# ---- 2. 数据集 & 增强 ----
train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
])
test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
])

root = "./data"
train_set = torchvision.datasets.CIFAR10(root=root, train=True,  download=True, transform=train_tf)
test_set  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)

# Windows 上 num_workers 建议先用 0，Linux 可用 4 或更多
# train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=0, pin_memory=(device.type=="cuda"))
# test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

batch = 256
train_loader = DataLoader(train_set,batch_size=batch,shuffle=True)
test_loader = DataLoader(test_set,batch_size=2*batch,shuffle=False)

# ---- 3. 模型 / 损失 / 优化器 / 学习率调度 ----
def resnet18_change_stem(num_classes=10):
    m = resnet18(weights=None, num_classes=num_classes)
    # 替换 stem：7x7/2 -> 3x3/1，去掉 maxpool
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

model = resnet18_change_stem(num_classes=10).to(device)

# model = torchvision.models.resnet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# 混合精度：RTX 40/50 系列强烈推荐（更快更省显存）
# scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
scaler = torch.amp.GradScaler("cuda",enabled=(device.type=="cuda") )

# ---- 4. 训练与评估函数 ----
def train_one_epoch(epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for images, labels in train_loader:
        # images = images.to(device, non_blocking=True) 和上面的pin_memory配套
        # labels = labels.to(device, non_blocking=True)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    print(f"[Train] Epoch {epoch:03d} | loss {epoch_loss:.4f} | acc {epoch_acc*100:.2f}% | lr {scheduler.get_last_lr()[0]:.4f} | {time.time()-t0:.1f}s")
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(epoch, split="Test"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
        running_loss += loss.item() * labels.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    print(f"[{split}] Epoch {epoch:03d} | loss {epoch_loss:.4f} | acc {epoch_acc*100:.2f}%")
    return epoch_loss, epoch_acc

# ---- 5. 训练主循环 ----
os.makedirs("results", exist_ok=True)
best_acc, best_path = 0.0, f"results/best_{seed}_{num_epochs}.pt"


metrics = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}
for epoch in range(1, num_epochs+1):
    tr_loss, tr_acc = train_one_epoch(epoch)
    te_loss, te_acc = evaluate(epoch, split="Test")
    metrics["train_loss"].append(tr_loss); metrics["train_acc"].append(tr_acc)
    metrics["test_loss"].append(te_loss);  metrics["test_acc"].append(te_acc)

    # 保存最好模型
    if te_acc > best_acc:
        best_acc = te_acc
        torch.save({"model": model.state_dict(),
                    "acc": best_acc,
                    "epoch": epoch}, best_path)
        print(f"✓ Saved best model: acc={best_acc*100:.2f}% @ epoch {epoch}")

# 保存指标
with open(f"results/metrics_{seed}_{num_epochs}.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f)
print(f"Best Test Acc: {best_acc*100:.2f}% | Weights saved to {best_path}")

