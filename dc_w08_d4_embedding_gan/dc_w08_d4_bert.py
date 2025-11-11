# =========================================================
# dc_w08_d4_bert.py  —  GAN-BERT assignment (MPS-safe)
# =========================================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from sklearn.metrics import roc_auc_score

# =========================================================
# Device setup
# =========================================================
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Device:", device)
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# =========================================================
# Paths
# =========================================================
TRAIN_PATH = "./data/train_essays.csv"
TEST_PATH = "./data/test_essays.csv"
PROMPT_PATH = "./data/train_prompts.csv"

tokenizer_save_path = "./models/tokenizer"
model_save_path = "./models/bert-base"

# =========================================================
# Load data
# =========================================================
src_train = pd.read_csv(TRAIN_PATH)
src_test = pd.read_csv(TEST_PATH)
src_prompt = pd.read_csv(PROMPT_PATH)

print("✅ Files loaded:")
print(f"Train: {src_train.shape}, Test: {src_test.shape}, Prompts: {src_prompt.shape}")

# Rename for clarity
src_train.rename(columns={"text": "essay_text", "generated": "label"}, inplace=True)
src_test.rename(columns={"text": "essay_text"}, inplace=True)

# =========================================================
# Tokenizer and model
# =========================================================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
pretrained_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
embedding_model = pretrained_model.bert.to(device)  # ensure model weights on MPS

os.makedirs(tokenizer_save_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
tokenizer.save_pretrained(tokenizer_save_path)
pretrained_model.save_pretrained(model_save_path)

# =========================================================
# Parameters
# =========================================================
train_batch_size = 8
test_batch_size = 16
lr = 1e-4
beta1 = 0.5
nz = 100
num_epochs = 3
num_hidden_layers = 6
train_ratio = 0.8

# =========================================================
# Dataset class
# =========================================================
class GANDAIGDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.labels is not None:
            return text, self.labels[idx]
        else:
            return text

# Split train/test
all_num = len(src_train)
train_num = int(all_num * train_ratio)
train_set = src_train.sample(train_num, random_state=42)
test_set = src_train.drop(train_set.index).reset_index(drop=True)

train_dataset = GANDAIGDataset(train_set["essay_text"].tolist(), train_set["label"].tolist())
test_dataset = GANDAIGDataset(test_set["essay_text"].tolist(), test_set["label"].tolist())

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# =========================================================
# Generator
# =========================================================
config = BertConfig(num_hidden_layers=num_hidden_layers)

class Generator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 256 * 128)
        self.conv_net = nn.Sequential(
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 768, 3, padding=1),
            nn.ReLU()
        )
        self.bert_encoder = BertEncoder(config)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 256).permute(0, 2, 1)
        x = self.conv_net(x).permute(0, 2, 1)
        outputs = self.bert_encoder(x)
        return outputs

# =========================================================
# Discriminator
# =========================================================
class SumBertPooler(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sum_hidden = hidden_states.sum(dim=1)
        sum_mask = sum_hidden.sum(1).unsqueeze(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_hidden / sum_mask

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_encoder = BertEncoder(config)
        self.bert_encoder.layer = nn.ModuleList([
            layer for layer in pretrained_model.bert.encoder.layer[:6]
        ])
        self.pooler = SumBertPooler()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input):
        out = self.bert_encoder(input)
        out = self.pooler(out.last_hidden_state)
        out = self.classifier(out)
        return torch.sigmoid(out).view(-1)

# =========================================================
# Helper functions
# =========================================================
def eval_auc(model):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in test_loader:
            enc = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            embed = embedding_model(**enc).last_hidden_state
            label = batch[1].float().to(device)
            outputs = model(embed)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(label.cpu().numpy())
    try:
        auc = roc_auc_score(actuals, predictions)
    except ValueError:
        auc = 0.5
    print("AUC:", auc)
    return auc

def preparation_embedding(texts):
    # Force all BERT embeddings to run on CPU (avoid MPS placeholder errors)
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embed = embedding_model.cpu()(**enc).last_hidden_state  # run entirely on CPU
    embedding_model.to(device)  # move back to MPS for later use
    return embed.to(device)



# =========================================================
# Training setup
# =========================================================
netG = Generator(input_dim=nz).to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# =========================================================
# Training loop
# =========================================================
def GAN_step(optimizerG, optimizerD, netG, netD, real_data, label, epoch, i):
    netD.zero_grad()
    batch_size = real_data.size(0)
    output = netD(real_data)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    noise = torch.randn(batch_size, nz, device=device)
    fake_data = netG(noise).last_hidden_state
    label.fill_(1)
    output = netD(fake_data.detach())
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    optimizerD.step()

    netG.zero_grad()
    label.fill_(0)
    output = netD(fake_data)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()

    if i % 20 == 0:
        print(f"[{epoch}/{num_epochs}][{i}/{len(train_loader)}] "
              f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
              f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")
    return optimizerG, optimizerD, netG, netD

model_infos = []
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        with torch.no_grad():
            embed = preparation_embedding(data[0])
        optimizerG, optimizerD, netG, netD = GAN_step(
            optimizerG, optimizerD, netG, netD,
            real_data=embed.to(device),
            label=data[1].float().to(device),
            epoch=epoch, i=i
        )
    auc_score = eval_auc(netD)
    model_infos.append({"epoch": epoch, "auc_score": auc_score})

print("✅ Training complete!")

# =========================================================
# Inference
# =========================================================
inference_dataset = GANDAIGDataset(src_test["essay_text"].tolist())
inference_loader = DataLoader(inference_dataset, batch_size=test_batch_size, shuffle=False)

netD.eval()
predictions = []
with torch.no_grad():
    for batch in inference_loader:
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        embed = embedding_model(**enc).last_hidden_state
        outputs = netD(embed)
        predictions.extend(outputs.cpu().numpy())

sub_df = pd.DataFrame({
    "id": src_test["id"],
    "prediction": predictions
})
os.makedirs("outputs", exist_ok=True)
sub_df.to_csv("outputs/submission.csv", index=False)
print("✅ Inference complete! Saved to outputs/submission.csv")
