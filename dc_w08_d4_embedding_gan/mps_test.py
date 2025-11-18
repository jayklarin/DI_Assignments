# =========================================================
# mps_test.py — Quick MPS + BERT sanity check
# =========================================================
import torch
from transformers import BertTokenizer, BertModel

print("🔥 Starting MPS sanity check...")

# ---------------------------------------------------------
# 1️⃣  Select device (MPS if available)
# ---------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ MPS is available. Using Apple GPU.")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available. Falling back to CPU.")

# ---------------------------------------------------------
# 2️⃣  Load model + tokenizer
# ---------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

# ---------------------------------------------------------
# 3️⃣  Create sample text and tokenize
# ---------------------------------------------------------
sample_text = ["This is a quick test to verify MPS execution."]
inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# ---------------------------------------------------------
# 4️⃣  Run forward pass
# ---------------------------------------------------------
with torch.no_grad():
    outputs = model(**inputs)

print("✅ Forward pass successful.")
print("Hidden state shape:", outputs.last_hidden_state.shape)
print("Pooled output shape:", outputs.pooler_output.shape)

# ---------------------------------------------------------
# 5️⃣  Confirm GPU memory use
# ---------------------------------------------------------
if device.type == "mps":
    print("GPU tensors allocated:", torch.mps.current_allocated_memory() / 1e6, "MB")

print("🎯 MPS test complete.")
