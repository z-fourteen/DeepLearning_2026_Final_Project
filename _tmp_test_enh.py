import torch
from src.models.transformer_enhanced import EnhancedTransformerModel
from src.models.transformer import TransformerStockModel

# Test all pooling modes
print("=== Pooling test ===")
for p in ['cls', 'last_step', 'attention', 'mean', 'max']:
    m = EnhancedTransformerModel(13, {'pooling': p})
    out = m(torch.randn(2, 10, 13))
    print(f'  pooling={p:10s}  OK  shape={tuple(out.shape)}')

# Test different num_features
print("\n=== Feature dim test ===")
for nf in [13, 18, 62]:
    m = EnhancedTransformerModel(nf, {'pooling': 'cls'})
    print(f'  num_features={nf:2d}  OK')

# Parameter count comparison
base = TransformerStockModel(13, {'pooling': 'cls'})                          # L=2, no FFN block
enh4 = EnhancedTransformerModel(13, {'pooling': 'cls'})                       # L=4 (default), deep_head=True
enh6 = EnhancedTransformerModel(13, {'pooling': 'cls', 'num_encoder_layers': 6}) # L=6

bp = sum(p.numel() for p in base.parameters())
ep4 = sum(p.numel() for p in enh4.parameters())
ep6 = sum(p.numel() for p in enh6.parameters())

print("\n=== Parameter count ===")
print(f"Baseline (L=2):          {bp:>8,}")
print(f"Enhanced (L=4, default): {ep4:>8,} (+{ep4 - bp:+,})")
print(f"Enhanced (L=6):          {ep6:>8,} (+{ep6 - bp:+,})")

# Verify no IntermediateFFN references remain
assert not hasattr(enh4, 'intermediate_ffn'), "FFN should be removed"
assert not hasattr(enh4, 'use_ffn'), "use_ffn should be removed"
print("\n✓ FFN removed, encoder depth increased")
