# Setup / ricostruzione dati

Il repo esclude `data/` (~870MB, interamente ricostruibile):

```bash
mkdir -p data && cd data

# 1. ARC-AGI-1 (400 task train)
curl -sL https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.tar.gz | tar xz

# 2. Generatore ufficiale ARC-GEN (fonte di verità della semantica + ARC-GEN-100K)
git clone --depth 1 https://github.com/google/ARC-GEN
unzip ARC-GEN/misc/ARC-GEN-100K.zip -d arc-gen-100k

# 3. Repo di riferimento
git clone --depth 1 https://github.com/xsot/google-code-golf-2025 ref_google-code-golf-2025
git clone --depth 1 https://github.com/michaelhodel/re-arc ref_re-arc
git clone --depth 1 https://github.com/farukalamai/neurogolf-2026-toolkit ref_neurogolf-2026-toolkit
git clone --depth 1 https://github.com/Roy6250/Neurogolf ref_Neurogolf   # TECHNIQUES.md

# 4. Dati ufficiali Kaggle (serve token in ~/.kaggle/token)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/token)
kaggle competitions download -c neurogolf-2026 -p official
unzip official/neurogolf-2026.zip -d official_data

# 5. Ricostruisci i task file (o usa direttamente official_data/)
cd .. && python3 - << 'EOF'
import json, os
arc='data/ARC-AGI-master/data/training'; gen='data/arc-gen-100k'; out='data/tasks'
os.makedirs(out, exist_ok=True)
hs=sorted(f[:-5] for f in os.listdir(arc))
json.dump(hs, open('data/hash_order.json','w'))
for i,h in enumerate(hs,1):
    a=json.load(open(f'{arc}/{h}.json')); g=json.load(open(f'{gen}/{h}.json'))
    json.dump({'train':a['train'],'test':a['test'],'arc-gen':g}, open(f'{out}/task{i:03d}.json','w'))
EOF

# 6. Bundle pubblici (opzionali, per merge_bundles)
# for k in <lista in src/ng/merge_bundles.py / cronologia>: kaggle kernels output $k -p data/bundles/$(basename $k)

# 7. symlink del toolkit ai task
ln -sfn "$(pwd)/data/tasks" toolkit/neurogolf-2026
```

Dipendenze: `pip3 install numpy onnx onnxruntime onnxoptimizer onnxsim kaggle`

Workflow quotidiano: vedi AGENT_GUIDE.md. Submission:
`python3 -m ng.package` (da root con `sys.path` su `src/`) poi
`kaggle competitions submit -c neurogolf-2026 -f submission.zip -m "msg"`.
