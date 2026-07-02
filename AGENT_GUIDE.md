# NeuroGolf 2026 — Guida operativa per la risoluzione delle task

Obiettivo: per ogni task assegnata, costruire la **rete ONNX più piccola possibile** che
implementa la trasformazione, e salvarla in `solutions/taskNNN.onnx`.

## Regole del gioco (dal giudice ufficiale)
- Rete: `input [1,10,30,30] float32` (one-hot dei colori 0-9; celle fuori griglia = tutti canali 0)
  → `output [1,10,30,30]`. Correttezza: `(output > 0)` deve eguagliare esattamente il one-hot atteso.
- **score = max(1, 25 − ln(params + memory))**
  - `params` = numero totale di elementi in initializer + tensori nei nodi Constant
  - `memory` = somma bytes dei tensori intermedi (shape reale a runtime; float32=4B/elem,
    bool=1B, float16=2B, int64=8B). I tensori chiamati esattamente `input`/`output` sono GRATIS.
  - Gli **attributi** dei nodi (perm di Transpose, pads di Conv, ecc.) sono GRATIS.
- Vietati: Loop, Scan, NonZero, Unique, Script, Function, Compress, Sequence*, attributi GRAPH
  (quindi niente If). Opset domain "" (default). Un solo input, un solo output. File ≤ 1.44MB.
- Reti a costo 0 (es. singolo Transpose/Identity) = 25 punti pieni.

## FONTE DI VERITÀ = il generatore Google ARC-GEN
La semantica esatta di ogni task è definita da `data/ARC-GEN/tasks/task_<hash>.py`
(hash in `data/hash_order.json`, task N = riga N-1). NON fidarti dell'intuizione sul
"concetto" del task: leggi il codice del generatore (e `common.py` per le utility come
`is_surrounded`, che è una scansione raster in-place, NON un flood fill!).
Riferimenti utili ma non normativi: `data/ref_re-arc/verifiers.py` (DSL leggibile),
`data/ref_google-code-golf-2025/merged/all_tasks.py` (soluzioni Python golfate).

## Workflow
```bash
cd /Users/beltromatti/Desktop/ML1
python3 -c "import sys; sys.path.insert(0,'src'); from ng.show import grids; grids(N)"   # ispeziona
# scrivi src/ng/tasks/tNNN.py con una funzione build() -> onnx.ModelProto (usa ng.builder.G)
python3 -c "
import sys; sys.path.insert(0,'src')
from ng.tasks.tNNN import build
from ng.check import check
print(check(build(), N, extra=300))   # extra = coppie fresche dal generatore (simula il privato)
"
```
`check()` salva automaticamente in `solutions/` se corretto E migliore dell'esistente.
Se `robust_bad > 0`: indaga la coppia fallita (`robust_fail_pair`), leggi il generatore,
correggi. Se l'ambiguità è irriducibile, salva comunque manualmente e annota in
`solutions/risk_ledger.json`.

## Pattern di compilazione (vedi anche data/ref_Neurogolf/TECHNIQUES.md — cookbook completo)
- Geometria (flip/rot/transpose/shift/riordino) = `Transpose` (gratis) + `Gather` con indici
  (params = len(idx)). Permutazioni separabili = 2 Gather 1-D.
- Permutazione colori = `Gather` axis=1 con idx[10] (10 params, 0 memoria!).
- Mix di canali = Conv 1x1 (100 params); depthwise (group=10) = k*k*10 params.
- Regole locali = Conv kxk + Relu/Sign; usa il **bias del Conv** per soglie (op in meno).
- Propagazione/flood = itera Conv piccolo + Mul con maschera. ATTENZIONE all'underflow:
  rinormalizza con Sign ogni ~30 iterazioni. Kernel 0.25 per evitare overflow.
- Cascata raster (is_surrounded) = punto fisso di Y ← B ∧ U(N∪Y) ∧ L(N∪Y) con R,D precalcolati.
- Output costante = fattorizza se possibile; altrimenti initializer [10,30,30] (9000 params).
- Riduzione memoria: lavora su [1,1,30,30] (3600B) o [1,2,30,30], non [1,10,30,30] (36000B);
  meno nodi = meno tensori; niente Cast inutili; il float16 dimezza ma verifica il supporto ORT.
- Template matching esatto = Conv con kernel 0/1 multi-canale + `Relu(x - (need-1))` per il match.

## Insidie note
1. Il canale 0 = colore NERO (dentro griglia), non "vuoto": una cella nera dell'output DEVE
   avere il canale 0 a 1. Fuori griglia: tutti i canali a 0.
2. Coppie con griglie > 30x30 vengono ignorate dal giudice (già filtrate da check()).
3. inf*0 = NaN: tieni i valori limitati (kernel ≤0.25, Sign periodici).
4. `>0`: valori negativi/zero = "spento". Puoi usare valori tipo 0.5 come "acceso".
5. Il generatore può avere semantica DIVERSA dal concetto ARC apparente (es. is_surrounded
   locale vs flood fill) — i 262 pair arc-gen ufficiali non sempre la rivelano: genera 300+.

## Criterio di qualità
1. Correttezza robusta (300+ coppie fresche) prima di tutto.
2. Poi minimizza cost = params + memory. Ogni ln() conta: cost 100→20.4pt, 1000→18.1,
   10000→15.8, 100000→13.5, 1M→11.2.
3. Registra il punteggio con `check()` e annota tecniche riusabili in NOTES.md della tua run.
