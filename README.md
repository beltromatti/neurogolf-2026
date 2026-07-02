# NeuroGolf 2026 — Kaggle Competition

Reti ONNX minime per le 400 task ARC-AGI-1. Score per task: `max(1, 25 − ln(params + memoria_intermedi_bytes))`, correttezza esatta `(out>0)` su train+test+arc-gen+benchmark privato. Deadline: 15 luglio 2026.

**Stato: LB 7074.09 (prima submission) → ~7089 attesi con merge2. Rank 1 = 7990.**

## Struttura
- `src/ng/` — framework: `builder.py` (DSL ONNX), `check.py` (validatore fedele al giudice + robustezza su coppie fresche dal generatore ufficiale), `show.py` (ispettore task), `merge_bundles.py`, `lossless_pass.py`, `validate_builders.py`, `package.py` (zip+submit), `tasks/tNNN.py` (builder per task)
- `solutions/` — le migliori reti correnti (`taskNNN.onnx`) + report (merge, lossless, rischi)
- `toolkit/` — fork del toolkit pubblico (validatore/triage/solver di base, Apache-2.0)
- `AGENT_GUIDE.md` — istruzioni operative complete (regole, pattern, insidie)
- `FLEET_NOTES.md` — coordinamento e tecniche scoperte
- `SETUP.md` — ricostruzione di `data/` (esclusa dal repo)

## Principi chiave scoperti
1. **La fonte di verità è il generatore Google ARC-GEN** (`data/ARC-GEN/tasks/task_<hash>.py`), non il "concetto" del task ARC.
2. **Robustezza**: validare su 300+ coppie fresche generate = proxy del benchmark privato (validatore calibrato: LB reale = stima locale al decimale).
3. **Costo**: attributi dei nodi gratis; tensori `input`/`output` gratis; dominano i tensori intermedi (bool=1B, fp16=2B — Conv fp16 funziona su ORT CPU).

## Prossimi passi
- Golf sistematico delle 248 task in fascia 16–18 pt (+860 punti potenziali, la strada per il rank 1)
- Risolvere le 6 task irrisolte: 2*, 18, 23, 187, 219, 255 (*2 = risolta con ambiguità nota, già conta ~10.8 su LB)
