# Fleet coordination notes

## Stato (aggiornare a ogni wave)
- Baseline toolkit (solve_all): ~14-20 task semplici (identity/flip/colorperm/conv).
- task002 risolta a mano (~10.8 pt, rischio residuo 2.7%/pair annotato nel ledger).
- Obiettivo: battere 7985.34 (rank 1). Richiede ~solve quasi tutto con avg ~20.

## Regole per gli agenti
1. Leggere AGENT_GUIDE.md PRIMA di iniziare.
2. Una task alla volta: ispeziona → leggi generatore ARC-GEN → progetta rete → check(extra=300).
3. Salvare builder in src/ng/tasks/tNNN.py (build() -> ModelProto). Il check salva in solutions/.
4. Se bloccato su una task > ~15 min di lavoro: annotare in FLEET_NOTES e passare alla successiva.
5. Aggiungere pattern riusabili scoperti in fondo a questo file (sezione Tecniche).

## Assegnazioni wave corrente
(compilato dal coordinatore)

## Tecniche scoperte dagli agenti
- (task002) Cascata raster is_surrounded = punto fisso con U/L dinamici e R/D statici;
  kernel (U+L)/2 con bias -0.5 in un solo Conv; Sign ogni 30 iter contro underflow.
- (task002) Template matching ring: angoli DON'T CARE (il generatore non li disegna).
