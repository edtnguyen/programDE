# Program-state definition from the first Hunter/Kyle table

The primary program-defined state set is based on the first table in `Program annotation notes (Kyle+Hunter).pdf`.

Primary set used by default:

- D0: P7, P22, P29 (pluripotency programs)
- D1: P28, P13, P30, P48 (primitive-streak programs)
- D2: P15, P14, P17 (vascular progenitor / lateral-mesoderm programs)
- D3: P33, P34, P2, P3, P8, P11 (anterior-like mesoderm, mesenchymal/YAP-TAZ, arterial/endothelial programs)

Programs available only in the expanded sensitivity set:

- P23: vascular lateral mesoderm; the source table recommends removal from Fig. 2G because selected knockdowns weakly affect it.
- P19: unknown mesoderm; the source table suggests possible removal.
- P27 and P24: identities remain unresolved in the source notes.

Programs marked to eliminate in the source table and excluded from both sets:

- P6, P21, P35, P42

The primary assignment rule does not force a cell into one of these programs. The global argmax is computed over all 50 cNMF programs, and a cell is retained only when that global argmax is in the primary curated set and its expected day matches the observed collection day.
