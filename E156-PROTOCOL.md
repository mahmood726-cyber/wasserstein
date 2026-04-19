# E156 Protocol — `wasserstein`

This repository is the source code and dashboard backing an E156 micro-paper on the [E156 Student Board](https://mahmood726-cyber.github.io/e156/students.html).

---

## `[167]` Automated Kaplan-Meier Digitization for Hazard Ratio Extraction

**Type:** methods  |  ESTIMAND: HR  
**Data:** 40 trials, 11 therapeutic areas, 13 gold-standard AF trials

### 156-word body

Can automated Kaplan-Meier digitization from published PDF figures produce hazard ratios accurate enough for meta-analytic pooling? We developed a seven-stage pipeline that rasterizes PDFs, classifies figures, calibrates axes via OCR, extracts curves through HSV color segmentation, separates arms with KMeans clustering, reconstructs individual patient data using the Guyot algorithm, and estimates hazard ratios via log-rank tests. The system was validated on 40 trials spanning 11 therapeutic areas including cardiac ablation, heart failure, and oncology using published hazard ratios as reference. Across 40 trials, 36 produced HR estimates within the published 95% CI, yielding 90 percent concordance with median relative error of 2.5 percent. Cross-validation against R IPDfromKM showed the Python pipeline outperforming with median errors of 1.2 versus 10.4 percent on gold-standard atrial fibrillation trials. Automated KM digitization provides usable effect estimates when individual patient data are unavailable for time-to-event meta-analyses. The limitation of color-dependent curve separation means monochrome or stylized figures may yield unreliable results.

### Submission metadata

```
Corresponding author: Mahmood Ahmad <mahmood.ahmad2@nhs.net>
ORCID: 0000-0001-9107-3704
Affiliation: Tahir Heart Institute, Rabwah, Pakistan

Links:
  Code:      https://github.com/mahmood726-cyber/wasserstein
  Protocol:  https://github.com/mahmood726-cyber/wasserstein/blob/main/E156-PROTOCOL.md
  Dashboard: https://mahmood726-cyber.github.io/wasserstein/

References (topic pack: restricted mean survival time / survival meta-analysis):
  1. Royston P, Parmar MK. 2013. Restricted mean survival time: an alternative to the hazard ratio for the design and analysis of randomized trials with a time-to-event outcome. BMC Med Res Methodol. 13:152. doi:10.1186/1471-2288-13-152
  2. Tierney JF, Stewart LA, Ghersi D, Burdett S, Sydes MR. 2007. Practical methods for incorporating summary time-to-event data into meta-analysis. Trials. 8:16. doi:10.1186/1745-6215-8-16

Data availability: No patient-level data used. Analysis derived exclusively
  from publicly available aggregate records. All source identifiers are in
  the protocol document linked above.

Ethics: Not required. Study uses only publicly available aggregate data; no
  human participants; no patient-identifiable information; no individual-
  participant data. No institutional review board approval sought or required
  under standard research-ethics guidelines for secondary methodological
  research on published literature.

Funding: None.

Competing interests: MA serves on the editorial board of Synthēsis (the
  target journal); MA had no role in editorial decisions on this
  manuscript, which was handled by an independent editor of the journal.

Author contributions (CRediT):
  [STUDENT REWRITER, first author] — Writing – original draft, Writing –
    review & editing, Validation.
  [SUPERVISING FACULTY, last/senior author] — Supervision, Validation,
    Writing – review & editing.
  Mahmood Ahmad (middle author, NOT first or last) — Conceptualization,
    Methodology, Software, Data curation, Formal analysis, Resources.

AI disclosure: Computational tooling (including AI-assisted coding via
  Claude Code [Anthropic]) was used to develop analysis scripts and assist
  with data extraction. The final manuscript was human-written, reviewed,
  and approved by the author; the submitted text is not AI-generated. All
  quantitative claims were verified against source data; cross-validation
  was performed where applicable. The author retains full responsibility for
  the final content.

Preprint: Not preprinted.

Reporting checklist: PRISMA 2020 (methods-paper variant — reports on review corpus).

Target journal: ◆ Synthēsis (https://www.synthesis-medicine.org/index.php/journal)
  Section: Methods Note — submit the 156-word E156 body verbatim as the main text.
  The journal caps main text at ≤400 words; E156's 156-word, 7-sentence
  contract sits well inside that ceiling. Do NOT pad to 400 — the
  micro-paper length is the point of the format.

Manuscript license: CC-BY-4.0.
Code license: MIT.

SUBMITTED: [ ]
```


---

_Auto-generated from the workbook by `C:/E156/scripts/create_missing_protocols.py`. If something is wrong, edit `rewrite-workbook.txt` and re-run the script — it will overwrite this file via the GitHub API._