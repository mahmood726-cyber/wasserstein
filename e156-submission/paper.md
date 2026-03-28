Mahmood Ahmad
Tahir Heart Institute
author@example.com

Automated Kaplan-Meier Digitization for Hazard Ratio Extraction

Can automated Kaplan-Meier digitization from published PDF figures produce hazard ratios accurate enough for meta-analytic pooling? We developed a seven-stage pipeline that rasterizes PDFs, classifies figures, calibrates axes via OCR, extracts curves through HSV color segmentation, separates arms with KMeans clustering, reconstructs individual patient data using the Guyot algorithm, and estimates hazard ratios via log-rank tests. The system was validated on 40 trials spanning 11 therapeutic areas including cardiac ablation, heart failure, and oncology using published hazard ratios as reference. Across 40 trials, 36 produced HR estimates within the published 95% CI, yielding 90 percent concordance with median relative error of 2.5 percent. Cross-validation against R IPDfromKM showed the Python pipeline outperforming with median errors of 1.2 versus 10.4 percent on gold-standard atrial fibrillation trials. Automated KM digitization provides usable effect estimates when individual patient data are unavailable for time-to-event meta-analyses. The limitation of color-dependent curve separation means monochrome or stylized figures may yield unreliable results.

Outside Notes

Type: methods
Primary estimand: HR
App: Wasserstein KM Extractor v1.4
Data: 40 trials, 11 therapeutic areas, 13 gold-standard AF trials
Code: https://github.com/mahmood726-cyber/wasserstein
Version: 1.4
Validation: DRAFT

References

1. Guyot P, Ades AE, Ouwens MJ, Welton NJ. Enhanced secondary analysis of survival data: reconstructing the data from published Kaplan-Meier survival curves. BMC Med Res Methodol. 2012;12:9.
2. Tierney JF, Stewart LA, Ghersi D, Burdett S, Sydes MR. Practical methods for incorporating summary time-to-event data into meta-analysis. Trials. 2007;8:16.
3. Borenstein M, Hedges LV, Higgins JPT, Rothstein HR. Introduction to Meta-Analysis. 2nd ed. Wiley; 2021.

AI Disclosure

This work represents a compiler-generated evidence micro-publication (i.e., a structured, pipeline-based synthesis output). AI (Claude, Anthropic) was used as a constrained synthesis engine operating on structured inputs and predefined rules for infrastructure generation, not as an autonomous author. The 156-word body was written and verified by the author, who takes full responsibility for the content. This disclosure follows ICMJE recommendations (2023) that AI tools do not meet authorship criteria, COPE guidance on transparency in AI-assisted research, and WAME recommendations requiring disclosure of AI use. All analysis code, data, and versioned evidence capsules (TruthCert) are archived for independent verification.
