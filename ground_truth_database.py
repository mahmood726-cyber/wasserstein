"""
Ground Truth Database
====================

Structured storage for real PDF validation with SHA-256 hashes for TruthCert compliance.

Features:
- SHA-256 content hashes for provenance tracking
- Verification status tracking
- Query and filtering capabilities
- Export to multiple formats (JSON, CSV)
- Evidence locator compliance for TruthCert bundles

Schema:
- GroundTruthEntry: Individual HR ground truth with evidence locators
- ValidationResult: Extraction vs ground truth comparison
- Database: Collection management with integrity checking

Author: Wasserstein KM Extractor Team
Date: February 2026
Version: 1.0
"""

import hashlib
import json
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Iterator, Any
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of ground truth verification."""
    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    RETRACTED = "retracted"


class YAxisScale(Enum):
    """Y-axis scale type for KM curves."""
    ZERO_TO_ONE = "0-1"
    ZERO_TO_HUNDRED = "0-100"
    UNKNOWN = "unknown"


class SourceType(Enum):
    """Source type for ground truth extraction."""
    ABSTRACT = "abstract"
    RESULTS_TEXT = "results_text"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    SUPPLEMENTARY = "supplementary"
    MANUAL_EXTRACTION = "manual_extraction"
    EXTERNAL_DATABASE = "external_database"


@dataclass
class NARTimepoint:
    """Number-at-risk value at a specific timepoint."""
    timepoint: float  # Time value (e.g., 0, 12, 24 months)
    value: int  # Number of patients at risk


@dataclass
class ArmInfo:
    """Information about a treatment arm in the KM curve."""
    label: str  # e.g., "pembrolizumab", "chemotherapy"
    color: str  # e.g., "blue", "red", "#FF0000"
    is_treatment: bool = False  # True for treatment arm, False for control
    nar_values: List[NARTimepoint] = field(default_factory=list)


@dataclass
class RCTGroundTruth:
    """
    Ground truth entry for RCT PDF validation with extended metadata.

    This schema supports comprehensive validation of KM curve extraction
    including axis calibration, arm identification, and NAR table validation.
    """
    # Primary identification
    paper_id: str
    pdf_path: str
    doi: str = ""

    # From paper text - reported HR values
    hr_reported: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    p_value: Optional[float] = None

    # Axis calibration - critical for accurate extraction
    x_axis_min: float = 0.0
    x_axis_max_months: float = 0.0  # Maximum time in months
    x_axis_unit: str = "months"  # "months", "years", "days", "weeks"
    y_axis_scale: str = "0-1"  # "0-1" or "0-100"
    y_axis_min: float = 0.0
    y_axis_max: float = 1.0

    # Arm identification - for correct treatment/control assignment
    arm1: Optional[ArmInfo] = None
    arm2: Optional[ArmInfo] = None
    n_arms: int = 2

    # NAR table information
    has_nar_table: bool = False
    nar_timepoints: List[float] = field(default_factory=list)  # e.g., [0, 12, 24, 36]

    # Figure metadata
    figure_page: int = 0
    figure_panel: str = ""  # "A", "B", "main", etc.
    figure_caption: str = ""

    # Outcome information
    therapeutic_area: str = ""
    outcome_type: str = ""  # "OS", "PFS", "DFS", "EFS"
    median_followup_months: Optional[float] = None

    # Sample sizes
    n_total: int = 0
    n_arm1: int = 0
    n_arm2: int = 0
    n_events_total: int = 0
    n_events_arm1: int = 0
    n_events_arm2: int = 0

    # Curve characteristics
    has_crossing_curves: bool = False
    has_confidence_bands: bool = False
    has_censoring_marks: bool = False

    # Verification
    verified: bool = False
    verified_by: str = ""
    verification_date: str = ""
    verification_notes: str = ""

    # Provenance
    created_date: str = ""
    modified_date: str = ""
    source_url: str = ""

    def get_x_axis_info(self) -> Dict[str, Any]:
        """Get x-axis calibration info as dict."""
        return {
            'min': self.x_axis_min,
            'max': self.x_axis_max_months,
            'unit': self.x_axis_unit
        }

    def get_y_axis_info(self) -> Dict[str, Any]:
        """Get y-axis calibration info as dict."""
        return {
            'min': self.y_axis_min,
            'max': self.y_axis_max,
            'scale': self.y_axis_scale
        }

    def get_nar_for_arm(self, arm_index: int) -> List[int]:
        """Get NAR values for specified arm (0 or 1)."""
        arm = self.arm1 if arm_index == 0 else self.arm2
        if arm and arm.nar_values:
            return [nar.value for nar in arm.nar_values]
        return []


@dataclass
class EvidenceLocator:
    """
    Evidence locator for TruthCert compliance.

    Every certified claim must reference its evidence source.
    """
    # Primary location
    url: str = ""
    file_path: str = ""
    page_number: Optional[int] = None

    # Content verification
    content_hash: str = ""  # SHA-256 of source content
    extraction_date: str = ""

    # Specificity
    section: str = ""  # e.g., "Results", "Figure 2 caption"
    text_excerpt: str = ""  # Exact text containing the claim (max 500 chars)

    def is_valid(self) -> bool:
        """Check if locator has minimum required fields."""
        return bool(self.url or self.file_path) and bool(self.content_hash)


@dataclass
class GroundTruthEntry:
    """
    Ground truth entry for a single HR comparison.

    TruthCert compliant:
    - Evidence locator (URL/path + hash)
    - Verification status
    - Provenance tracking
    """
    # Identification
    entry_id: str
    pdf_id: str

    # Core HR data
    hr: float
    ci_lower: float
    ci_upper: float
    p_value: Optional[float] = None

    # Comparison metadata
    comparison_label: str = ""  # e.g., "treatment vs control"
    arm1_label: str = ""  # e.g., "pembrolizumab"
    arm2_label: str = ""  # e.g., "chemotherapy"

    # Source tracking
    source_type: str = "abstract"
    source_location: str = ""  # Specific page/section
    evidence_locator: Optional[EvidenceLocator] = None

    # Verification
    verification_status: str = "unverified"
    verified_by: str = ""
    verification_date: str = ""
    verification_notes: str = ""

    # Additional context
    outcome_type: str = ""  # "OS", "PFS", "DFS", etc.
    median_followup: Optional[float] = None  # months
    n_events_arm1: Optional[int] = None
    n_events_arm2: Optional[int] = None

    # Provenance
    created_date: str = ""
    modified_date: str = ""


@dataclass
class PDFMetadata:
    """Complete metadata for a PDF in the database."""
    pdf_id: str
    pdf_hash: str  # SHA-256 of PDF file

    # Bibliographic
    title: str = ""
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    year: int = 0
    doi: str = ""
    pmcid: str = ""
    pmid: str = ""

    # Classification
    therapeutic_area: str = ""
    journal_style: str = ""

    # Figure info
    n_km_figures: int = 0
    figure_pages: List[int] = field(default_factory=list)
    has_nar_table: bool = False
    n_arms: int = 2
    has_crossing_curves: bool = False

    # File info
    local_path: str = ""
    file_size_bytes: int = 0
    source_url: str = ""

    # Provenance
    download_date: str = ""
    added_to_db_date: str = ""


@dataclass
class ValidationResult:
    """
    Result of validating extraction against ground truth.

    Used for tracking extraction accuracy on real PDFs.
    """
    entry_id: str
    pdf_id: str

    # Ground truth
    hr_true: float
    ci_lower_true: float
    ci_upper_true: float

    # Extracted
    hr_extracted: Optional[float]
    ci_lower_extracted: Optional[float]
    ci_upper_extracted: Optional[float]

    # Metrics
    extraction_success: bool
    absolute_error: Optional[float] = None
    relative_error_pct: Optional[float] = None
    hr_within_ci: bool = False  # Is true HR within extracted CI?
    ci_overlap: bool = False  # Do CIs overlap?

    # Quality
    extraction_confidence: float = 0.0
    quality_grade: str = ""
    extraction_method: str = ""

    # Timing
    extraction_time: float = 0.0
    validation_date: str = ""

    # Notes
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class GroundTruthDatabase:
    """
    Database for managing ground truth entries with TruthCert compliance.

    Features:
    - SHA-256 hashes for all content
    - Evidence locators for provenance
    - Verification workflow
    - Query and filter capabilities
    - Export to JSON/CSV
    """

    def __init__(self, db_path: str = "real_validation_datasets"):
        """
        Initialize the ground truth database.

        Parameters
        ----------
        db_path : str
            Path to database directory
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Data storage files
        self.entries_file = self.db_path / "ground_truth_entries.json"
        self.pdfs_file = self.db_path / "pdf_metadata.json"
        self.validations_file = self.db_path / "validation_results.json"

        # In-memory storage
        self.entries: Dict[str, GroundTruthEntry] = {}
        self.pdfs: Dict[str, PDFMetadata] = {}
        self.validations: Dict[str, ValidationResult] = {}

        # Database hash for integrity
        self._db_hash: str = ""

        # Load existing data
        self._load_data()

    def _load_data(self):
        """Load all data from disk."""
        self._load_entries()
        self._load_pdfs()
        self._load_validations()
        self._update_db_hash()

    def _load_entries(self):
        """Load ground truth entries."""
        if self.entries_file.exists():
            try:
                with open(self.entries_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for entry_id, entry_data in data.items():
                    # Convert evidence locator
                    if entry_data.get('evidence_locator'):
                        entry_data['evidence_locator'] = EvidenceLocator(**entry_data['evidence_locator'])
                    self.entries[entry_id] = GroundTruthEntry(**entry_data)
                logger.info(f"Loaded {len(self.entries)} ground truth entries")
            except Exception as e:
                logger.warning(f"Could not load entries: {e}")

    def _load_pdfs(self):
        """Load PDF metadata."""
        if self.pdfs_file.exists():
            try:
                with open(self.pdfs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for pdf_id, pdf_data in data.items():
                    self.pdfs[pdf_id] = PDFMetadata(**pdf_data)
                logger.info(f"Loaded {len(self.pdfs)} PDF metadata entries")
            except Exception as e:
                logger.warning(f"Could not load PDF metadata: {e}")

    def _load_validations(self):
        """Load validation results."""
        if self.validations_file.exists():
            try:
                with open(self.validations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for val_id, val_data in data.items():
                    self.validations[val_id] = ValidationResult(**val_data)
                logger.info(f"Loaded {len(self.validations)} validation results")
            except Exception as e:
                logger.warning(f"Could not load validations: {e}")

    def _save_entries(self):
        """Save ground truth entries to disk."""
        data = {}
        for entry_id, entry in self.entries.items():
            entry_dict = asdict(entry)
            # Convert evidence locator
            if entry.evidence_locator:
                entry_dict['evidence_locator'] = asdict(entry.evidence_locator)
            data[entry_id] = entry_dict

        with open(self.entries_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_pdfs(self):
        """Save PDF metadata to disk."""
        data = {pdf_id: asdict(pdf) for pdf_id, pdf in self.pdfs.items()}
        with open(self.pdfs_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_validations(self):
        """Save validation results to disk."""
        data = {val_id: asdict(val) for val_id, val in self.validations.items()}
        with open(self.validations_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_all(self):
        """Save all data to disk."""
        self._save_entries()
        self._save_pdfs()
        self._save_validations()
        self._update_db_hash()
        logger.info("Database saved")

    def _update_db_hash(self):
        """Update database integrity hash."""
        hasher = hashlib.sha256()

        # Hash entries
        for entry_id in sorted(self.entries.keys()):
            hasher.update(entry_id.encode())
            hasher.update(str(self.entries[entry_id].hr).encode())

        # Hash PDFs
        for pdf_id in sorted(self.pdfs.keys()):
            hasher.update(pdf_id.encode())
            hasher.update(self.pdfs[pdf_id].pdf_hash.encode())

        # Hash validations
        for val_id in sorted(self.validations.keys()):
            val = self.validations[val_id]
            hasher.update(val_id.encode())
            hasher.update(str(val.hr_true).encode())
            hasher.update(str(val.hr_extracted).encode())
            hasher.update(str(val.extraction_success).encode())

        # Hash RCT ground-truth payload (stored in a separate file)
        rct_file = self.db_path / "rct_ground_truth.json"
        if rct_file.exists():
            try:
                hasher.update(rct_file.read_bytes())
            except Exception as e:
                logger.debug(f"Could not hash RCT ground truth file: {e}")

        self._db_hash = hasher.hexdigest()[:16]

    def get_db_hash(self) -> str:
        """Get current database integrity hash."""
        return self._db_hash

    @staticmethod
    def compute_content_hash(content: bytes) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def add_entry(self, entry: GroundTruthEntry) -> str:
        """
        Add a ground truth entry.

        Parameters
        ----------
        entry : GroundTruthEntry
            Entry to add

        Returns
        -------
        str
            Entry ID
        """
        if not entry.entry_id:
            entry.entry_id = f"{entry.pdf_id}_hr_{len(self.entries) + 1}"

        if not entry.created_date:
            entry.created_date = datetime.now().isoformat()

        self.entries[entry.entry_id] = entry
        self._save_entries()
        self._update_db_hash()
        return entry.entry_id

    def add_pdf(self, pdf: PDFMetadata) -> str:
        """
        Add PDF metadata.

        Parameters
        ----------
        pdf : PDFMetadata
            PDF metadata to add

        Returns
        -------
        str
            PDF ID
        """
        if not pdf.added_to_db_date:
            pdf.added_to_db_date = datetime.now().isoformat()

        self.pdfs[pdf.pdf_id] = pdf
        self._save_pdfs()
        self._update_db_hash()
        return pdf.pdf_id

    def add_validation(self, validation: ValidationResult) -> str:
        """
        Add a validation result.

        Parameters
        ----------
        validation : ValidationResult
            Validation result to add

        Returns
        -------
        str
            Validation ID
        """
        val_id = f"{validation.pdf_id}_{validation.entry_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        if not validation.validation_date:
            validation.validation_date = datetime.now().isoformat()

        # Compute metrics if not provided
        if validation.hr_extracted is not None and validation.absolute_error is None:
            validation.absolute_error = abs(validation.hr_extracted - validation.hr_true)
            if validation.hr_true > 0:
                validation.relative_error_pct = validation.absolute_error / validation.hr_true * 100

        self.validations[val_id] = validation
        self._save_validations()
        self._update_db_hash()
        return val_id

    def get_entry(self, entry_id: str) -> Optional[GroundTruthEntry]:
        """Get a ground truth entry by ID."""
        return self.entries.get(entry_id)

    def get_pdf(self, pdf_id: str) -> Optional[PDFMetadata]:
        """Get PDF metadata by ID."""
        return self.pdfs.get(pdf_id)

    def get_entries_for_pdf(self, pdf_id: str) -> List[GroundTruthEntry]:
        """Get all ground truth entries for a PDF."""
        return [e for e in self.entries.values() if e.pdf_id == pdf_id]

    def get_validations_for_pdf(self, pdf_id: str) -> List[ValidationResult]:
        """Get all validation results for a PDF."""
        return [v for v in self.validations.values() if v.pdf_id == pdf_id]

    # =========================================================================
    # Query Methods
    # =========================================================================

    def query_entries(self,
                      therapeutic_area: Optional[str] = None,
                      journal_style: Optional[str] = None,
                      verification_status: Optional[str] = None,
                      hr_range: Optional[Tuple[float, float]] = None,
                      has_nar: Optional[bool] = None,
                      min_arms: Optional[int] = None) -> List[GroundTruthEntry]:
        """
        Query ground truth entries with filters.

        Parameters
        ----------
        therapeutic_area : str, optional
            Filter by area (oncology, cardiovascular, etc.)
        journal_style : str, optional
            Filter by journal style (nejm, lancet, jco, jama)
        verification_status : str, optional
            Filter by verification status
        hr_range : tuple, optional
            Filter by HR range (min, max)
        has_nar : bool, optional
            Filter for entries with NAR tables
        min_arms : int, optional
            Filter for multi-arm trials with at least N arms

        Returns
        -------
        List of matching entries
        """
        results = []

        for entry in self.entries.values():
            # Get associated PDF metadata
            pdf = self.pdfs.get(entry.pdf_id)

            # Apply filters
            if therapeutic_area and pdf and pdf.therapeutic_area != therapeutic_area:
                continue

            if journal_style and pdf and pdf.journal_style != journal_style:
                continue

            if verification_status and entry.verification_status != verification_status:
                continue

            if hr_range:
                if not (hr_range[0] <= entry.hr <= hr_range[1]):
                    continue

            if has_nar is not None and pdf:
                if pdf.has_nar_table != has_nar:
                    continue

            if min_arms is not None and pdf:
                if pdf.n_arms < min_arms:
                    continue

            results.append(entry)

        return results

    def query_pdfs(self,
                   therapeutic_area: Optional[str] = None,
                   journal_style: Optional[str] = None,
                   has_ground_truth: bool = True,
                   has_local_pdf: bool = False,
                   year_range: Optional[Tuple[int, int]] = None) -> List[PDFMetadata]:
        """
        Query PDF metadata with filters.

        Parameters
        ----------
        therapeutic_area : str, optional
            Filter by area
        journal_style : str, optional
            Filter by journal style
        has_ground_truth : bool
            Only return PDFs with ground truth entries
        has_local_pdf : bool
            Only return PDFs with local files
        year_range : tuple, optional
            Filter by year range

        Returns
        -------
        List of matching PDFs
        """
        results = []

        for pdf in self.pdfs.values():
            if therapeutic_area and pdf.therapeutic_area != therapeutic_area:
                continue

            if journal_style and pdf.journal_style != journal_style:
                continue

            if has_ground_truth:
                entries = self.get_entries_for_pdf(pdf.pdf_id)
                if not entries:
                    continue

            if has_local_pdf and not pdf.local_path:
                continue

            if year_range:
                if not (year_range[0] <= pdf.year <= year_range[1]):
                    continue

            results.append(pdf)

        return results

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for validation results.

        Returns
        -------
        Dict with summary statistics
        """
        if not self.validations:
            return {'n_validations': 0}

        successful = [v for v in self.validations.values() if v.extraction_success]
        failed = [v for v in self.validations.values() if not v.extraction_success]

        abs_errors = [v.absolute_error for v in successful if v.absolute_error is not None]
        rel_errors = [v.relative_error_pct for v in successful if v.relative_error_pct is not None]

        summary = {
            'n_validations': len(self.validations),
            'n_successful': len(successful),
            'n_failed': len(failed),
            'success_rate': len(successful) / len(self.validations) * 100 if self.validations else 0,
        }

        if abs_errors:
            import statistics
            summary['hr_mae'] = statistics.mean(abs_errors)
            summary['hr_rmse'] = (sum(e**2 for e in abs_errors) / len(abs_errors)) ** 0.5
            summary['hr_within_20pct'] = sum(1 for r in rel_errors if r <= 20) / len(rel_errors) * 100

        within_ci = sum(1 for v in successful if v.hr_within_ci)
        summary['hr_within_ci_rate'] = within_ci / len(successful) * 100 if successful else 0

        return summary

    # =========================================================================
    # Verification Workflow
    # =========================================================================

    def verify_entry(self,
                     entry_id: str,
                     verified_by: str,
                     status: str = "verified",
                     notes: str = "") -> bool:
        """
        Mark an entry as verified.

        Parameters
        ----------
        entry_id : str
            Entry to verify
        verified_by : str
            Verifier identifier
        status : str
            New verification status
        notes : str
            Verification notes

        Returns
        -------
        bool
            True if successful
        """
        if entry_id not in self.entries:
            return False

        entry = self.entries[entry_id]
        entry.verification_status = status
        entry.verified_by = verified_by
        entry.verification_date = datetime.now().isoformat()
        entry.verification_notes = notes
        entry.modified_date = datetime.now().isoformat()

        self._save_entries()
        self._update_db_hash()
        return True

    def get_unverified_entries(self) -> List[GroundTruthEntry]:
        """Get all unverified entries."""
        return [e for e in self.entries.values() if e.verification_status == "unverified"]

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_ground_truth_json(self, output_path: Optional[str] = None) -> str:
        """
        Export ground truth in standard JSON format.

        Parameters
        ----------
        output_path : str, optional
            Output file path

        Returns
        -------
        str
            Path to output file
        """
        if output_path is None:
            output_path = str(self.db_path / "ground_truth.json")

        data = {}
        for entry_id, entry in self.entries.items():
            pdf = self.pdfs.get(entry.pdf_id, PDFMetadata(pdf_id=entry.pdf_id, pdf_hash=""))

            data[entry_id] = {
                'pdf_id': entry.pdf_id,
                'pdf_hash': pdf.pdf_hash,
                'source_url': pdf.source_url,
                'hr': entry.hr,
                'ci_lower': entry.ci_lower,
                'ci_upper': entry.ci_upper,
                'source_location': entry.source_location,
                'page_number': pdf.figure_pages[0] if pdf.figure_pages else None,
                'verified': entry.verification_status == "verified"
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        return output_path

    def export_validation_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export validation results to CSV.

        Parameters
        ----------
        output_path : str, optional
            Output file path

        Returns
        -------
        str
            Path to output file
        """
        if output_path is None:
            output_path = str(self.db_path / "validation_results.csv")

        fieldnames = [
            'pdf_id', 'entry_id',
            'hr_true', 'ci_lower_true', 'ci_upper_true',
            'hr_extracted', 'ci_lower_extracted', 'ci_upper_extracted',
            'extraction_success', 'absolute_error', 'relative_error_pct',
            'hr_within_ci', 'quality_grade', 'extraction_method'
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for val in self.validations.values():
                row = {
                    'pdf_id': val.pdf_id,
                    'entry_id': val.entry_id,
                    'hr_true': val.hr_true,
                    'ci_lower_true': val.ci_lower_true,
                    'ci_upper_true': val.ci_upper_true,
                    'hr_extracted': val.hr_extracted,
                    'ci_lower_extracted': val.ci_lower_extracted,
                    'ci_upper_extracted': val.ci_upper_extracted,
                    'extraction_success': val.extraction_success,
                    'absolute_error': val.absolute_error,
                    'relative_error_pct': val.relative_error_pct,
                    'hr_within_ci': val.hr_within_ci,
                    'quality_grade': val.quality_grade,
                    'extraction_method': val.extraction_method
                }
                writer.writerow(row)

        return output_path

    def export_for_truthcert(self, output_path: Optional[str] = None) -> str:
        """
        Export data in TruthCert-compliant format.

        Each entry includes:
        - Evidence locators
        - Content hashes
        - Verification status

        Parameters
        ----------
        output_path : str, optional
            Output file path

        Returns
        -------
        str
            Path to output file
        """
        if output_path is None:
            output_path = str(self.db_path / "truthcert_bundle.json")

        bundle = {
            'version': '1.0',
            'generated': datetime.now().isoformat(),
            'db_hash': self._db_hash,
            'claims': []
        }

        for entry in self.entries.values():
            pdf = self.pdfs.get(entry.pdf_id)

            claim = {
                'claim_id': entry.entry_id,
                'claim_type': 'hazard_ratio',
                'value': {
                    'hr': entry.hr,
                    'ci_lower': entry.ci_lower,
                    'ci_upper': entry.ci_upper
                },
                'evidence': {
                    'locator': {
                        'url': pdf.source_url if pdf else '',
                        'file_hash': pdf.pdf_hash if pdf else '',
                        'page': pdf.figure_pages[0] if pdf and pdf.figure_pages else None,
                        'section': entry.source_location
                    },
                    'text_excerpt': entry.evidence_locator.text_excerpt if entry.evidence_locator else ''
                },
                'verification': {
                    'status': entry.verification_status,
                    'verified_by': entry.verified_by,
                    'date': entry.verification_date
                },
                'provenance': {
                    'created': entry.created_date,
                    'modified': entry.modified_date
                }
            }
            bundle['claims'].append(claim)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(bundle, f, indent=2)

        return output_path

    # =========================================================================
    # RCT Ground Truth Management
    # =========================================================================

    def add_rct_ground_truth(self, rct: 'RCTGroundTruth') -> str:
        """
        Add an RCT ground truth entry to the database.

        Parameters
        ----------
        rct : RCTGroundTruth
            RCT ground truth to add

        Returns
        -------
        str
            Paper ID
        """
        if not rct.created_date:
            rct.created_date = datetime.now().isoformat()

        # Store as JSON in a separate file
        rct_file = self.db_path / "rct_ground_truth.json"

        # Load existing RCT data
        rct_data = {}
        if rct_file.exists():
            try:
                with open(rct_file, 'r', encoding='utf-8') as f:
                    rct_data = json.load(f)
            except Exception:
                pass

        # Convert to dict for storage
        rct_dict = self._rct_to_dict(rct)
        rct_data[rct.paper_id] = rct_dict

        # Save
        with open(rct_file, 'w', encoding='utf-8') as f:
            json.dump(rct_data, f, indent=2, ensure_ascii=False)
        self._update_db_hash()

        logger.info(f"Added RCT ground truth: {rct.paper_id}")
        return rct.paper_id

    def get_rct_ground_truth(self, paper_id: str) -> Optional['RCTGroundTruth']:
        """
        Get an RCT ground truth entry by paper ID.

        Parameters
        ----------
        paper_id : str
            Paper ID to retrieve

        Returns
        -------
        RCTGroundTruth or None
        """
        rct_file = self.db_path / "rct_ground_truth.json"
        if not rct_file.exists():
            return None

        try:
            with open(rct_file, 'r', encoding='utf-8') as f:
                rct_data = json.load(f)

            if paper_id in rct_data:
                return self._dict_to_rct(rct_data[paper_id])
        except Exception as e:
            logger.warning(f"Error loading RCT ground truth: {e}")

        return None

    def get_all_rct_ground_truths(self) -> List['RCTGroundTruth']:
        """
        Get all RCT ground truth entries.

        Returns
        -------
        List of RCTGroundTruth
        """
        rct_file = self.db_path / "rct_ground_truth.json"
        if not rct_file.exists():
            return []

        try:
            with open(rct_file, 'r', encoding='utf-8') as f:
                rct_data = json.load(f)

            return [self._dict_to_rct(data) for data in rct_data.values()]
        except Exception as e:
            logger.warning(f"Error loading RCT ground truths: {e}")
            return []

    def query_rct_ground_truths(self,
                                 therapeutic_area: Optional[str] = None,
                                 outcome_type: Optional[str] = None,
                                 has_nar: Optional[bool] = None,
                                 verified_only: bool = False,
                                 hr_range: Optional[Tuple[float, float]] = None) -> List['RCTGroundTruth']:
        """
        Query RCT ground truths with filters.

        Parameters
        ----------
        therapeutic_area : str, optional
            Filter by therapeutic area
        outcome_type : str, optional
            Filter by outcome (OS, PFS, etc.)
        has_nar : bool, optional
            Filter by NAR table presence
        verified_only : bool
            Only return verified entries
        hr_range : tuple, optional
            Filter by HR range (min, max)

        Returns
        -------
        List of matching RCTGroundTruth entries
        """
        all_rcts = self.get_all_rct_ground_truths()
        results = []

        for rct in all_rcts:
            # Apply filters
            if verified_only and not rct.verified:
                continue

            if (therapeutic_area and
                    rct.therapeutic_area.lower() != therapeutic_area.lower()):
                continue

            if has_nar is not None and rct.has_nar_table != has_nar:
                continue

            if outcome_type and rct.outcome_type.upper() != outcome_type.upper():
                continue

            if hr_range:
                if not (hr_range[0] <= rct.hr_reported <= hr_range[1]):
                    continue

            results.append(rct)

        return results

    def _rct_to_dict(self, rct: 'RCTGroundTruth') -> Dict:
        """Convert RCTGroundTruth to dict for JSON storage."""
        data = {
            'paper_id': rct.paper_id,
            'pdf_path': rct.pdf_path,
            'doi': rct.doi,
            'hr_reported': rct.hr_reported,
            'ci_lower': rct.ci_lower,
            'ci_upper': rct.ci_upper,
            'p_value': rct.p_value,
            'x_axis_min': rct.x_axis_min,
            'x_axis_max_months': rct.x_axis_max_months,
            'x_axis_unit': rct.x_axis_unit,
            'y_axis_scale': rct.y_axis_scale,
            'y_axis_min': rct.y_axis_min,
            'y_axis_max': rct.y_axis_max,
            'n_arms': rct.n_arms,
            'has_nar_table': rct.has_nar_table,
            'nar_timepoints': rct.nar_timepoints,
            'figure_page': rct.figure_page,
            'figure_panel': rct.figure_panel,
            'figure_caption': rct.figure_caption,
            'therapeutic_area': rct.therapeutic_area,
            'outcome_type': rct.outcome_type,
            'median_followup_months': rct.median_followup_months,
            'n_total': rct.n_total,
            'n_arm1': rct.n_arm1,
            'n_arm2': rct.n_arm2,
            'n_events_total': rct.n_events_total,
            'n_events_arm1': rct.n_events_arm1,
            'n_events_arm2': rct.n_events_arm2,
            'has_crossing_curves': rct.has_crossing_curves,
            'has_confidence_bands': rct.has_confidence_bands,
            'has_censoring_marks': rct.has_censoring_marks,
            'verified': rct.verified,
            'verified_by': rct.verified_by,
            'verification_date': rct.verification_date,
            'verification_notes': rct.verification_notes,
            'created_date': rct.created_date,
            'modified_date': rct.modified_date,
            'source_url': rct.source_url,
        }

        # Convert arm info
        if rct.arm1:
            data['arm1'] = {
                'label': rct.arm1.label,
                'color': rct.arm1.color,
                'is_treatment': rct.arm1.is_treatment,
                'nar_values': [{'timepoint': n.timepoint, 'value': n.value}
                              for n in rct.arm1.nar_values]
            }
        if rct.arm2:
            data['arm2'] = {
                'label': rct.arm2.label,
                'color': rct.arm2.color,
                'is_treatment': rct.arm2.is_treatment,
                'nar_values': [{'timepoint': n.timepoint, 'value': n.value}
                              for n in rct.arm2.nar_values]
            }

        return data

    def _dict_to_rct(self, data: Dict) -> 'RCTGroundTruth':
        """Convert dict to RCTGroundTruth."""
        # Parse arm info
        arm1 = None
        arm2 = None

        if 'arm1' in data and data['arm1']:
            arm1_data = data['arm1']
            arm1 = ArmInfo(
                label=arm1_data.get('label', ''),
                color=arm1_data.get('color', ''),
                is_treatment=arm1_data.get('is_treatment', False),
                nar_values=[NARTimepoint(n['timepoint'], n['value'])
                           for n in arm1_data.get('nar_values', [])]
            )

        if 'arm2' in data and data['arm2']:
            arm2_data = data['arm2']
            arm2 = ArmInfo(
                label=arm2_data.get('label', ''),
                color=arm2_data.get('color', ''),
                is_treatment=arm2_data.get('is_treatment', False),
                nar_values=[NARTimepoint(n['timepoint'], n['value'])
                           for n in arm2_data.get('nar_values', [])]
            )

        return RCTGroundTruth(
            paper_id=data.get('paper_id', ''),
            pdf_path=data.get('pdf_path', ''),
            doi=data.get('doi', ''),
            hr_reported=data.get('hr_reported', 0.0),
            ci_lower=data.get('ci_lower', 0.0),
            ci_upper=data.get('ci_upper', 0.0),
            p_value=data.get('p_value'),
            x_axis_min=data.get('x_axis_min', 0.0),
            x_axis_max_months=data.get('x_axis_max_months', 0.0),
            x_axis_unit=data.get('x_axis_unit', 'months'),
            y_axis_scale=data.get('y_axis_scale', '0-1'),
            y_axis_min=data.get('y_axis_min', 0.0),
            y_axis_max=data.get('y_axis_max', 1.0),
            arm1=arm1,
            arm2=arm2,
            n_arms=data.get('n_arms', 2),
            has_nar_table=data.get('has_nar_table', False),
            nar_timepoints=data.get('nar_timepoints', []),
            figure_page=data.get('figure_page', 0),
            figure_panel=data.get('figure_panel', ''),
            figure_caption=data.get('figure_caption', ''),
            therapeutic_area=data.get('therapeutic_area', ''),
            outcome_type=data.get('outcome_type', ''),
            median_followup_months=data.get('median_followup_months'),
            n_total=data.get('n_total', 0),
            n_arm1=data.get('n_arm1', 0),
            n_arm2=data.get('n_arm2', 0),
            n_events_total=data.get('n_events_total', 0),
            n_events_arm1=data.get('n_events_arm1', 0),
            n_events_arm2=data.get('n_events_arm2', 0),
            has_crossing_curves=data.get('has_crossing_curves', False),
            has_confidence_bands=data.get('has_confidence_bands', False),
            has_censoring_marks=data.get('has_censoring_marks', False),
            verified=data.get('verified', False),
            verified_by=data.get('verified_by', ''),
            verification_date=data.get('verification_date', ''),
            verification_notes=data.get('verification_notes', ''),
            created_date=data.get('created_date', ''),
            modified_date=data.get('modified_date', ''),
            source_url=data.get('source_url', ''),
        )

    def get_rct_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for RCT ground truth entries.

        Returns
        -------
        Dict with summary statistics
        """
        rcts = self.get_all_rct_ground_truths()
        if not rcts:
            return {'n_rcts': 0}

        verified = [r for r in rcts if r.verified]
        with_nar = [r for r in rcts if r.has_nar_table]
        hrs = [r.hr_reported for r in rcts if r.hr_reported > 0]

        summary = {
            'n_rcts': len(rcts),
            'n_verified': len(verified),
            'n_with_nar': len(with_nar),
            'verification_rate': len(verified) / len(rcts) * 100 if rcts else 0,
            'nar_rate': len(with_nar) / len(rcts) * 100 if rcts else 0,
        }

        if hrs:
            import statistics
            summary['hr_mean'] = statistics.mean(hrs)
            summary['hr_median'] = statistics.median(hrs)
            summary['hr_min'] = min(hrs)
            summary['hr_max'] = max(hrs)

        # By outcome type
        summary['by_outcome'] = {}
        for rct in rcts:
            outcome = rct.outcome_type or 'unknown'
            summary['by_outcome'][outcome] = summary['by_outcome'].get(outcome, 0) + 1

        return summary

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        stats = {
            'total_pdfs': len(self.pdfs),
            'total_entries': len(self.entries),
            'total_validations': len(self.validations),
            'pdfs_with_local_file': sum(1 for p in self.pdfs.values() if p.local_path),
            'entries_verified': sum(1 for e in self.entries.values() if e.verification_status == "verified"),
        }

        # By therapeutic area
        stats['by_therapeutic_area'] = {}
        for pdf in self.pdfs.values():
            area = pdf.therapeutic_area or 'unknown'
            stats['by_therapeutic_area'][area] = stats['by_therapeutic_area'].get(area, 0) + 1

        # By journal style
        stats['by_journal_style'] = {}
        for pdf in self.pdfs.values():
            style = pdf.journal_style or 'unknown'
            stats['by_journal_style'][style] = stats['by_journal_style'].get(style, 0) + 1

        # HR distribution
        hrs = [e.hr for e in self.entries.values()]
        if hrs:
            stats['hr_min'] = min(hrs)
            stats['hr_max'] = max(hrs)
            stats['hr_mean'] = sum(hrs) / len(hrs)

        # Multi-arm trials
        stats['multi_arm_trials'] = sum(1 for p in self.pdfs.values() if p.n_arms > 2)

        # With crossing curves
        stats['crossing_curve_trials'] = sum(1 for p in self.pdfs.values() if p.has_crossing_curves)

        # Validation summary
        stats['validation_summary'] = self.get_validation_summary()

        return stats


def create_database(db_path: str = "real_validation_datasets") -> GroundTruthDatabase:
    """
    Create or open a ground truth database.

    Parameters
    ----------
    db_path : str
        Path to database directory

    Returns
    -------
    GroundTruthDatabase
    """
    return GroundTruthDatabase(db_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Ground Truth Database")
    print("=" * 50)

    # Create database
    db = create_database()

    # Print statistics
    stats = db.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total PDFs: {stats['total_pdfs']}")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Verified entries: {stats['entries_verified']}")
    print(f"  Total validations: {stats['total_validations']}")

    if stats.get('by_therapeutic_area'):
        print(f"\nBy Therapeutic Area:")
        for area, count in stats['by_therapeutic_area'].items():
            print(f"  {area}: {count}")

    print(f"\nDatabase hash: {db.get_db_hash()}")
