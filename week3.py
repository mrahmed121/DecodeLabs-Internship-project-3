
"""
================================================================================
  DECODELABS INDUSTRIAL TRAINING KIT — ARTIFICIAL INTELLIGENCE
  Project 3: Enterprise AI Recommendation Logic (Tech Stack Recommender)
  Vector Mapping Space & Content-Based Filtering Pipeline

  VVIP Professional Edition | Industry-Grade Refactor
  Batch: 2026 | Powered by DecodeLabs
================================================================================

Upgrades Applied:
  1. Secure Configuration Management (python-dotenv + os.environ)
  2. Advanced Error Handling & Custom Exception Hierarchy
  3. Interactive Terminal UI with Box-Drawing & Safe ANSI Colors
  4. Dual-Channel Auto-Logging (Console + app.log file)
  5. Performance Optimization, Type Hints & Comprehensive Documentation
"""

from __future__ import annotations

import os
import re
import sys
import logging
import textwrap
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional, Final
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — SECURE CONFIGURATION MANAGEMENT (dotenv + Environment Variables)
# ═══════════════════════════════════════════════════════════════════════════════

load_dotenv(override=True)


class Config:
    """
    Centralized, secure configuration manager.

    All tunable parameters are injected via environment variables (/.env file)
    so that no hardcoded secrets or paths ever touch version control.

    Usage:
        Create a `.env` file in the project root:
            CSV_PATH=raw_skills.csv
            LOG_LEVEL=INFO
            MIN_SKILLS=3
            TOP_N=3
    """
    CSV_PATH: Final[str]       = os.getenv("CSV_PATH", "raw_skills.csv")
    LOG_LEVEL: Final[str]      = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE: Final[str]       = os.getenv("LOG_FILE", "app.log")
    MIN_SKILLS: Final[int]     = int(os.getenv("MIN_SKILLS", "3"))
    TOP_N: Final[int]          = int(os.getenv("TOP_N", "3"))
    SIMILARITY_HIGH: Final[float] = float(os.getenv("SIMILARITY_HIGH", "0.35"))
    SIMILARITY_MED: Final[float]  = float(os.getenv("SIMILARITY_MED", "0.10"))
    FALLBACK_SKILLS: Final[List[str]] = os.getenv(
        "FALLBACK_SKILLS", "Python,Cloud Computing,Automation"
    ).split(",")

    @classmethod
    def validate(cls) -> None:
        """Runtime validation of critical environment parameters."""
        if cls.MIN_SKILLS < 1:
            raise ValueError("MIN_SKILLS must be >= 1")
        if cls.TOP_N < 1:
            raise ValueError("TOP_N must be >= 1")
        if not (0.0 <= cls.SIMILARITY_HIGH <= 1.0):
            raise ValueError("SIMILARITY_HIGH must be in [0.0, 1.0]")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CUSTOM EXCEPTION HIERARCHY (Advanced Error Handling)
# ═══════════════════════════════════════════════════════════════════════════════

class RecommenderError(Exception):
    """Base exception for all recommender-domain errors."""
    pass


class DatasetIngestionError(RecommenderError):
    """Raised when CSV ingestion fails (missing file, malformed schema, IO error)."""
    pass


class InsufficientInputError(RecommenderError):
    """Raised when user provides fewer skills than the MIN_SKILLS threshold."""
    pass


class VectorizationError(RecommenderError):
    """Raised when TF-IDF vectorization or cosine similarity computation fails."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — AUTO-LOGGING SYSTEM (Dual-Channel: Console + File)
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_logging() -> logging.Logger:
    """
    Initialize a production-grade logger with dual handlers.

    - StreamHandler   → Console (colored, human-readable)
    - FileHandler     → app.log (persistent, timestamped, debug-level)

    Returns:
        logging.Logger: The root logger instance for the application.
    """
    logger = logging.getLogger("DecodeLabs_Recommender")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on re-import
    if logger.handlers:
        return logger

    # ── Console Handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))
    console_fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # ── File Handler (Persistent Audit Trail) ─────────────────────────────────
    file_handler = logging.FileHandler(Config.LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    logger.info("Logging infrastructure initialized | File: %s | Level: %s", Config.LOG_FILE, Config.LOG_LEVEL)
    return logger


LOGGER: Final[logging.Logger] = initialize_logging()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — INTERACTIVE TERMINAL UI (Box-Drawing & Safe ANSI Colors)
# ═══════════════════════════════════════════════════════════════════════════════

class TerminalUI:
    """
    Production-grade terminal UI renderer using Unicode box-drawing characters.

    All ANSI color codes are wrapped in a safe toggle so the UI degrades gracefully
    on terminals that do not support color (e.g., Windows CMD without ANSI, CI logs).
    """

    # Unicode box-drawing glyphs
    HORIZONTAL: Final[str] = "═"
    VERTICAL:   Final[str] = "║"
    TOP_LEFT:   Final[str] = "╔"
    TOP_RIGHT:  Final[str] = "╗"
    BOT_LEFT:   Final[str] = "╚"
    BOT_RIGHT:  Final[str] = "╝"
    T_LEFT:     Final[str] = "╠"
    T_RIGHT:    Final[str] = "╣"
    BULLET:     Final[str] = "▸"

    def __init__(self, use_color: bool = True) -> None:
        """
        Args:
            use_color: If False, all ANSI escape sequences are suppressed.
        """
        self._color = use_color and self._supports_ansi()

    @staticmethod
    def _supports_ansi() -> bool:
        """Heuristic: disable colors on Windows legacy terminals unless explicitly forced."""
        if os.getenv("FORCE_COLOR", "0") == "1":
            return True
        if os.name == "nt" and not os.getenv("ANSICON") and not os.getenv("WT_SESSION"):
            return False
        return sys.stdout.isatty()

    def _c(self, code: str, text: str) -> str:
        """Wrap text in an ANSI color code if colors are enabled."""
        if not self._color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text: str) -> str:
        return self._c("1", text)

    def green(self, text: str) -> str:
        return self._c("92", text)

    def yellow(self, text: str) -> str:
        return self._c("93", text)

    def red(self, text: str) -> str:
        return self._c("91", text)

    def cyan(self, text: str) -> str:
        return self._c("36", text)

    def magenta(self, text: str) -> str:
        return self._c("35", text)

    def header(self, title: str, width: int = 75) -> str:
        """Render a framed header banner."""
        pad = (width - len(title) - 2) // 2
        line = f"{self.TOP_LEFT}{self.HORIZONTAL * (width - 2)}{self.TOP_RIGHT}"
        text_line = f"{self.VERTICAL}{' ' * pad}{self.bold(title)}{' ' * (width - 2 - pad - len(title))}{self.VERTICAL}"
        return f"\n{line}\n{text_line}\n{self.T_LEFT}{self.HORIZONTAL * (width - 2)}{self.T_RIGHT}"

    def footer(self, width: int = 75) -> str:
        """Render a closing footer banner."""
        return f"{self.BOT_LEFT}{self.HORIZONTAL * (width - 2)}{self.BOT_RIGHT}\n"

    def info_line(self, label: str, value: str, width: int = 75) -> str:
        """Render a key-value pair inside the frame."""
        content = f"  {self.BULLET} {label}: {value}"
        padding = width - 2 - len(content)
        return f"{self.VERTICAL}{content}{' ' * max(padding, 0)}{self.VERTICAL}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — DATA MODELS (Type-Safe Structures)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Recommendation:
    """Immutable data transfer object for a single job recommendation."""
    rank: int
    role: str
    similarity_pct: float
    confidence_label: str
    gap_report: str
    required_skills: str


@dataclass
class UserProfile:
    """Validated user input container with synonym expansion."""
    raw_skills: List[str] = field(default_factory=list)
    expanded_skills: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def skill_set(self) -> Set[str]:
        """Return a normalized (lowercased) set of all expanded skills."""
        return {s.lower().strip() for s in self.expanded_skills}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — CORE RECOMMENDER ENGINE (Optimized & Documented)
# ═══════════════════════════════════════════════════════════════════════════════

class TechStackRecommender:
    """
    Enterprise-grade content-based recommendation engine.

    Pipeline:
        1. Ingest job-role metadata (CSV or programmatic fallback).
        2. Capture & sanitize user skill profile via interactive CLI.
        3. Expand synonyms to enrich semantic matching.
        4. Vectorize via TF-IDF and compute cosine similarity.
        5. Rank roles, classify confidence tiers, and report skill gaps.

    Attributes:
        csv_path: Path to the external skills database.
        vectorizer: sklearn TfidfVectorizer (pre-compiled token regex).
        dataframe: pandas DataFrame holding roles and required skills.
        ui: TerminalUI instance for consistent visual output.
        _synonym_map: Pre-loaded dictionary for skill synonym expansion.
        _skill_cleaner: Pre-compiled regex for input sanitization.
    """

    def __init__(self, csv_path: Optional[str] = None) -> None:
        self.csv_path: str = csv_path or Config.CSV_PATH
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
            norm="l2",               # L2 normalization for stable cosine similarity
            use_idf=True,
            smooth_idf=True
        )
        self.ui: TerminalUI = TerminalUI()
        self._synonym_map: Dict[str, str] = {
            "ml": "Machine Learning",
            "ai": "Artificial Intelligence",
            "cicd": "CI/CD",
            "web": "Web Design Frontend HTML CSS",
            "cloud": "Cloud Computing AWS Azure",
            "ds": "Data Scientist Statistics",
            "py": "Python",
            "js": "JavaScript",
            "db": "Database SQL Postgres MongoDB",
            "devops": "Docker Kubernetes CI/CD Jenkins"
        }
        # Pre-compile regex for O(1) input sanitization
        self._skill_cleaner: re.Pattern = re.compile(r"[^a-zA-Z\s\-]")

        LOGGER.info("Engine initialized | CSV target: %s", self.csv_path)
        self.dataframe: pd.DataFrame = self._initialize_dataset()

    # ── 5.1 Dataset Ingestion ─────────────────────────────────────────────────
    def _initialize_dataset(self) -> pd.DataFrame:
        """
        Attempt to load job metadata from CSV; fall back to embedded baseline
        if the file is missing, unreadable, or schema-incompatible.

        Raises:
            DatasetIngestionError: Only if the fallback itself is corrupted
                                   (defensive, should never happen).

        Returns:
            pd.DataFrame: Validated DataFrame with 'Job_Role' & 'Required_Skills'.
        """
        if not os.path.exists(self.csv_path):
            LOGGER.warning("External CSV not found at '%s'. Loading baseline fallback.", self.csv_path)
            return self._load_fallback()

        try:
            df = pd.read_csv(self.csv_path)
            if {"Job_Role", "Required_Skills"}.issubset(df.columns):
                LOGGER.info("Dataset ingested successfully | Rows: %d | Source: %s", len(df), self.csv_path)
                return df
            else:
                LOGGER.error("CSV schema mismatch | Required columns: Job_Role, Required_Skills")
                return self._load_fallback()
        except pd.errors.EmptyDataError:
            LOGGER.error("CSV file is empty: %s", self.csv_path)
            return self._load_fallback()
        except pd.errors.ParserError as exc:
            LOGGER.error("CSV parse failure: %s | %s", self.csv_path, exc)
            return self._load_fallback()
        except OSError as exc:
            LOGGER.error("OS-level read error on '%s': %s", self.csv_path, exc)
            return self._load_fallback()

    def _load_fallback(self) -> pd.DataFrame:
        """Embedded baseline dataset — guaranteed to succeed."""
        baseline = {
            "Job_Role": [
                "Data Scientist",
                "DevOps Engineer",
                "Backend Developer",
                "Frontend Developer",
                "Cloud Architect"
            ],
            "Required_Skills": [
                "Python Machine Learning Data Analysis SQL Statistics TensorFlow PyTorch Scikit-Learn",
                "AWS Docker Kubernetes CI/CD Automation Linux Git Jenkins Ansible",
                "Java Python SQL APIs Django Node Express Backend Postgres MongoDB RestAPI",
                "JavaScript HTML CSS React Angular Frontend Web Design TypeScript Bootstrap Vue",
                "AWS Cloud Computing Azure Security Infrastructure Automation Terraform CloudFormation"
            ]
        }
        LOGGER.info("Baseline programmatic metadata store loaded | Rows: %d", len(baseline["Job_Role"]))
        return pd.DataFrame(baseline)

    # ── 5.2 Input Capture & Sanitization ────────────────────────────────────────
    def capture_onboarding_profile(self) -> UserProfile:
        """
        Interactive CLI survey to collect user skills.

        Validates:
            - Non-empty input after regex sanitization.
            - Minimum skill count (default 3, driven by MIN_SKILLS env var).

        On validation failure, logs the error and returns a safe fallback profile
        so the application never crashes.

        Returns:
            UserProfile: Validated, timestamped, synonym-expanded user profile.
        """
        print(self.ui.header("DECODELABS DIGITAL MATCHMAKER"))
        print(f"{self.ui.VERTICAL}  Please enter exactly {Config.MIN_SKILLS} skills, pressing Enter after each one.{' ' * (75 - 2 - 72)}{self.ui.VERTICAL}")
        print(self.ui.T_LEFT + self.ui.HORIZONTAL * 73 + self.ui.T_RIGHT)

        raw_skills: List[str] = []
        for idx in range(1, Config.MIN_SKILLS + 1):
            try:
                raw_input = input(f"{self.ui.VERTICAL}  {self.ui.BULLET} Skill {idx}: ")
            except (EOFError, KeyboardInterrupt):
                LOGGER.warning("User interrupted input stream at skill %d", idx)
                print(f"\n{self.ui.red('Input stream interrupted. Using fallback profile.')}")
                return self._build_fallback_profile()

            cleaned = self._skill_cleaner.sub("", raw_input).strip()
            if cleaned:
                raw_skills.append(cleaned)
                LOGGER.debug("Captured skill %d: '%s'", idx, cleaned)
            else:
                LOGGER.warning("Empty or invalid input at skill %d", idx)

        if len(raw_skills) < Config.MIN_SKILLS:
            LOGGER.error(
                "Insufficient input density | Provided: %d | Required: %d | Switching to fallback.",
                len(raw_skills), Config.MIN_SKILLS
            )
            print(f"\n{self.ui.red(f'ERROR: Only {len(raw_skills)} valid skill(s) provided. Minimum required: {Config.MIN_SKILLS}.')}")
            print(f"{self.ui.yellow('Activating safe fallback profile...')}\n")
            return self._build_fallback_profile()

        expanded = self._expand_synonyms(raw_skills)
        profile = UserProfile(raw_skills=raw_skills, expanded_skills=expanded)
        LOGGER.info("Profile captured | Raw: %s | Expanded: %s", raw_skills, expanded)
        return profile

    def _build_fallback_profile(self) -> UserProfile:
        """Construct a guaranteed-safe fallback profile from environment config."""
        skills = [s.strip() for s in Config.FALLBACK_SKILLS]
        expanded = self._expand_synonyms(skills)
        LOGGER.info("Fallback profile activated | Skills: %s", expanded)
        return UserProfile(raw_skills=skills, expanded_skills=expanded)

    # ── 5.3 Semantic Enrichment (Synonym Expansion) ────────────────────────────
    def _expand_synonyms(self, skills_list: List[str]) -> List[str]:
        """
        Enrich the user vocabulary by appending canonical synonyms.

        This improves vector-space coverage when users abbreviate terms
        (e.g., 'ml' → 'Machine Learning').

        Args:
            skills_list: Sanitized raw skill strings from user input.

        Returns:
            List[str]: Original skills + appended canonical expansions.
        """
        expanded: List[str] = []
        for skill in skills_list:
            expanded.append(skill)
            lowered = skill.lower()
            if lowered in self._synonym_map:
                canonical = self._synonym_map[lowered]
                expanded.append(canonical)
                LOGGER.debug("Synonym expansion | '%s' → '%s'", skill, canonical)
        return expanded

    # ── 5.4 Vectorization & Similarity Computation ──────────────────────────────
    def compute_recommendation_matrix(self, profile: UserProfile) -> pd.DataFrame:
        """
        Transform user and job descriptions into TF-IDF vectors and compute
        pairwise cosine similarity.

        Args:
            profile: Validated UserProfile containing expanded skills.

        Raises:
            VectorizationError: If sklearn raises an unexpected exception during
                                vectorization or similarity computation.

        Returns:
            pd.DataFrame: Original dataframe augmented with a 'Similarity_Score' column.
        """
        user_profile_string = " ".join(profile.expanded_skills)
        corpus: List[str] = self.dataframe["Required_Skills"].tolist() + [user_profile_string]

        LOGGER.debug("Vectorizing corpus | Documents: %d | User profile length: %d chars", len(corpus), len(user_profile_string))

        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except ValueError as exc:
            LOGGER.exception("TF-IDF vectorization failed")
            raise VectorizationError(f"TF-IDF fit_transform failed: {exc}") from exc

        item_vectors = tfidf_matrix[:-1]
        user_vector = tfidf_matrix[-1]

        try:
            scores = cosine_similarity(user_vector, item_vectors).flatten()
        except Exception as exc:
            LOGGER.exception("Cosine similarity computation failed")
            raise VectorizationError(f"Cosine similarity failed: {exc}") from exc

        scored_df = self.dataframe.copy()
        scored_df["Similarity_Score"] = scores
        LOGGER.info("Recommendation matrix computed | Max similarity: %.4f", float(np.max(scores)))
        return scored_df

    # ── 5.5 Confidence Classification & Gap Analysis ──────────────────────────
    def _classify_confidence(self, score: float) -> str:
        """
        Map a similarity score to a human-readable confidence tier.

        Tiers are driven by environment variables SIMILARITY_HIGH and SIMILARITY_MED.

        Args:
            score: Cosine similarity in [0.0, 1.0].

        Returns:
            str: ANSI-colored confidence label.
        """
        if score >= Config.SIMILARITY_HIGH:
            return self.ui.green("HIGH COMPATIBILITY")
        elif score >= Config.SIMILARITY_MED:
            return self.ui.yellow("MEDIUM COMPATIBILITY")
        return self.ui.red("LOW COMPATIBILITY")

    def _compute_skill_gaps(self, row: pd.Series, user_skills: Set[str]) -> str:
        """
        Identify missing skills by set difference between job requirements
        and the user's expanded vocabulary.

        Args:
            row: A single row from the scored dataframe.
            user_skills: Normalized set of user skills.

        Returns:
            str: Comma-separated gap report (max 4 items) or a perfect-match message.
        """
        job_skills: Set[str] = set(row["Required_Skills"].lower().split())
        missing: List[str] = sorted(list(job_skills - user_skills))
        if not missing:
            return self.ui.green("None! Perfect Match.")
        display_gaps = ", ".join(missing[:4])
        if len(missing) > 4:
            display_gaps += f" {self.ui.cyan(f'+{len(missing) - 4} more...')}"
        return self.ui.cyan(display_gaps)

    # ── 5.6 Presentation Layer ────────────────────────────────────────────────
    def display_tailored_output(self, scored_df: pd.DataFrame, profile: UserProfile, is_fallback: bool = False) -> None:
        """
        Render the top-N recommendations inside a polished terminal frame.

        Args:
            scored_df: DataFrame with 'Similarity_Score' column.
            profile: The user profile (for gap analysis).
            is_fallback: If True, renders a visual fallback warning banner.
        """
        sorted_matrix = scored_df.sort_values(by="Similarity_Score", ascending=False)
        top_n = sorted_matrix.head(Config.TOP_N)
        user_skill_set = profile.skill_set()

        print(self.ui.header("CRITICAL ARCHITECTURE REPORT: TOP-3 RECOMMENDATIONS"))

        if is_fallback:
            print(f"{self.ui.VERTICAL}  {self.ui.yellow('⚠ FALLBACK MODE ACTIVE — Profile generated from default configuration.')}{' ' * 12}{self.ui.VERTICAL}")
            print(self.ui.T_LEFT + self.ui.HORIZONTAL * 73 + self.ui.T_RIGHT)

        recommendations: List[Recommendation] = []
        for rank, (_, row) in enumerate(top_n.iterrows(), start=1):
            match_pct = float(row["Similarity_Score"]) * 100.0
            confidence = self._classify_confidence(row["Similarity_Score"])
            gaps = self._compute_skill_gaps(row, user_skill_set)

            rec = Recommendation(
                rank=rank,
                role=str(row["Job_Role"]),
                similarity_pct=match_pct,
                confidence_label=confidence,
                gap_report=gaps,
                required_skills=str(row["Required_Skills"])
            )
            recommendations.append(rec)

            # Render card
            print(f"{self.ui.VERTICAL}  {self.ui.bold(f'{rank}. 🎯 JOB ROLE ARCHETYPE')} : {self.ui.bold(rec.role)} ({confidence}){' ' * 15}{self.ui.VERTICAL}")
            print(f"{self.ui.VERTICAL}     📈 Vector Fit Matching Index : {match_pct:.2f}%{' ' * 42}{self.ui.VERTICAL}")
            print(f"{self.ui.VERTICAL}     ⚠️  Core Skill Gaps Detected  : {gaps}{' ' * 40}{self.ui.VERTICAL}")
            print(f"{self.ui.VERTICAL}     🧬 Full Vocabulary Matrix    : {textwrap.shorten(rec.required_skills, width=60, placeholder='...')}{' ' * 5}{self.ui.VERTICAL}")
            if rank < Config.TOP_N:
                print(self.ui.T_LEFT + self.ui.HORIZONTAL * 73 + self.ui.T_RIGHT)

        print(self.ui.footer())
        LOGGER.info("Displayed %d recommendations to user", len(recommendations))


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — MAIN ORCHESTRATOR (Graceful Entry Point)
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Application entry point with top-level exception shielding.

    Any unhandled exception is caught, logged to app.log, and presented
    to the user as a clean error message without exposing stack traces.
    """
    try:
        Config.validate()
    except ValueError as exc:
        LOGGER.error("Configuration validation failed: %s", exc)
        print(f"\n[CONFIG ERROR] {exc}\n")
        sys.exit(1)

    try:
        engine = TechStackRecommender()
        profile = engine.capture_onboarding_profile()
        result_matrix = engine.compute_recommendation_matrix(profile)

        max_sim = float(np.max(result_matrix["Similarity_Score"]))
        LOGGER.info("Maximum similarity score achieved: %.4f", max_sim)

        if max_sim == 0.0:
            LOGGER.warning("Zero-similarity state detected — activating fallback profile")
            fallback = engine._build_fallback_profile()
            result_matrix = engine.compute_recommendation_matrix(fallback)
            engine.display_tailored_output(result_matrix, fallback, is_fallback=True)
        else:
            engine.display_tailored_output(result_matrix, profile, is_fallback=False)

        LOGGER.info("Pipeline completed successfully")

    except RecommenderError as exc:
        LOGGER.error("Recommender domain error: %s", exc)
        print(f"\n{TerminalUI(use_color=True).red(f'[APPLICATION ERROR] {exc}')}\n")
        sys.exit(1)
    except Exception as exc:
        LOGGER.exception("Unhandled runtime exception")
        print(f"\n{TerminalUI(use_color=True).red(f'[CRITICAL FAILURE] An unexpected error occurred. Check app.log for details.')}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()