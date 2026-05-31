🎯 DecodeLabs AI Engineering Internship — Week 3
Project 3: Enterprise AI Recommendation Logic (Tech Stack Recommender)
Internship Track: AI Engineering Intern
Framework: Scikit-Learn | Content-Based Filtering | TF-IDF Vectorization
Algorithm: Cosine Similarity Matching
Architecture: Vector Mapping Space & Content-Based Filtering Pipeline
Batch: 2026 | Powered by DecodeLabs
📋 Table of Contents
Project Overview
Key Features
Technical Stack
Architecture Diagram
Project Structure
Installation & Setup
Configuration (.env)
How to Run
Interactive Demo
How It Works
Results & Output
Screenshots
Learning Outcomes
Acknowledgements
🎯 Project Overview
This project implements an Enterprise AI Recommendation System that matches user skills to optimal job roles using Content-Based Filtering and TF-IDF Vectorization.
What It Does:
🧑‍💻 Captures user skills through an interactive terminal survey
🧠 Expands abbreviations and synonyms (e.g., "ml" → "Machine Learning")
📊 Vectorizes both user profile and job requirements using TF-IDF
🎯 Computes Cosine Similarity to find the best matching job roles
📈 Ranks top recommendations with confidence tiers and skill gap analysis
Job Roles Covered:
Table
Role	Key Skills
🧪 Data Scientist	Python, Machine Learning, SQL, Statistics, TensorFlow, PyTorch
⚙️ DevOps Engineer	AWS, Docker, Kubernetes, CI/CD, Linux, Jenkins, Ansible
💻 Backend Developer	Java, Python, SQL, APIs, Django, Node.js, MongoDB
🎨 Frontend Developer	JavaScript, HTML, CSS, React, Angular, TypeScript, Vue
☁️ Cloud Architect	AWS, Azure, Security, Terraform, CloudFormation, Infrastructure
✨ Key Features
Table
Feature	Description	Status
✅ Secure Config	Environment variables via .env — no hardcoded paths or settings	✅
✅ Synonym Expansion	Auto-expands abbreviations (ml, ai, cicd, web, cloud, ds, py, js, db, devops)	✅
✅ TF-IDF Vectorization	Converts text skills into numerical vectors for mathematical comparison	✅
✅ Cosine Similarity	Measures directional alignment between user and job skill vectors	✅
✅ Confidence Tiers	HIGH (≥35%), MEDIUM (≥10%), LOW (<10%) compatibility scoring	✅
✅ Skill Gap Analysis	Identifies missing skills for each recommended role	✅
✅ CSV Ingestion	Auto-detects and loads external raw_skills.csv dataset	✅
✅ Fallback Dataset	Embedded baseline data if CSV is missing or corrupted	✅
✅ Robust Error Handling	Custom exceptions for dataset, vectorization, and I/O failures	✅
✅ Auto-Logging	Dual-channel logging (Console + app.log file)	✅
✅ Professional Terminal UI	Unicode box-drawing frames with safe ANSI colors	✅
✅ Type Hints	Full function typing for code clarity and IDE support	✅
🛠️ Technical Stack
plain
Python 3.10+
├── Standard Library
│   ├── os, re, sys, logging, textwrap
│   ├── dataclasses, datetime, pathlib
│   └── typing (Dict, List, Tuple, Optional, Final)
│
├── Third-Party
│   ├── pandas              → Data manipulation & CSV ingestion
│   ├── numpy               → Numerical computations
│   ├── scikit-learn        → TF-IDF Vectorizer + Cosine Similarity
│   └── python-dotenv       → Secure configuration management
│
└── Design Patterns
    ├── Strategy (Matching Algorithm)
    ├── Factory (Dataset Loader)
    └── DTO (Recommendation dataclass)
🏗️ Architecture Diagram
plain
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INPUT LAYER                                │
│                    Interactive Terminal Survey                          │
│              (3 Skills → Regex Sanitization → Expansion)                  │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────────┐
│                      DATA INGESTION LAYER                                 │
│  ┌─────────────────────┐      ┌─────────────────────┐                  │
│  │   CSV File Loader   │      │  Fallback Dataset   │                  │
│  │  (raw_skills.csv)   │─────→│  (Embedded JSON)    │                  │
│  └─────────────────────┘      └─────────────────────┘                  │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────────┐
│                    VECTORIZATION LAYER (TF-IDF)                         │
│                                                                         │
│   User Skills ──→┌─────────────┐                                       │
│                   │  TF-IDF     │──→ Skill Vectors (Sparse Matrix)      │
│   Job Skills ────→│ Vectorizer  │                                       │
│                   └─────────────┘                                       │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────────┐
│                  SIMILARITY COMPUTATION LAYER                           │
│                                                                         │
│   User Vector ──→┌─────────────────┐                                   │
│                   │ Cosine Similarity │──→ Similarity Scores [0.0 - 1.0] │
│   Job Vectors ──→│   (sklearn)     │                                   │
│                   └─────────────────┘                                   │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────────┐
│                   RECOMMENDATION ENGINE LAYER                             │
│                                                                         │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│   │  Rank & Sort    │  │ Confidence Tier │  │ Skill Gap       │         │
│   │  (Top-N)        │  │ Classification  │  │ Analysis        │         │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
📁 Project Structure
plain
DecodeLabs-Week3-TechRecommender/
│
├── 📄 README.md                          ← You are here
├── 📄 decodelabs_project3_vvip.py        ← Main application (single file)
├── 📄 .env                               ← Environment configuration (DO NOT UPLOAD TO GITHUB!)
├── 📄 .gitignore                         ← Excludes .env, logs/, *.log
├── 📄 requirements.txt                   ← Python dependencies
│
├── 📂 logs/
│   └── 📄 app.log                        ← Auto-generated audit trail
│
├── 📄 raw_skills.csv                     ← Optional external dataset
│
└── 📂 assets/
    ├── 📸 screenshot_input.png             ← User skill input screenshot
    ├── 📸 screenshot_output.png            ← Recommendation results screenshot
    └── 📸 screenshot_logs.png            ← app.log audit trail screenshot
⚙️ Installation & Setup
Step 1: Clone or Download
bash
git clone https://github.com/yourusername/DecodeLabs-Week3-TechRecommender.git
cd DecodeLabs-Week3-TechRecommender
Step 2: Create Virtual Environment (Recommended)
bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
Step 3: Install Dependencies
bash
pip install pandas numpy scikit-learn python-dotenv
Or use requirements file:
bash
pip install -r requirements.txt
Step 4: Create .env File
Create a file named .env in the same directory as the Python script:
env
# ── Dataset ───────────────────────────
CSV_PATH=raw_skills.csv

# ── Logging ───────────────────────────
LOG_LEVEL=INFO
LOG_FILE=app.log
LOG_DIR=logs

# ── Recommendation Engine ─────────────
MIN_SKILLS=3
TOP_N=3
SIMILARITY_HIGH=0.35
SIMILARITY_MED=0.10

# ── Fallback Profile ──────────────────
FALLBACK_SKILLS=Python,Cloud Computing,Automation
⚠️ IMPORTANT: Add .env to your .gitignore file so it never gets uploaded to GitHub!
gitignore
# .gitignore
.env
logs/
*.log
__pycache__/
venv/
🚀 How to Run
Run the Main Script
bash
python decodelabs_project3_vvip.py
Expected Interaction Flow:
plain
╔═══════════════════════════════════════════════════════════════════════════╗
║          DECODELABS DIGITAL MATCHMAKER: INTERACTIVE SURVEY                ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Please enter exactly 3 skills, pressing Enter after each one.            ║
╠═══════════════════════════════════════════════════════════════════════════╣
▸ Skill 1: Python
▸ Skill 2: Machine Learning
▸ Skill 3: SQL

╔═══════════════════════════════════════════════════════════════════════════╗
║               CRITICAL ARCHITECTURE REPORT: TOP-3 RECOMMENDATIONS           ║
╠═══════════════════════════════════════════════════════════════════════════╣
 1. 🎯 JOB ROLE ARCHETYPE : Data Scientist ([HIGH COMPATIBILITY])
    📈 Vector Fit Matching Index : 85.42%
    ⚠️  Core Skill Gaps Detected  : Statistics, TensorFlow, PyTorch
    🧬 Full Vocabulary Matrix    : Python Machine Learning Data Analysis...

╠═══════════════════════════════════════════════════════════════════════════╣
 2. 🎯 JOB ROLE ARCHETYPE : Backend Developer ([MEDIUM COMPATIBILITY])
    📈 Vector Fit Matching Index : 62.15%
    ⚠️  Core Skill Gaps Detected  : Java, Django, Node.js, Express
    🧬 Full Vocabulary Matrix    : Java Python SQL APIs Django Node...

╠═══════════════════════════════════════════════════════════════════════════╣
 3. 🎯 JOB ROLE ARCHETYPE : DevOps Engineer ([LOW COMPATIBILITY])
    📈 Vector Fit Matching Index : 18.33%
    ⚠️  Core Skill Gaps Detected  : AWS, Docker, Kubernetes, CI/CD...
    🧬 Full Vocabulary Matrix    : AWS Docker Kubernetes CI/CD...

╚═══════════════════════════════════════════════════════════════════════════╝
🎮 Interactive Demo
Sample Inputs & Expected Outputs:
Table
Input Skills	Top Recommendation	Confidence	Key Gaps
Python, ML, SQL	Data Scientist	HIGH	Statistics, TensorFlow
AWS, Docker, Linux	DevOps Engineer	HIGH	Kubernetes, Jenkins
HTML, CSS, JS	Frontend Developer	HIGH	React, Angular, TypeScript
Cloud, Security, Python	Cloud Architect	HIGH	Terraform, Azure
Random, Words, Here	(Fallback activated)	—	—
Synonym Expansion Examples:
Table
User Types	Expanded To
ml	Machine Learning
ai	Artificial Intelligence
cicd	CI/CD
web	Web Design Frontend HTML CSS
cloud	Cloud Computing AWS Azure
ds	Data Scientist Statistics
py	Python
js	JavaScript
db	Database SQL Postgres MongoDB
devops	Docker Kubernetes CI/CD Jenkins
🔬 How It Works
Step 1: TF-IDF Vectorization
plain
User Input:  "Python Machine Learning SQL"
Job Role 1:  "Python Machine Learning Data Analysis SQL Statistics TensorFlow"
Job Role 2:  "AWS Docker Kubernetes CI/CD Automation Linux Git Jenkins"

TF-IDF converts these into sparse numerical vectors where:
- Rare terms (e.g., "TensorFlow") get higher weights
- Common terms (e.g., "Python") get lower weights
Step 2: Cosine Similarity
plain
Similarity = cos(θ) = (A · B) / (||A|| × ||B||)

Range: 0.0 (completely different) → 1.0 (identical)
Step 3: Confidence Classification
plain
Score ≥ 0.35  →  🟢 HIGH COMPATIBILITY
Score ≥ 0.10  →  🟡 MEDIUM COMPATIBILITY
Score < 0.10  →  🔴 LOW COMPATIBILITY
Step 4: Skill Gap Analysis
Python
missing_skills = job_skills_set - user_skills_set
# Shows exactly which skills the user needs to learn
📊 Results & Output
Console Output Includes:
✅ Top-N ranked job recommendations (default: 3)
✅ Vector Fit Matching Index (percentage similarity)
✅ Color-coded confidence badge (Green/Yellow/Red)
✅ Core Skill Gaps (missing skills per role)
✅ Full Vocabulary Matrix (complete skill set for the role)
Log File (logs/app.log) Captures:
plain
[2026-05-31 11:15:32] [INFO] [initialize_logging:95] Logging infrastructure initialized
[2026-05-31 11:15:35] [INFO] [_initialize_dataset:142] Dataset ingested successfully | Rows: 5
[2026-05-31 11:15:38] [DEBUG] [_expand_synonyms:312] Synonym expansion | 'ml' → 'Machine Learning'
[2026-05-31 11:15:38] [INFO] [compute_recommendation_matrix:245] Recommendation matrix computed | Max similarity: 0.8542
📸 Screenshots
Add your execution screenshots in the /assets/ folder and reference them here:
Table
Screenshot	Description
User entering 3 skills in terminal
Top-3 recommendations with confidence & gaps
app.log showing audit trail
Optional custom dataset loading
🎓 Learning Outcomes
Through this project, I have demonstrated:
✅ Content-Based Filtering — Building recommendation systems without collaborative data
✅ TF-IDF Vectorization — Converting unstructured text into machine-readable vectors
✅ Cosine Similarity — Measuring semantic alignment in high-dimensional space
✅ NLP Preprocessing — Regex sanitization, synonym expansion, token normalization
✅ Secure Configuration — Zero hardcoded values; full .env externalization
✅ Defensive Programming — Custom exceptions, fallback datasets, graceful degradation
✅ Professional Logging — Structured, timestamped, dual-channel audit system
✅ Terminal UI Design — Unicode box-drawing with safe, auto-detecting ANSI colors
✅ Type Safety — Comprehensive type hints across all functions and classes
✅ DataFrame Operations — pandas for CSV ingestion, manipulation, and sorting
🙏 Acknowledgements
DecodeLabs — For providing this structured internship and learning opportunity
Scikit-Learn Team — For the robust ML framework (TfidfVectorizer, cosine_similarity)
Pandas Development Team — For powerful data manipulation tools
Python Software Foundation — For the versatile standard library
📬 Contact
Intern Name: [M AHMED ALI ]
Email: [muhammadahmedali607@gmail.com ]
LinkedIn: [  www.linkedin.com/in/muhammad-ahmed-ali-123125406]
GitHub: [https://github.com/mrahmed121]
<div align="center">
⭐ DecodeLabs AI Engineering Internship — Week 1 ⭐
Project 1: AI-Powered Conversational Bot (Rule-Based Edition)
Submitted as part of the official internship program.
</div>
