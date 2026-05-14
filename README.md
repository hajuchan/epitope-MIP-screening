# Epitope-MIP Computational Screening Pipeline

엑소좀 테트라스파닌 ECL2(CD63 / CD81 / CD9)를 선택적으로 인식하는 에피토프-각인 MIP의 최적 모노머 조합을 계산화학적으로 스크리닝하는 **6-Phase end-to-end 파이프라인**.

도킹(AutoDock-GPU) + MD(GROMACS) + 다목적 최적화(NSGA-II) + cross-docking selectivity penalty + 합성 레시피 자동 생성을 한 번의 명령으로 실행한다.

---

## 목차

1. [프로젝트 배경](#1-프로젝트-배경)
2. [파이프라인 전체 구조](#2-파이프라인-전체-구조)
3. [Phase 1 — 에피토프 추출 & 구조 준비](#3-phase-1--에피토프-추출--구조-준비)
4. [Phase 2 — Single Monomer Docking](#4-phase-2--single-monomer-docking-smd)
5. [Phase 3 — Multi-Monomer Simultaneous Docking + NSGA-II](#5-phase-3--multi-monomer-simultaneous-docking--nsga-ii)
6. [Phase 4 — Pre-polymerization MD](#6-phase-4--pre-polymerization-md)
7. [Phase 5 — VIP Cavity Rebinding](#7-phase-5--vip-cavity-rebinding)
8. [Phase 6 — 합성 레시피](#8-phase-6--합성-레시피)
9. [모노머 및 가교제 라이브러리](#9-모노머-및-가교제-라이브러리)
10. [핵심 알고리즘 업데이트 (A/B/C 시리즈)](#10-핵심-알고리즘-업데이트-abc-시리즈)
11. [실행 방법](#11-실행-방법)
12. [검증 프레임워크 (3-Level)](#12-검증-프레임워크-3-level)
13. [핵심 파라미터 요약](#13-핵심-파라미터-요약)
14. [설치 및 환경](#14-설치-및-환경)
15. [참고 문헌](#15-참고-문헌)

---

## 1. 프로젝트 배경

### 1.1 엑소좀과 테트라스파닌

엑소좀(exosome)은 30–150 nm 세포외 소포체로 암·신경퇴행성 질환 진단 바이오마커로 주목받는다. 표면 표지자 중 가장 보편적인 것이 **테트라스파닌 CD9 / CD63 / CD81**이다. 현재 검출 표준은 항체 기반(ELISA, flow cytometry)이지만 항체는 비용이 높고 열·pH 안정성이 낮아 POC(point-of-care) 진단에 부적합하다.

### 1.2 MIP (Molecularly Imprinted Polymer)

MIP는 "플라스틱 항체"로 불리는 합성 인식 소재이다. 템플릿 분자가 존재하는 상태에서 기능성 모노머와 가교제를 중합시키면, 템플릿 제거 후 폴리머 매트릭스에 모양·기능기가 상보적인 cavity가 남는다. 단점은 단백질처럼 큰 템플릿에는 적용이 어려운 점인데, **에피토프 각인(epitope imprinting)** — 단백질 표면에 노출된 짧은 펩타이드(보통 9–16 잔기)만 템플릿으로 사용 — 이 한계를 극복한다 [1].

### 1.3 본 파이프라인이 푸는 문제

> "주어진 단백질의 에피토프에 대해, **어떤 기능성 모노머를 몇 종, 어떤 가교제, 어떤 비율로 합성하면** 가장 선택적이고 친화도 높은 MIP가 만들어지는가?"

기존 in-silico 스크리닝은 (a) 단일 모노머 도킹만 수행 [9 기준 이전], (b) selectivity를 평가하지 않음 [9], (c) 합성 호환성(중합 메커니즘) 고려 없음. 본 파이프라인은 이 세 가지를 모두 해결한다.

### 1.4 타겟

| 단백질 | 구조 source | ECL2 (도킹 receptor) | Head 16-mer (합성 template) | N-Glycan |
|--------|------|------------|----------------------|----------|
| **CD63** | AlphaFold P08962 | 103–203 | 155–170 `EKIPSMSKNRVPDSCC` | **3개** (N130, N150, **N172**) |
| **CD81** | PDB 5TCX | 113–201 | 168–183 `SVLKNNLCPSGSNIIS` | 0 |
| **CD9** | PDB 6K4J | 112–195 | 156–171 `AGGVEQFISDICPKKD` | 1 (N52) |

CD63에는 head 후보가 4개 (canonical 16-mer + N130/N150/**N172** glycan 주변 3종). Phase 1의 A1 알고리즘이 SASA·GRAVY·protrusion·glycan 노출도를 평가하여 자동으로 최적 head를 선택한다 (현재 CD63는 `head_glyco_N172` 선택됨 — N172 글리코실레이션 site 포함).

---

## 2. 파이프라인 전체 구조

```
┌─ Phase 1 ─ 에피토프 추출 + 구조 준비 ────────────────────────┐
│ • 구조 다운로드 (AlphaFold / RCSB) → ECL2 추출              │
│ • A1: multi-epitope candidate 평가 (CD63 4개 후보)          │
│ • B1: SASA · B2: GRAVY · BLAST 고유성 검증                  │
│ • 20 ns stability MD (RMSD < 3 Å)                           │
│ • A2: K-medoids 5-conformer 추출                             │
│ → ECL2 receptor PDBQT (×6) + head 16-mer template           │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌─ Phase 2 ─ Single Monomer Docking (SMD) ─────────────────────┐
│ • AutoDock4-GPU LGA, 50 GA runs × 24 monomer × 3 target × 6  │
│   conformer = 432 dockings/target                            │
│ • A3: multi-pose clustering (RMSD 2 Å, ≥3 members)           │
│ • B3: 5종 decoy로 enrichment factor (EF > 1.5 검증)          │
│ → BE matrix + filtered top 12 monomers per target            │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌─ Phase 3 ─ MMSD + NSGA-II + Cross-Docking Selectivity ───────┐
│ • Sequential MMSD (Rajpal 2024) — multi-monomer 동시 도킹    │
│ • C2: NSGA-II 3-objective Pareto                              │
│       (Affinity + Selectivity + Synthesizability)            │
│ • **Cross-MMSD ΔΔG penalty** — 같은 조합을 다른 target       │
│   receptor에도 도킹 → 비선택적 조합에 페널티 (본 파이프라인  │
│   의 핵심 contribution)                                      │
│ • 가교제 자동 선택 (silane / vinyl 호환성 metadata 기반)     │
│ • B5: DFT validation hook                                    │
│ → Top PC (functional monomer set + crosslinker)              │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌─ Phase 4 ─ Pre-polymerization MD ────────────────────────────┐
│ • Head 16-mer template + 25개 모노머 PACKMOL 배치 + 350 ns MD│
│ • PolCA force field (Si) + GAFF2 (vinyl) + amber99sb-ildn    │
│ • A5/B7: solvent sweep (water / EtOH-water / DMSO)           │
│ • B6: ratio sweep (5 preset)                                 │
│ • Contact freq · residence time · EBN · HBNMax · per-atom RDF│
│ • MM-GBSA + per-residue decomposition                        │
│ • Q1→Q4 convergence + PBC centering (trjconv -pbc mol)       │
│ → 합성 비율 (EBN 기반) + cavity 안정성                       │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌─ Phase 5 ─ VIP Cavity Rebinding ─────────────────────────────┐
│ • 균등 간격 10 snapshot (cherry-picking 방지)                │
│ • 모노머 position restraint (1000 kJ/mol/nm²) → 중합 근사    │
│ • Template removal test (10 ns) + rebinding MD (50 ns)       │
│ • Cross-rebinding (own / cross_target1 / cross_target2)      │
│ • A6: Selectivity Index + bootstrap 95% CI                   │
│ • B8: multi-pose ensemble (snap 0)                           │
│ → SI > 1.5 + bootstrap CI lower > 1.5 → selective cavity     │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌─ Phase 6 ─ 합성 레시피 ──────────────────────────────────────┐
│ • A7: target별 MIP + NIP recipe 동시 생성 (대조군 포함)      │
│ • B10: initiator mole percent (예: 11.21 mg Irgacure)        │
│ • pH 호환성 (모든 모노머 stable 범위 교집합)                 │
│ • Q-e reactivity (vinyl monomer pair r₁·r₂ ∈ [0.1, 10])      │
│ • EV / glycan heterogeneity / serum cross-reactivity 경고    │
│ → recipe JSON + 합성 protocol (sol-gel / radical / surface)  │
└──────────────────────────────────────────────────────────────┘
                              ↓
                     verify_phase.py {1..6}
              (3-level: code · algorithm · physics/chem)
                              ↓
                  results/reports/phaseN_*.{json,md}
```

---

## 3. Phase 1 — 에피토프 추출 & 구조 준비

**파일**: `code/pipeline/phase1_epitope_prep.py`

### 3.1 구조 source

- **CD63**: AlphaFold DB API [25] (UniProt P08962). pLDDT > 70 검증.
- **CD81 / CD9**: RCSB PDB (5TCX, 6K4J 결정 구조).
- **Missing side chain 수정**: pdbfixer [31].

### 3.2 도킹 receptor vs 합성 template

| 용도 | 추출 영역 | 길이 | Phase |
|------|----------|------|-------|
| **도킹 receptor** | ECL2 전체 (CCG 디설피드 포함) | ~90 잔기 | Phase 2, 3 |
| **합성 template** | Head 16-mer (실제 펩타이드 주문 영역) | 16 잔기 | Phase 4, 5 |

ECL2 전체를 사용하는 이유: head만 자르면 CCG 모티프의 disulfide bond가 끊어져 구조 불안정. ECL2 전체를 도킹 receptor로 쓰되 **grid center는 head 16-mer 중심**에 맞춤으로써 도킹이 head 영역으로 유도된다.

### 3.3 A1: Multi-epitope candidate auto-selection

CD63는 글리코실레이션 site가 3개여서 epitope 후보가 4개 존재:
- `head_canonical` (155–170, glycan 없음)
- `head_glyco_N130` (122–137, N130 포함)
- `head_glyco_N150` (143–158, N150 포함)
- `head_glyco_N172` (157–172, N172 포함) ← **현재 선택됨**

각 후보를 다음 metric으로 평가하여 best 선택:
- **B1 SASA** (Shrake–Rupley): 표면 노출도 (≥ 50 Ų/residue → accessible)
- **B2 GRAVY** (Kyte–Doolittle): 친수성-소수성 균형 (-1.0 < GRAVY < +1.0 → balanced)
- **Protrusion**: 단백질 평균 표면에서의 돌출
- **Glycan exposure**: 글리칸 site의 SASA

### 3.4 Protonation & 검증

- **PROPKA 3.5** [4]: pH 7.4 (PBS) 상태 결정. His/Cys 양성자화 자동 할당.
- **BLAST** [2]: 16-mer가 인간 프로테옴에서 고유한지 확인. >70% identity 다른 단백질이 있으면 경고.

### 3.5 Stability MD

- **GROMACS** [18] + **amber99sb-ildn** [20] + **TIP3P** [19] + 0.15 M NaCl
- 20 ns NPT (300 K, 1 bar)
- 검증: backbone RMSD < 3.0 Å [3]

### 3.6 A2: K-medoids conformer extraction

20 ns trajectory에서 RMSD-based K-medoids clustering → 5개 대표 conformer 추출 (pairwise RMSD > 1 Å로 다양성 보장). 각 conformer를 OpenBabel [29]/ADFR로 receptor PDBQT 변환 → Phase 2에서 6개 receptor(원본 + 5 conformer)에 도킹하여 **수용체 유연성 반영**.

### 3.7 출력

```
results/phase1/{target}/
├── AF_P08962.pdb           # 원본 (CD63만 AlphaFold)
├── {target}_ecl2.pdb       # ECL2 전체 (도킹)
├── {target}_head.pdb       # head 16-mer (합성 template)
├── {target}_md/md.{gro,xtc}# 20 ns stability MD
├── conf_{0..4}.pdb         # K-medoids 5 conformer
└── {target}_receptor.pdbqt # AutoDock 포맷
```

---

## 4. Phase 2 — Single Monomer Docking (SMD)

**파일**: `code/pipeline/phase2_smd.py`

### 4.1 도킹 엔진

- **AutoDock4** [5] Lamarckian Genetic Algorithm (LGA)
- GPU 가속: **AutoDock-GPU** [6] (동일 force field + scoring, ~100–350× 속도)
- GA runs = 50, population = 150, max evals = 2,500,000, grid spacing = 0.375 Å, grid points = 60×60×60

### 4.2 비표준 원자 (Si, B) 처리

AutoDock4 기본 force field에 Si/B 미포함. 3-step pipeline으로 해결:

1. **PDBQT 생성**: SMILES → Si→S, B→C proxy 치환 → meeko로 PDBQT 생성 → 원래 원자 타입 복원
2. **커스텀 파라미터 파일** `AD4_parameters_custom.dat`: UFF [8] 기반 — Si_3 (R_ii=4.295 Å, ε=0.402 kcal/mol), B_3 (R_ii=4.083 Å, ε=0.180 kcal/mol)
3. **AutoDock-GPU 호출**: `--derivtype Si=S/B=C` + `--parameter_file` 동시 지정

### 4.3 Ensemble docking

24 monomer × 3 target × 6 receptor = **최대 432 도킹**. 각 (monomer, target) 쌍에 대해 6 receptor 중 best BE 선택 → `be_matrix[target][monomer]`.

### 4.4 A3: Multi-pose clustering

각 DLG 파일을 파싱하여 RMSD ≤ 2.0 Å, ≥3 멤버 클러스터 식별. 최대 5개 클러스터 → `pose_clusters[monomer]` (대안 binding mode 보존).

### 4.5 B3: Decoy baseline (enrichment factor)

5종 decoy 분자 (랜덤 small molecule)에 동일 docking → real monomer BE 분포 vs decoy 분포 비교. **Enrichment Factor (EF) > 1.5** → real이 random보다 유의미하게 강함 → 도킹 결과 신뢰 가능.

### 4.6 필터링

- BE ≤ -2.0 kcal/mol (유의미한 결합)
- 상위 12개를 Phase 3로 전달
- ΔΔG selectivity는 Phase 3 NSGA-II + cross-docking에서 평가 (SMD에서는 individual 결합만)

---

## 5. Phase 3 — Multi-Monomer Simultaneous Docking + NSGA-II

**파일**: `code/pipeline/phase3_mmsd.py`

### 5.1 MMSD 프로토콜 (Rajpal 2024 [9])

Sequential docking — Step k에서 monomer k를 **(에피토프 + 이전 k-1 모노머 pose 병합 receptor)** 에 도킹. 이전 모노머가 차지한 공간을 피해 새 위치에 도킹되므로 multi-monomer **synergy / interference** 평가가 가능.

```python
mmsd_sum    = Σ BE_k                    # 모든 step BE 합
smd_sum     = Σ SMD_BE_k                # individual docking 합
delta_sum   = mmsd_sum - smd_sum
              < 0 → synergy (cooperative)
              > 0 → interference (steric clash)
```

### 5.2 C2: NSGA-II 3-objective Pareto optimization

기존 [9]는 affinity 단일 목적이었으나, 본 파이프라인은 **NSGA-II** (Deb 2002, pymoo 구현)로 3개 목표를 동시 최적화:

| Objective | 정의 | Direction |
|-----------|------|-----------|
| **Affinity** | `mmsd_per_monomer = mmsd_sum / n_monomers` | 최소화 (음수일수록 강함) |
| **Selectivity** | `-selectivity_score` (Phase 2 ΔΔG 또는 cross-MMSD ΔΔG) | 최소화 |
| **Synthesizability** | `-(synth_score / 10)` (boronate bonus, pH window, Q-e 점수) | 최소화 |

- Pop size = 20, n_gen = 15 → ~300 evals (cache hit ~ 80 unique)
- Repair: `RoundingRepair()` (integer chromosome → 모노머 인덱스)
- 결과: **Pareto front** (≥20 non-dominated 조합)

### 5.3 Cross-MMSD ΔΔG Selectivity Penalty (본 파이프라인의 핵심)

NSGA-II 각 evaluation마다, 같은 monomer combo를 **다른 target receptor에도 cross-docking**하여 실제 cooperative binding 기반 ΔΔG를 산출:

```
own_sum  = MMSD sum on own target
off_mean = mean(MMSD sum on each off-target)
ΔΔG     = own_sum - off_mean              # negative = own preferred (selective)

penalty = w · max(0, ΔΔG - threshold)
        = 0.5 · max(0, ΔΔG - (-1.0))
        = 0.5 · max(0, ΔΔG + 1.0)
```

| ΔΔG (kcal/mol) | 해석 | penalty |
|---|---|---|
| ≤ -1.0 | 충분히 선택적 | 0 |
| -0.5 | 거의 동등 | 0.25 |
| 0 | 비선택적 | 0.50 |
| +1.0 | anti-selective | 1.00 |

- `threshold = -1.0` kcal/mol: RT (298 K) × 1.5 ≈ "5배 이상 binding probability 차이"
- `w = 0.5`: affinity와 동등 가중 (Garcia-Ortegon 2022 [32])

NSGA-II `_evaluate()` 내부에서 `selectivity_score`를 cross-MMSD ΔΔG 기반으로 **override** → Pareto front 자체가 선택적 방향으로 진화한다.

비용: MMSD 1회 → 3회 (own + 2 cross). 약 3시간 추가.

근거 [9,32,33]:
- **Rajpal 2024** [9]: MMSD에 selectivity 미포함 → 본 파이프라인의 contribution
- **Garcia-Ortegon 2022** [32]: DOCKSTRING hinge penalty `f(s) = score + min(off_score, threshold)`
- **Mestres 2011** [33]: Selectivity entropy (Shannon-Boltzmann)

### 5.4 가교제 자동 선택

MMSD 마지막 step에서 호환 가교제 전부 도킹 → BE 최소 선택. 호환성은 **polymerization metadata** 기반:

```python
has_silane  = any monomer.polymerization == "silane"
has_radical = any monomer.polymerization in {"vinyl", "catechol"}

if silane and not radical → [TEOS, TMOS]
if radical and not silane → [MBAAm, EGDMA, DVB, TRIM]
if both                   → []  # mixed: cannot one-pot synthesize
```

- `surface` 모노머 (APBA, FPBA): polymerization 안 함, 다른 매트릭스에 grafting → crosslinker 선택에 영향 없음

### 5.5 B5: DFT validation hook

Top-N PC에 대해 Psi4 M06-2X/def2-TZVP single point 계산을 위한 hook 설치 (현재 stub). xTB fallback도 framework 마련.

### 5.6 출력

```
results/phase3/
├── phase3_mmsd_results.json    # method, pareto_front, top_pcs, all_results
├── phase3_{target}_bo.png      # Pareto front visualization
├── monomers/                   # 가교제 PDBQT cache
└── {target}/{pc_id}/
    ├── step{k}_{monomer}/      # 각 step의 docking output
    └── cross_{other_target}/   # cross-MMSD output
```

---

## 6. Phase 4 — Pre-polymerization MD

**파일**: `code/pipeline/phase4_md_validation.py`

### 6.1 시스템 구축

- **Template**: Head 16-mer (실제 합성 조건과 일치)
- **모노머 배치**: Phase 3 optimal combo의 functional × 5 copy + crosslinker × 5 copy = **25개 모노머**
- **배치 방식**: 단백질 중심 반경 3.1–4.1 nm 구 껍질 랜덤 (min sep 1 nm) — PACKMOL [27] 방식 [12]
- **Force field**:
  - Protein: amber99sb-ildn [20]
  - Vinyl monomers: GAFF2 (acpype [28])
  - Si-containing: **PolCA** [13] (GAFF2 + Si LJ + bond eq distances Si-C=0.186 nm, Si-O=0.164 nm)

### 6.2 MD 프로토콜

| 단계 | 조건 |
|------|------|
| Solvation | Cubic box, 0.5 nm padding, TIP3P, 0.15 M NaCl |
| EM | Steepest descent 50,000 steps, Fmax < 1000 kJ/mol/nm |
| NVT eq | 100 ps, V-rescale 300 K |
| NPT eq | 100 ps, Parrinello-Rahman 1 bar |
| **Production** | **350 ns** (or 20 ns quick mode), dt=2 fs, PME, LINCS h-bonds, GPU |

### 6.3 분석 (후반 50%, stride 100)

| 지표 | 방법 | 근거 |
|------|------|------|
| Contact frequency | head 6 Å 이내 접촉 프레임 비율 (MDAnalysis [26]) | 문헌 표준 [12] |
| Residence time | 연속 접촉 프레임 수 | 결합 안정성 |
| **EBN** (Effective Binding Number) | 프레임당 동시 접촉 모노머 최대값 | Yuan 2024 [14] |
| **HBNMax** | HydrogenBondAnalysis (d-a 3.5 Å, angle 150°) | Yuan 2024 [14] |
| Per-atom RDF | OH / NH / CO 기능기별 g(r) peak | Yuan 2024 [14] |
| Monomer pair distance | 모노머 간 min distance | Cavity compactness |
| Crosslinker proximity | 가교제-모노머 min dist < 10 Å | Rajpal 2023 [10] |
| RMSD/RMSF/Rg/H-bond | gmx rms/rmsf/gyrate/hbond | 구조 안정성 |
| **MM-GBSA** + per-residue decomp | gmx_MMPBSA (igb=5, saltcon=0.15, idecomp=2) [7,15] | ΔG + hotspot |
| Convergence | Q1→Q4 contact freq diff < 10% | Polania 2024 [12] |
| **PBC centering** | `gmx trjconv -pbc mol -center` | 시각화·분석 정확도 |

### 6.4 A5/B7: Solvent sweep

`MD_SOLVENT_SWEEP = True` 시 water / EtOH-water 3:1 / DMSO 3개 solvent에서 MD 반복 → best solvent 자동 선택 (cavity compactness, RMSD, MM-GBSA 기준).

### 6.5 B6: Ratio sweep

`MD_RATIO_SWEEP = True` 시 functional:crosslinker 비율 5개 preset (3:7 / 4:6 / 5:5 / 6:4 / 7:3) 비교.

### 6.6 합성 비율 결정 (EBN 기반)

```
ratio_i = EBN_i / min(EBN_j for all j)
```

EBN이 높은 모노머 = template 표면에 결합 site 많음 → 더 많이 넣어 모든 site 포화. Crosslinker는 functional 합과 동량 [14].

---

## 7. Phase 5 — VIP Cavity Rebinding

**파일**: `code/pipeline/phase5_rebinding.py`

VIP (Virtually Imprinted Polymer) [11] 방식 — 중합을 position restraint로 근사하여 cavity 형성과 rebinding을 한 simulation에서 검증.

### 7.1 Snapshot 선택

Phase 4 trajectory 후반 50%에서 **균등 간격 10 frame** 추출. Cherry-picking 방지 — 실제 중합은 UV/열에 의해 랜덤 시점에 발생하므로 "최적 프레임" 선택은 과적합.

### 7.2 중합 근사

모노머 heavy atoms에 **harmonic position restraint** (k = 1000 kJ mol⁻¹ nm⁻², GROMACS standard) → "현재 위치에서 cross-linking되어 polymer network에 잠긴" 상태.

### 7.3 Template removal test (10 ns)

- 모노머 restrained + template + 물 자유
- Template이 cavity 이탈 (RMSD > 5 Å) → "removable" = template 세척 가능 = **적정 결합 강도**
- 이탈 못 함 → "stuck" = 결합 너무 강함 = 실제 합성에서 template 제거 어려움 → IF 저하

### 7.4 Rebinding MD (50 ns × {own, cross_t1, cross_t2})

동일 cavity에 own/다른 head 재삽입 후 50 ns. Cross-rebinding으로 selectivity 정량화:

```
SI (Selectivity Index) = RMSD_other / RMSD_own
  SI > 1.5            → selective
  SI 1.0 – 1.5        → weak selectivity
  SI < 1.0            → cross-reactive
```

Welch's t-test로 own vs other RMSD 차이의 유의성 (p < 0.05).

### 7.5 A6: Bootstrap CI for SI

SI는 단일 통계량 → 신뢰구간이 필요. **Bootstrap resampling** (1000 iter) → 95% CI:
- CI lower > 1.5 → 통계적으로 robust한 selectivity
- CI 1–2 polish: SI 점추정이 1.5여도 CI가 0.8–2.5라면 신뢰 못 함

### 7.6 B8: Multi-pose ensemble (snap 0)

대안 binding mode 확인. Snap 0에서 conformer × replica 다중 rebinding → cavity가 단일 mode인지 multiple mode 허용인지 평가.

### 7.7 Multi-restart fallback

특정 snap이 발산하면 (RMSD > 10 Å within 1 ns) seed 바꿔 최대 3회 재시도 — extension 안 함, 새 run으로 재시작.

### 7.8 출력

```
results/phase5/{target}/
├── snapshot_{0..9}/
│   ├── rebind_own/md/         # 50 ns own rebinding
│   ├── rebind_{other1}/md/    # cross rebinding
│   ├── rebind_{other2}/md/
│   └── removal_test/md/       # 10 ns removal
├── snapshot_0/multipose/      # B8 multi-pose (snap 0만)
└── phase5_rebinding_results.json
```

---

## 8. Phase 6 — 합성 레시피

**파일**: `code/pipeline/phase6_recipe.py`

### 8.1 Trigger 조건

Phase 5에서 **SI > 1.5 + bootstrap CI lower > 1.5 + ≥3/10 snapshot 재현성**을 통과한 target에 대해서만 recipe 생성. 실패 target은 제외.

### 8.2 A7: MIP + NIP 동시 생성

Target별로 **MIP와 대조 NIP(Non-Imprinted Polymer) 한 파일에** 생성:
- MIP: template 포함 합성
- NIP: 동일 조성, template 제외 → background binding 측정 → IF (Imprinting Factor) 산출 가능

### 8.3 합성 protocol 분기

| Polymerization | Initiator | 조건 | 비고 |
|----------------|-----------|------|------|
| **silane** (sol-gel) | TEOS/TMOS | RT 16 h | Stöber-like |
| **vinyl** (free-radical) | Irgacure 819 / AIBN | UV 365 nm 4 h / 60 °C 12 h | DA: catechol auto-ox |
| **surface graft** | aldehyde-amine Schiff base | RT 2 h | APBA/FPBA |
| **mixed** (silane + vinyl) | solid-phase glass bead [3,30] | 2-step | Sehit 2024 |

### 8.4 B10: Initiator mole percent → 실제 mg

```
initiator_mol = 0.01 × total_vinyl_mol
initiator_mass_mg = initiator_mol × M_initiator × 1000
```

예: 100 mg vinyl monomers + Irgacure 819 (M=418) → **0.42 mg** Irgacure.

### 8.5 pH 호환성

각 모노머 stable pH 범위의 **교집합**이 합성 pH window. 비어 있으면 synthesizable=False.

### 8.6 Q-e reactivity (vinyl 조합용)

각 vinyl monomer pair의 reactivity ratio r₁·r₂ 계산. **[0.1, 10] 범위**이면 ideal copolymerization 가능 — 그 외엔 alternating / block 경향 → warning.

### 8.7 Warning labels

- **EV context**: tetraspanin target일 때 — exosome 시료에서 다른 EV marker와의 cross-reactivity 가능성
- **Glycan heterogeneity**: CD63 N172 dual-imprinting 시 — 글리칸 microheterogeneity로 cavity 균일성 보장 어려움
- **Serum cross-reactivity**: boronate (APBA/VPBA/FPBA/AAPBA) 사용 시 — 혈청 당단백질과 cross-react

### 8.8 출력

```
results/phase6/
├── phase6_recipes.json
├── CD63_recipe.txt           # MIP + NIP 한 파일
├── CD81_recipe.txt
├── CD9_recipe.txt
└── synthesis_protocol.txt    # 통합 protocol
```

---

## 9. 모노머 및 가교제 라이브러리

### 9.1 Silane (14종, sol-gel)

| ID | 화합물 | 주요 상호작용 |
|----|--------|--------------|
| PTES | Phenyltriethoxysilane | π-π, 소수성 |
| APTES | (3-Aminopropyl)triethoxysilane | H-bond D, electrostatic |
| APTMS | (3-Aminopropyl)trimethoxysilane | H-bond, electrostatic |
| UPTMS | 3-Ureidopropyltrimethoxysilane | Multi H-bond (urea D+A) |
| MPTMS | (3-Mercaptopropyl)trimethoxysilane | Thiol-Cys, H-bond |
| IBTES | Isobutyltriethoxysilane | 소수성, alkyl |
| MTMS | Methyltrimethoxysilane | 소수성 |
| EDTMS | TMS-propyl ethylenediamine | Chelate H-bond |
| ICTES | TES-propyl isocyanate | 공유결합 (Lys ε-NH₂) |
| VTMS | Vinyltrimethoxysilane | 소수성 + 가교 가능 |
| GPTMS | (3-Glycidyloxypropyl)TMS | Epoxy 공유 |
| DIDMS | Dimethyldimethoxysilane | 강한 소수성 |
| CETES | 3-Cyanopropyl-TES | 약한 H-bond A (CN) |
| TTMS | p-Tolyl-TMS | π-π + 소수성 |

### 9.2 Vinyl / Acrylic (10종, free-radical)

| ID | 화합물 | 상호작용 |
|----|--------|---------|
| AA | Acrylic acid | Electrostatic (Lys/Arg/His) |
| MAA | Methacrylic acid | Electrostatic + 소수성 |
| AAm | Acrylamide | H-bond (D+A) |
| NIPAm | N-Isopropylacrylamide | H-bond + 소수성 |
| 4VIm | 4-Vinylimidazole | π-π, H-bond, His mimic |
| HEMA | 2-Hydroxyethyl methacrylate | H-bond (OH) |
| TBAm | N-tert-Butylacrylamide | 소수성 |
| VPBA | 4-Vinylphenylboronic acid | Glycan diol + 가교 |
| AAPBA | N-Acryloyl-3-aminophenylboronic acid | Glycan diol + 가교 |

### 9.3 Catechol (2종, auto-ox pH > 7)

| DA | Dopamine | Multi H-bond, catechol |
| NE | Norepinephrine | DA + extra OH |

### 9.4 Surface-grafted boronate (2종, polymerization NOT)

| APBA | 3-Aminophenylboronic acid | Glycan diol; amine으로 surface grafting |
| FPBA | 4-Formylphenylboronic acid | Glycan diol; aldehyde Schiff base [3] |

### 9.5 Crosslinker (6종, 자동 선택)

| 가교제 | 유형 | 관능기수 | 호환 모노머 |
|--------|------|---------|-----------|
| TEOS | silane | 4 | silane only |
| TMOS | silane | 4 | silane only |
| MBAAm | vinyl | 2 | vinyl/catechol |
| EGDMA | vinyl | 2 | vinyl/catechol |
| DVB | vinyl | 2 | vinyl |
| TRIM | vinyl | 3 | vinyl/catechol |

### 9.6 Polymerization 호환성 (one-pot 합성)

```
silane   ↔ silane                    (sol-gel only)
vinyl    ↔ vinyl, catechol           (radical)
catechol ↔ vinyl, catechol
surface  ↔ any matrix                (grafting)
epoxy    ↔ any matrix                (side-chain covalent)
```

**Mixed silane + radical은 one-pot 불가** — solid-phase 2-step만 허용 [3,30].

---

## 10. 핵심 알고리즘 업데이트 (A/B/C 시리즈)

이 파이프라인은 17개 업데이트 기능을 통합한다. 모두 `config.py`에서 flag로 토글 가능 (기본 ON).

### A 시리즈 — Algorithm 개선

| 코드 | 기능 | 위치 |
|------|------|------|
| **A1** | Multi-epitope candidate auto-evaluation | Phase 1 |
| **A2** | K-medoids conformer extraction (5종) | Phase 1 |
| **A3** | Multi-pose docking clustering | Phase 2 |
| **A4** | Bayesian Optimization (GP) for MMSD (fallback) | Phase 3 |
| **A5** | Solvent sweep (water/EtOH/DMSO) | Phase 4 |
| **A6** | Bootstrap 95% CI for Selectivity Index | Phase 5 |
| **A7** | Per-CD MIP + NIP recipe in one file | Phase 6 |

### B 시리즈 — Benchmark 강화

| 코드 | 기능 | 위치 |
|------|------|------|
| **B1** | SASA-based accessibility check | Phase 1 |
| **B2** | GRAVY hydrophilicity balance | Phase 1 |
| **B3** | Decoy baseline + enrichment factor | Phase 2 |
| **B5** | DFT validation hook (Psi4 stub) | Phase 3 |
| **B6** | Functional:crosslinker ratio sweep | Phase 4 |
| **B7** | Multi-solvent comparison | Phase 4 |
| **B8** | Multi-pose rebinding ensemble | Phase 5 |
| **B9** | FEP framework (stub) | Phase 5 |
| **B10** | Initiator mole percent → 실제 mg | Phase 6 |

### C 시리즈 — Computational 강화

| 코드 | 기능 | 위치 |
|------|------|------|
| **C2** | **NSGA-II 3-objective Pareto** + **cross-MMSD ΔΔG penalty** | Phase 3 |

---

## 11. 실행 방법

### 11.1 기본 사용

```bash
conda activate MIPscreen

cd code

# 전체 6-phase 파이프라인 (resume 자동)
python3 -m pipeline.run_pipeline --target CD63 CD81 CD9

# Fresh run (기존 결과 무시)
python3 -m pipeline.run_pipeline --target CD63 CD81 CD9 --fresh

# 특정 Phase만
python3 -m pipeline.run_pipeline --phase 3 --target CD63
python3 -m pipeline.run_pipeline --phase 5 --target CD63 CD81 CD9

# Quick MD (디버깅, 20 ns)
python3 -m pipeline.run_pipeline --phase 4 --quick-md --target CD63
```

### 11.2 검증 호출

```bash
# Phase N 완료 후 자동 검증
python3 verify_phase.py 1    # Level 1 + 2 + 3 보고서 생성
python3 verify_phase.py 3
# → results/reports/phase{N}_verification.json
# → results/reports/phase{N}_summary.md
```

### 11.3 디렉토리 구조

```
Monomer_screening_in_Bio/
├── README.md
├── environment.yml
├── code/
│   ├── verify_phase.py              # 3-level verifier
│   └── pipeline/
│       ├── config.py                # 모든 파라미터 + 17 feature flag
│       ├── run_pipeline.py          # 오케스트레이터
│       ├── phase1_epitope_prep.py
│       ├── phase2_smd.py
│       ├── phase3_mmsd.py           # NSGA-II + cross-MMSD
│       ├── phase4_md_validation.py
│       ├── phase5_rebinding.py
│       ├── phase6_recipe.py
│       ├── generate_report.py
│       ├── utils_structure.py       # PDBQT, grid, RDKit
│       ├── utils_autodock.py        # AutoDock4-GPU 호출
│       ├── utils_gromacs.py         # GROMACS wrapper
│       └── utils_analysis.py        # SASA, GRAVY, EBN, SI, bootstrap
└── results/
    ├── phase{1..6}/                 # phase별 raw output
    ├── reports/
    │   ├── phase{N}_verification.json
    │   ├── phase{N}_summary.md
    │   └── pipeline_verification_plan.md
    └── logs/                        # phase{N}_*.log
```

---

## 12. 검증 프레임워크 (3-Level)

각 phase 완료 후 `verify_phase.py N`이 자동 보고서 생성. 모든 Level이 통과하면 `decision: proceed_to_phase_(N+1)`.

### 12.1 Level 1 — Code integrity

- Process exit code 0, no Traceback
- 모든 expected output 파일 생성
- JSON 파싱 가능
- Critical warnings (NaN, empty slice, OOM) 없음

### 12.2 Level 2 — Algorithm correctness

- 새 기능 실제 호출 (A1, A2, A3, A6, A7, B1, B2, B3, B5, B6, B7, B8, B10)
- Config flag → 행동 일치 (예: `nsga2` default → NSGA-II 호출)
- Fallback chain 정상 (pymoo 없으면 BO, BO 없으면 greedy)
- 의도된 metric 출력 (Pareto front, bootstrap CI, GRAVY 등)

### 12.3 Level 3 — Physics / Chemistry validity

- **MD centering**: protein COM이 box 중심 ±1 nm
- **No PBC split**: protein span / box ratio < 0.95
- **Stability**: peptide head RMSD < 3 Å
- **Convergence**: Q1→Q4 contact drift < 10%, RMSD drift < 1.5 Å
- **Energy validity**: ∞ / NaN 없음
- **Polymerization compatibility**: top PC가 silane-only or vinyl-only (mixed 없음)
- **pH window**: 모든 모노머 호환 pH 범위 ≠ ∅
- **Q-e**: vinyl pair r₁·r₂ ∈ [0.1, 10]

### 12.4 진행 결정 protocol

```
verify → report

if Level 1 fail (코드 에러):
    → STOP, debug, restart phase

if Level 2 fail (기능 미작동):
    → 원인 (config? dependency? wiring?) → fix → restart

if Level 3 fail (물리 문제):
    → 분석:
       - PBC split → trjconv -pbc mol -center
       - Drift → MD 연장 또는 multi-restart
       - 통계 부족 → n_snapshots 증가

else (전부 통과):
    → save report → proceed to phase N+1
```

---

## 13. 핵심 파라미터 요약

| Phase | 파라미터 | 값 | 근거 |
|-------|---------|-----|------|
| 1 | 에피토프 길이 | 16 잔기 | Teixeira 2021 [1] |
| 1 | 도킹 receptor | ECL2 전체 | Disulfide 유지 |
| 1 | Ensemble conformer | 5 (K-medoids) | 수용체 유연성 |
| 1 | Stability RMSD threshold | < 3 Å | Yuan 2024 [14] |
| 2 | GA runs | 50 | AutoDock4 default |
| 2 | BE threshold | ≤ -2.0 kcal/mol | 유의미한 결합 |
| 2 | Top N for Phase 3 | 12 | BE 상위 |
| 2 | Decoy EF threshold | > 1.5 | Real > random |
| 3 | Optimizer | NSGA-II (default) | C2: 다목적 |
| 3 | Pop size × n_gen | 20 × 15 = 300 | pymoo |
| 3 | 조합 크기 | 2–6 (자동) | 가변 |
| 3 | **SELECTIVITY_WEIGHT** | **0.5** | **Garcia-Ortegon 2022 [32]** |
| 3 | **SELECTIVITY_DDG_THRESHOLD** | **-1.0 kcal/mol** | **≈ 1.5·RT (298 K)** |
| 4 | Template | Head 16-mer | 합성 조건 동일 |
| 4 | 모노머 수 | 25 (각 type 5) | 통계 + 효율 |
| 4 | MD 시간 | 350 ns (production) | 평형 도달 |
| 4 | Contact cutoff | 6 Å | vdW 포함 |
| 4 | H-bond | d-a 3.5 Å, angle 150° | Yuan 2024 [14] |
| 4 | Convergence | Q1→Q4 diff < 10% | Polania 2024 [12] |
| 4 | MM-GBSA | igb=5, saltcon=0.15, idecomp=2 | Kumar 2024 [15] |
| 5 | Snapshot | 균등 10개 | Cherry-picking 방지 |
| 5 | Restraint k | 1000 kJ/mol/nm² | Zink 2018 [11] |
| 5 | Rebinding 기준 | RMSD < 5 Å | head 직경 1/3 |
| 5 | SI threshold | > 1.5 | Mohsenzadeh 2024 [16] |
| 5 | Bootstrap CI | 1000 iter, 95% | 표준 통계 |
| 5 | 재현성 | ≥ 3/10 snapshot | A6 + B8 |
| 6 | Initiator | 1 mol% of vinyl | B10 |

---

## 14. 설치 및 환경

### 14.1 환경 구축

```bash
# 1. Conda 환경 생성 (MIPscreen — 17 dependency)
conda env create -f environment.yml
conda activate MIPscreen

# 2. GROMACS GPU 빌드 (필수)
# https://manual.gromacs.org/current/download.html
# config.py의 GMX_BIN 경로 수정

# 3. AutoDock-GPU (Phase 2/3 가속)
# https://github.com/ccsb-scripps/AutoDock-GPU
# config.py의 AUTODOCK_GPU_BIN 경로 수정

# 4. (Optional) PolCA force field (Si)
# https://github.com/MJorge78/polca

# 5. 설치 확인
python -c "import rdkit, MDAnalysis, pymoo, skopt, sklearn_extra; print('OK')"
gmx --version
autodock_gpu_128wi --version
```

### 14.2 주요 Python 의존성

| 패키지 | 용도 | Phase |
|--------|------|-------|
| **rdkit** | SMILES → 3D, PDBQT 변환 | 1, 2, 3 |
| **mdtraj** | RMSD, SASA, K-medoids 좌표 | 1 |
| **MDAnalysis** | contact, RDF, H-bond | 4, 5 |
| **scikit-learn-extra** | K-medoids clustering | 1 |
| **scikit-optimize** | Bayesian Optimization (A4 fallback) | 3 |
| **pymoo** | NSGA-II multi-objective (C2 default) | 3 |
| **gmx_MMPBSA** | MM-GBSA + per-residue decomp | 4 |
| **propka** | pH-based protonation | 1 |
| **pdbfixer** | missing side chain | 1 |
| **meeko** | AutoDock PDBQT 생성 | 2, 3 |
| **openbabel** | 좌표 변환 | 1, 2 |
| **biopython** | NCBI BLAST, PDB I/O | 1 |
| **matplotlib** | Pareto front, BO plot | 3, 4 |

### 14.3 테스트 환경

- OS: Ubuntu 22.04 (WSL2 on Win11)
- Python: 3.13
- GROMACS: 2025.2 (GPU build, NVIDIA CUDA)
- AutoDock-GPU: v1.6
- GPU: NVIDIA RTX 4070 Ti (12 GB VRAM)
- Total fresh-run 시간: ~10 일 (CD63 + CD81 + CD9 전 phase)

---

## 15. 참고 문헌

### MIP 에피토프 설계

[1] Teixeira SPB et al. "Epitope-imprinted polymers: Design principles." *Science Advances* 2021;7:eabi9884.

[2] Bossi AM et al. "MIP by epitope imprinting: bioinformatics." *Anal. Bioanal. Chem.* 2021;413:6101-6112.

[3] Sehit E et al. "Computationally Designed Epitope-Mediated Imprinted Polymers." *ACS Sensors* 2024;9:1831-1841.

### 도킹 + 모노머 스크리닝

[4] Li H et al. "PROPKA pKa prediction." *Proteins* 2005;61:704-721.

[5] Morris GM et al. "AutoDock4." *J. Comput. Chem.* 2009;30:2785-2791.

[6] Santos-Martins D et al. "AutoDock-GPU." *J. Chem. Theory Comput.* 2021;17:1060-1073.

[7] Sullivan MV et al. "Rational Design of Selective MIPs for Proteins." *J. Phys. Chem. B* 2019;123:5432-5443.

[8] Rappe AK et al. "UFF force field." *JACS* 1992;114:10024-10035.

[9] Rajpal S et al. "Multi-monomer simultaneous docking for epitope imprinting." *Sci. Rep.* 2024;14:23057.

[10] Rajpal S, Mizaikoff B. "In silico multi-monomer combinations." *J. Mater. Chem. B* 2022;10:6618-6626.

### VIP + MD 시뮬레이션

[11] Zink S et al. "Virtually imprinted polymers (VIPs)." *Phys. Chem. Chem. Phys.* 2018;20:13145-13152.

[12] Polania LC, Jiménez VA. "MD simulations in pre-polymerization mixtures for peptide recognition." *J. Mol. Model.* 2024;30:266.

[13] Jorge M et al. "PolCA force field for organosilicon." *ACS Phys. Chem. Au* 2021;1:34-49.

### 정량적 분석

[14] Yuan J et al. "Computational and Experimental Comparison of MIPs — EBN, HBNMax." *Molecules* 2024;29:4236.

[15] Kumar MD et al. "Computational modelling of electropentamer for MIP — per-residue MMPBSA decomposition." *J. Mol. Graph. Model.* 2024;128:108715.

[16] Mohsenzadeh E et al. "Design of MIPs using computational methods." *WIREs Comput. Mol. Sci.* 2024;14:e1713.

[17] MIP-PhAC Dataset. "MD/MM-PBSA and DFT Resources for MIP Design." *Data* 2025;10:205.

### 소프트웨어 + 방법론

[18] Abraham MJ et al. "GROMACS." *SoftwareX* 2015;1-2:19-25.

[19] Jorgensen WL et al. "TIP3P water model." *J. Chem. Phys.* 1983;79:926-935.

[20] Lindorff-Larsen K et al. "Amber ff99SB-ILDN side-chain torsion." *Proteins* 2010;78:1950-1958.

[21] Bussi G et al. "V-rescale thermostat." *J. Chem. Phys.* 2007;127:014102.

[22] Parrinello M, Rahman A. "Parrinello-Rahman barostat." *J. Appl. Phys.* 1981;52:7182-7190.

[23] Darden T et al. "Particle mesh Ewald (PME)." *J. Chem. Phys.* 1993;98:10089-10092.

[24] Hess B et al. "LINCS constraint solver." *J. Comput. Chem.* 1997;18:1463-1472.

[25] Jumper J et al. "AlphaFold." *Nature* 2021;596:583-589.

[26] Michaud-Agrawal N et al. "MDAnalysis." *J. Comput. Chem.* 2011;32:2319-2327.

[27] Martínez L et al. "PACKMOL." *J. Comput. Chem.* 2009;30:2157-2164.

[28] Sousa da Silva AW, Vranken WF. "ACPYPE." *BMC Res. Notes* 2012;5:367.

[29] O'Boyle NM et al. "Open Babel." *J. Cheminform.* 2011;3:33.

[30] Poma A et al. "Solid-phase nanoMIP synthesis." *Adv. Funct. Mater.* 2013;23:5537-5543.

[31] Eastman P et al. "OpenMM 7." *PLOS Comput. Biol.* 2017;13:e1005659.

### Selectivity 방법론 (본 파이프라인 핵심)

[32] Garcia-Ortegon M et al. "DOCKSTRING — selectivity scoring." *J. Chem. Inf. Model.* 2022;62:3486-3502.

[33] Mestres J et al. "Selectivity entropy." *BMC Bioinformatics* 2011;12:94.

[34] Deb K et al. "NSGA-II: A fast and elitist multiobjective genetic algorithm." *IEEE Trans. Evol. Comput.* 2002;6:182-197.

### 다목적 최적화

[35] Garcia-Ortegon M et al. "DOCKSTRING benchmarks for ligand design." *JCIM* 2022;62:3486.

[36] Frazier PI. "Bayesian Optimization tutorial." arXiv:1807.02811 (2018).

### 실험 검증

[37] Kowalczyk A et al. "SPR and QCM-D for CD9/CD63/CD81." *Anal. Chem.* 2023;95:9520-9530.

---

*Last updated: 2026-05-13 — selectivity-aware MMSD (cross-docking ΔΔG penalty) added to Phase 3*
