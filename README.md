# Epitope-MIP Computational Screening Pipeline

엑소좀 테트라스파닌 ECL2(CD63 / CD81 / CD9)를 선택적으로 인식하는 에피토프-각인 MIP(Molecularly Imprinted Polymer)의 최적 모노머 조합을 계산화학적으로 스크리닝하는 5-Phase 파이프라인.

---

## 목차

1. [프로젝트 배경](#1-프로젝트-배경)
2. [파이프라인 전체 구조](#2-파이프라인-전체-구조)
3. [Phase 1: 에피토프 추출 및 구조 준비](#3-phase-1-에피토프-추출-및-구조-준비)
4. [Phase 2: Single Monomer Docking (SMD)](#4-phase-2-single-monomer-docking-smd)
5. [Phase 3: Multi-Monomer Simultaneous Docking (MMSD)](#5-phase-3-multi-monomer-simultaneous-docking-mmsd)
6. [Phase 4: MD 검증 및 결합 자유 에너지](#6-phase-4-md-검증-및-결합-자유-에너지)
7. [Phase 5: 합성 레시피 생성](#7-phase-5-합성-레시피-생성)
8. [모노머 라이브러리](#8-모노머-라이브러리)
9. [디렉토리 구조](#9-디렉토리-구조)
10. [실행 방법](#10-실행-방법)
11. [검증 프레임워크](#11-검증-프레임워크)
12. [핵심 파라미터 요약](#12-핵심-파라미터-요약)
13. [참고 논문](#13-참고-논문)
14. [설치 및 환경](#14-설치-및-환경)

---

## 1. 프로젝트 배경

### 1.1 엑소좀과 테트라스파닌

엑소좀(exosome)은 세포에서 분비되는 30–150 nm 세포외 소포체로, 암 진단과 세포 간 통신에 핵심적인 역할을 한다. 엑소좀 표면의 **CD9**, **CD63**, **CD81** 테트라스파닌은 가장 보편적인 바이오마커이지만, 현재 검출은 **항체 기반**으로 비용이 높고 안정성이 낮다.

### 1.2 MIP 접근

Molecularly Imprinted Polymer(분자각인 고분자)는 "플라스틱 항체"로, 합성 비용이 낮고 물리적 안정성이 뛰어나다. **에피토프 각인(epitope imprinting)**은 전체 단백질 대신 표면 노출 펩타이드(에피토프)만 템플릿으로 사용하여, 거대 분자 각인의 한계를 극복한다 (Teixeira et al., Science Advances 2021).

### 1.3 타겟 선정 근거

| 단백질 | PDB | ECL2 Head | N-Glycan | 특성 |
|--------|-----|-----------|----------|------|
| **CD63** | AlphaFold P08962 | 150-195 | **3개** (N130, N150, N172) | 가장 큰 head, heavy glycosylation |
| **CD81** | 5TCX | 157-190 | **0개** | 소수성 head, 비글리코실화 |
| **CD9** | 6K4J | 152-183 | **1개** | 가장 작은 head |

**핵심 차별화**: CD63의 3개 N-glycan은 보론산(APBA) 모노머로 선택적 인식 가능 → CD81(비글리코실화)과 극적 구분.

---

## 2. 파이프라인 전체 구조

```
Phase 1: 에피토프 추출 + 구조 준비
  ├── PDB/AlphaFold 다운로드
  ├── ECL2 head region 추출 (9-16 잔기)
  ├── 물리화학적 특성 분석 (GRAVY, pI, glycosylation)
  ├── AlphaFold pLDDT 품질 검증
  └── (선택) GROMACS 100ns MD 안정성 확인
    ↓ 에피토프 PDB + receptor PDBQT

Phase 2: Single Monomer Docking (SMD) — AutoDock4
  ├── 24개 functional monomer × 3 에피토프 = 72 도킹
  ├── fpocket 결합 부위 예측 (Sullivan 2019)
  ├── Backbone H-bond 분석 (2차 구조 파괴 위험)
  ├── (선택) 10ns contact MD (Sehit 2024)
  ├── 선택도 매트릭스: ΔΔG = BE(target) - mean(BE(non-target))
  └── 필터: BE ≤ -2.0 kcal/mol, ΔΔG ≤ -0.5 kcal/mol
    ↓ 타겟별 후보 모노머 리스트

Phase 3: Multi-Monomer Simultaneous Docking (MMSD) — Rajpal 2024
  ├── 고정 2종 (최고 BE + TEOS 가교제) + 변수 2종 조합
  ├── 4-monomer sequential docking (이전 결과를 수용체에 병합)
  ├── MMSD sum vs SMD sum → 시너지/간섭 판별
  ├── 경쟁 분석: 같은 부위 점유 시 페널티 (Sullivan 2019)
  └── 비경쟁 + 저에너지 우선 랭킹 → 상위 8 PC 선발
    ↓ 타겟별 상위 Polymer Composition (PC)

Phase 4: GROMACS MD 검증 + MM-GBSA
  ├── GAFF2 모노머 파라미터화 (acpype)
  ├── 200ns 전원자 MD (amber99sb-ildn + GAFF2 + TIP3P)
  ├── 궤적 분석: RMSD, RMSF, H-bond, Rg
  ├── DSSP 2차 구조 변화 추적 (계산적 CD 대체)
  ├── MM-GBSA 결합 자유 에너지 (Sullivan 2019)
  └── 교차 반응성: CD63-PC를 CD81/CD9 에피토프에 테스트
    ↓ 검증된 PC + ΔG 순위

Phase 5: 합성 레시피 + 검증 프로토콜
  ├── 모노머 비율 (1:20 에피토프:모노머, Sehit 2024)
  ├── 합성 프로토콜 (sol-gel / solid-phase / free-radical)
  ├── CD63 이중 에피토프 (펩타이드 + glycan, Teixeira 2021)
  ├── 검증: SPR two-state fitting + QCM-D (Kowalczyk 2023)
  └── CD 분광법 2차 구조 확인 (Sullivan 2019)
    ↓ 최종 레시피 JSON + 합성 프로토콜 텍스트
```

---

## 3. Phase 1: 에피토프 추출 및 구조 준비

**파일**: `code/pipeline/phase1_epitope_prep.py`

### 과학적 원리

에피토프 각인에서 **펩타이드 길이는 9–16 잔기**가 최적이다 (Teixeira et al. 2021):
- 9-mer 이하: 특이성 부족 (다른 단백질과 서열 공유)
- 16-mer 이상: 분자 내 접힘으로 각인 효율 저하

ECL2의 **head subdomain** (helix C-D + 연결 루프)은 서열 가변성이 가장 높아, 세 단백질 간 구분에 최적인 에피토프 후보이다.

### 워크플로우

1. **구조 다운로드**: RCSB PDB (CD81, CD9) 또는 AlphaFold DB (CD63)
2. **AlphaFold 품질 검증**: pLDDT > 70 (B-factor 컬럼에서 추출)
3. **Head region 추출**: Bio.PDB로 잔기 범위 선택, 별도 PDB 저장
4. **물리화학적 분석**: BioPython ProteinAnalysis
   - GRAVY (소수성), pI, MW, H-bond donor/acceptor 수
   - N-glycosylation sequon (NxS/T) 자동 탐지
5. **수용체 PDBQT 생성**: ADFR prepare_receptor4 (또는 meeko fallback)
6. **(선택) MD 안정성**: GROMACS 100ns → RMSD < 3.0 Å (Sehit/Altintas 2024)

### 참고

- Sehit E et al. "Computationally Designed Epitope-Mediated Imprinted Polymers." *ACS Sensors* 2024;9:1831-1841.
- Teixeira SPB et al. "Epitope-imprinted polymers: Design principles." *Science Advances* 2021;7:eabi9884.

---

## 4. Phase 2: Single Monomer Docking (SMD)

**파일**: `code/pipeline/phase2_smd.py`

### 과학적 원리

AutoDock4의 **Lamarckian Genetic Algorithm (LGA)**으로 각 모노머를 에피토프에 개별 도킹한다. Rajpal et al. (2024)의 Table 1에서 11개 실란 모노머의 SMD 결과가 실험과 일치함을 보였다.

AutoDock4를 Vina 대신 사용하는 이유:
- **클러스터 분석**: RMSD 기반 결합 모드 그룹핑 → rank-1 클러스터 mean BE 추출
- **MMSD 호환**: `.dpf` 파일에서 sequential docking 자연스럽게 지원
- **상호작용 분석**: 클러스터별 H-bond/π-π/vdW 유형 추적 가능

### Sullivan 2019 기반 개선

1. **결합 부위 예측**: fpocket으로 에피토프 표면의 결합 포켓 식별 → blind docking 대신 focused docking
2. **Backbone H-bond 분석**: 모노머-단백질 backbone H-bond 비율 > 30%이면 2차 구조 파괴 위험으로 경고
3. **비경쟁 원칙**: 모노머가 단백질 표면에 균일하게 분포해야 좋은 MIP

### Sehit 2024 기반 개선

- **Contact MD**: 각 (모노머, 에피토프) 쌍에 10ns 짧은 MD 실행 → 잔기별 접촉 빈도로 모노머 순위 산출

### 도킹 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| GA runs | 50 | Lamarckian GA 독립 실행 횟수 |
| Population | 150 | GA 집단 크기 |
| Evaluations | 2,500,000 | 최대 에너지 평가 |
| Grid spacing | 0.375 Å | AutoGrid4 격자 간격 |
| Grid points | 60×60×60 | 격자 차원 |
| BE threshold | -2.0 kcal/mol | 유의미한 결합 최소값 |
| ΔΔG threshold | -0.5 kcal/mol | 최소 선택도 |

---

## 5. Phase 3: Multi-Monomer Simultaneous Docking (MMSD)

**파일**: `code/pipeline/phase3_mmsd.py`

### 과학적 원리 (Rajpal et al. 2024)

단일 모노머 도킹(SMD)은 개별 결합력만 평가하지만, 실제 MIP는 **다수 모노머의 동시 상호작용**으로 작동한다. MMSD는 항체 파라토프의 다중점 결합을 모사한다.

**핵심 발견** (Rajpal 2024, Table 2):
- APTMS, APTES: SMD에서 하위권이었으나 MMSD에서 BE가 **30-60% 증가** (시너지)
- DIDMS, IBTES: MMSD에서 5-7% 감소 (간섭)
- **실험 검증**: MMSD 예측 최상위 PC I (PTES+APTMS+APTES+TEOS)이 IF 1.73으로 가장 높은 성능

### Sequential Docking 프로토콜

```
Step 1: Fixed-1 (최고 BE 모노머) → 에피토프에 도킹 → best pose
Step 2: Fixed-1 pose를 수용체에 병합
Step 3: Fixed-2 (TEOS 가교제) → (에피토프 + Fixed-1)에 도킹
Step 4: Fixed-2 pose 병합
Step 5: Variable-1 → (에피토프 + Fixed-1 + Fixed-2)에 도킹
Step 6: Variable-1 병합
Step 7: Variable-2 → 전체 복합체에 도킹
```

### 평가 지표

- **MMSD sum**: 4개 모노머 BE의 합 (낮을수록 좋음)
- **SMD sum**: 동일 4개 모노머의 개별 SMD BE 합
- **Δ = MMSD - SMD**: 음수 = 시너지, 양수 = 간섭
- **경쟁 분석** (Sullivan 2019): 두 모노머가 5 Å 이내에 도킹 → 같은 부위 경쟁 → 페널티

### 참고

- Rajpal S et al. "Rational design based on multi-monomer simultaneous docking for epitope imprinting of SARS-CoV-2 spike protein." *Sci. Rep.* 2024;14:23057.
- Rajpal S, Mizaikoff B. "An in silico predictive method to select multi-monomer combinations for peptide imprinting." *J. Mater. Chem. B* 2022;10:6618-6626.

---

## 6. Phase 4: MD 검증 및 결합 자유 에너지

**파일**: `code/pipeline/phase4_md_validation.py`

### 과학적 원리

GROMACS를 이용한 **200ns 전원자 MD 시뮬레이션**으로 도킹 결과의 동적 안정성을 검증한다. Sullivan et al. (2019)의 프로토콜에 따라 **MM-GBSA**로 결합 자유 에너지를 계산한다.

### MD 프로토콜

| 단계 | 시간 | 조건 |
|------|------|------|
| Energy minimization | - | Steepest descent, 50,000 steps |
| NVT equilibration | 100 ps | V-rescale 300K, position restraints |
| NPT equilibration | 100 ps | Parrinello-Rahman 1 bar |
| **Production MD** | **200 ns** | dt=2fs, LINCS h-bonds |

### 분석 항목

1. **RMSD**: 단백질 backbone 안정성 (목표: < 0.3 nm)
2. **RMSF**: 잔기별 요동 (유연한 루프 식별)
3. **H-bond**: 에피토프-모노머 수소결합 수 시계열
4. **Rg**: 회전 반경 (구조 compact 유지 확인)
5. **DSSP 2차 구조** (Sullivan 2019): 모노머가 α-helix를 파괴하는지 추적 — CD 분광법의 계산적 대체
6. **MM-GBSA** (igb=5, saltcon=0.15): 결합 자유 에너지 ΔG (마지막 50ns, 100 프레임)

### 교차 반응성 테스트

CD63에 최적화된 PC를 CD81과 CD9 에피토프에 적용하여 **선택도** 확인:
```
ΔΔG_selectivity = ΔG(CD63-PC on CD63) - ΔG(CD63-PC on CD81)
```
음수일수록 CD63에 더 선택적.

### 참고

- Sullivan MV et al. "Toward rational design of selective MIPs for proteins." *J. Phys. Chem. B* 2019;123:5432-5443.
- Rebelo P et al. "Rational In Silico Design of MIPs." *Int. J. Mol. Sci.* 2023;24:6785.

---

## 7. Phase 5: 합성 레시피 생성

**파일**: `code/pipeline/phase5_recipe.py`

### 합성 프로토콜 3종

| 방법 | 모노머 유형 | 핵심 특징 | 참고 |
|------|------------|----------|------|
| **Sol-gel** | 실란 | SiO₂ NP + TEOS 가교, RT 16h | Rajpal 2024 |
| **Solid-phase** | 실란/비닐 | Glass bead + 고온 용출, nanoMIP 생산 | Sehit 2024 |
| **Free-radical** | 비닐/아크릴 | APS/TEMED 또는 AIBN 개시제 | 전통적 방법 |

### CD63 이중 에피토프 전략 (Teixeira 2021)

CD63의 3개 N-glycan을 활용한 **펩타이드 + glycan 이중 각인**:
- Layer 1: 펩타이드 에피토프 각인 (기능성 모노머)
- Layer 2: N-acetylneuraminic acid(시알산) 각인 (APBA 보론산)
- CD81(비글리코실화)과의 선택도 극대화

### 실험 검증 계획 (Kowalczyk 2023 기반)

1. **물질 특성화**: FTIR, SEM, zeta potential, DLS
2. **2차 구조 확인**: CD 분광법 (모노머가 에피토프 구조를 파괴하지 않는지)
3. **결합 성능**: IF > 3, KD < 50 nM (Teixeira 2021 기준)
4. **SPR**: Two-state reaction model fitting (Kowalczyk 2023)
5. **QCM-D**: 6.1×10⁴ - 6.1×10⁷ particles/mL 범위 교정
6. **교차 반응성**: 각 MIP를 3개 에피토프 + HSA, BSA, lysozyme에 테스트

---

## 8. 모노머 라이브러리

### A. 실란 모노머 (15종, Sol-gel용)

| 약어 | 이름 | 주 상호작용 |
|------|------|------------|
| PTES | Phenyltriethoxysilane | π-π stacking, 소수성 |
| APTES | (3-Aminopropyl)triethoxysilane | H-bond donor, 정전기 |
| APTMS | (3-Aminopropyl)trimethoxysilane | H-bond, 정전기 |
| UPTMS | 3-Ureidopropyltrimethoxysilane | Multi H-bond (urea D+A) |
| MPTMS | (3-Mercaptopropyl)trimethoxysilane | Thiol-Cys, H-bond |
| IBTES | Isobutyltriethoxysilane | 소수성, alkyl |
| MTMS | Methyltrimethoxysilane | 소수성 |
| **TEOS** | Tetraethyl orthosilicate | **가교제** |
| EDTMS | N-[3-(Trimethoxysilyl)propyl]ethylenediamine | 킬레이트형 H-bond |
| ICTES | 3-(Triethoxysilyl)propyl isocyanate | 공유결합 (Lys) |
| VTMS | Vinyltrimethoxysilane | 소수성 + 가교 가능 |
| GPTMS | (3-Glycidyloxypropyl)trimethoxysilane | Epoxy 공유결합 |
| DIDMS | Dimethyldimethoxysilane | 강한 소수성 |
| CETES | 3-Cyanopropyltriethoxysilane | 약한 H-bond acceptor (CN) |
| TTMS | p-Tolyltrimethoxysilane | π-π + 약한 소수성 |

### B. 비닐/아크릴 모노머 (12종, Free-radical용)

| 약어 | 이름 | 주 상호작용 |
|------|------|------------|
| AA | Acrylic acid | 정전기 (Lys, Arg, His) |
| MAA | Methacrylic acid | 정전기 + 약한 소수성 |
| AAm | Acrylamide | H-bond (D+A) |
| NIPAm | N-Isopropylacrylamide | H-bond + 소수성 |
| 4VIm | 4(5)-Vinylimidazole | π-π, H-bond, His 모방 |
| HEMA | 2-Hydroxyethyl methacrylate | H-bond (OH) |
| **MBAAm** | N,N'-Methylenebisacrylamide | **가교제** |
| DA | Dopamine hydrochloride | Multi H-bond, catechol |
| NE | Norepinephrine | DA-유사 + 추가 OH |
| TBAm | N-tert-Butylacrylamide | 소수성 |
| **APBA** | 3-Aminophenylboronic acid | **Glycan(diol) 인식 — CD63 특이적** |
| **EGDMA** | Ethylene glycol dimethacrylate | **가교제** |

> TEOS, MBAAm, EGDMA는 가교제로 functional monomer 스크리닝에서 제외. 실제 스크리닝 대상은 **24종 functional monomer**.

---

## 9. 디렉토리 구조

```
Monomer screening in Bio/
├── run_pipeline.py                    # 엔트리포인트
├── environment.yml                    # Conda 환경 (GROMACS)
├── README.md                          # 이 문서
│
├── code/
│   ├── pipeline/
│   │   ├── config.py                  # 전역 설정 (타겟, 모노머, 파라미터)
│   │   ├── run_pipeline.py            # 5-Phase 오케스트레이터
│   │   ├── phase1_epitope_prep.py     # 에피토프 추출 + 분석
│   │   ├── phase2_smd.py             # AutoDock4 SMD + 선택도
│   │   ├── phase3_mmsd.py            # MMSD sequential docking
│   │   ├── phase4_md_validation.py   # GROMACS MD + MM-GBSA
│   │   ├── phase5_recipe.py          # 합성 레시피 생성
│   │   ├── generate_report.py        # HTML 리포트
│   │   ├── utils_structure.py        # PDB 다운로드, PDBQT 변환
│   │   ├── utils_autodock.py         # AutoDock4/AutoGrid4 래퍼
│   │   ├── utils_gromacs.py          # GROMACS MD + MM-PBSA 래퍼
│   │   └── utils_analysis.py         # 결합 부위, H-bond, DSSP, 경쟁 분석
│   │
│   └── validation/
│       ├── config_validation.py       # 검증 참조 데이터 (Rajpal 2024)
│       ├── run_validation.py          # 검증 실행기
│       ├── validate_smd.py            # SMD 결과 vs 논문 Table 1
│       ├── validate_mmsd.py           # MMSD 결과 vs 논문 Table 1S
│       └── validate_ranking.py        # 실험 IF와 Spearman 상관
│
├── background papers/                 # 참고 논문 PDF
│   ├── Rajpal et al. - 2024 - ...     # MMSD 방법론
│   ├── Sullivan et al. - 2019 - ...   # MM-GBSA + comonomer 설계
│   ├── Sehit et al. - 2024 - ...      # 에피토프 MD + solid-phase
│   ├── Teixeira et al. - 2021 - ...   # 에피토프 선정 원칙
│   └── Kowalczyk et al. - 2023 - ...  # SPR/QCM-D 테트라스파닌 검증
│
├── datasets/                          # (향후 실험 데이터)
│
└── results/                           # 파이프라인 출력
    ├── phase1/                        # 에피토프 구조, 특성
    ├── phase2/                        # SMD 결과, 히트맵
    ├── phase3/                        # MMSD 결과, 비교 플롯
    ├── phase4/                        # MD 궤적, MM-GBSA
    ├── phase5/                        # 레시피, 프로토콜
    └── reports/                       # HTML 리포트, 로그
```

---

## 10. 실행 방법

```bash
# 환경 활성화
conda activate GROMACS

# 전체 파이프라인 (5 Phase 순차 실행)
python run_pipeline.py

# 특정 타겟만
python run_pipeline.py --target CD63

# 특정 Phase만
python run_pipeline.py --phase 1                 # 에피토프 준비
python run_pipeline.py --phase 2                 # SMD 도킹
python run_pipeline.py --phase 3                 # MMSD
python run_pipeline.py --phase 4                 # MD 검증
python run_pipeline.py --phase 5                 # 레시피

# 옵션
python run_pipeline.py --skip-stability-md       # Phase 1 MD 건너뛰기
python run_pipeline.py --quick-md                # Phase 4: 50ns (200ns 대신)
python run_pipeline.py --no-cross-reactivity     # Phase 4: 교차 반응성 건너뛰기

# HTML 리포트만 생성
python run_pipeline.py --report

# 검증 실행
cd code && python -m validation.run_validation
```

---

## 11. 검증 프레임워크

`code/validation/` 디렉토리는 파이프라인의 정확도를 Rajpal et al. (2024)의 실험 데이터와 비교하여 검증한다.

### 검증 대상

1. **SMD 순위 재현** (Rajpal 2024, Table 1): SARS-CoV-2 spike 에피토프에 대한 11개 실란 모노머 BE 순위
2. **MMSD 시너지 재현** (Rajpal 2024, Table 2): APTMS, APTES의 MMSD BE 30-60% 증가 확인
3. **실험 IF 상관** (Rajpal 2024, Table 3): PC I (IF 1.73) > PC V, VI (대조군) 순서 재현
4. **Spearman 순위 상관**: 계산 MMSD sum vs 실험 IF (ρ > 0.7 목표)

---

## 12. 핵심 파라미터 요약

| Phase | 파라미터 | 기본값 | 근거 |
|-------|---------|--------|------|
| 1 | 에피토프 길이 | 9-16 잔기 | Teixeira 2021 |
| 1 | pLDDT 기준 | ≥ 70 | AlphaFold 품질 |
| 1 | 에피토프:모노머 비율 | 1:20 | Sehit 2024 |
| 2 | GA runs | 50 | AutoDock4 기본 |
| 2 | BE 기준 | ≤ -2.0 kcal/mol | 유의미한 결합 |
| 2 | ΔΔG 기준 | ≤ -0.5 kcal/mol | 선택도 |
| 2 | Backbone H-bond | ≤ 30% | Sullivan 2019 |
| 3 | 조합 크기 | 4-monomer | Rajpal 2024 |
| 3 | 경쟁 거리 | 5.0 Å | Sullivan 2019 |
| 3 | High-affinity | ≤ -11.0 kcal/mol | Rajpal 2024 |
| 4 | MD 시간 | 200 ns | Sullivan 2019 |
| 4 | MM-PBSA 방법 | GBSA (igb=5) | Sullivan 2019 |
| 4 | 온도 | 300 K | 표준 조건 |
| 5 | 목표 KD | < 50 nM | Teixeira 2021 |
| 5 | 목표 IF | > 3 | Teixeira 2021 |

---

## 13. 참고 논문

### 핵심 방법론

1. **Rajpal S** et al. "Rational design based on multi-monomer simultaneous docking for epitope imprinting of SARS-CoV-2 spike protein." *Sci. Rep.* 2024;14:23057. — **MMSD 프로토콜 (Phase 3)**

2. **Rajpal S**, Mizaikoff B. "An in silico predictive method to select multi-monomer combinations for peptide imprinting." *J. Mater. Chem. B* 2022;10:6618-6626. — **MMSD 원본 논문**

3. **Sehit E** et al. "Computationally Designed Epitope-Mediated Imprinted Polymers versus Conventional Epitope Imprints for Human Adenovirus Detection." *ACS Sensors* 2024;9:1831-1841. — **에피토프 MD 안정성, contact MD, solid-phase 합성 (Phase 1, 2)**

4. **Sullivan MV** et al. "Toward Rational Design of Selective Molecularly Imprinted Polymers (MIPs) for Proteins: Computational and Experimental Studies of Acrylamide Based Polymers for Myoglobin." *J. Phys. Chem. B* 2019;123:5432-5443. — **SiteMap, MM-GBSA, backbone H-bond, 비경쟁 원칙 (Phase 2, 3, 4)**

5. **Teixeira SPB** et al. "Epitope-imprinted polymers: Design principles of synthetic binding partners for natural biomacromolecules." *Science Advances* 2021;7:eabi9884. — **에피토프 선정, 이중 에피토프, 성능 목표 (Phase 1, 5)**

### 타겟 검증

6. **Kowalczyk A** et al. "Parallel SPR and QCM-D Quantitative Analysis of CD9, CD63, and CD81 Tetraspanins." *Anal. Chem.* 2023;95:9520-9530. — **SPR/QCM-D 프로토콜, EV 친화도 순서 (Phase 5 검증)**

### 보론산 MIP

7. **Bie Z** et al. "Boronate-affinity glycan-oriented surface imprinting." *Angew. Chem. Int. Ed.* 2015;54:10211-10215. — **APBA glycan 인식 (CD63 특이성)**

### 에피토프 선정

8. **Bossi AM** et al. "Molecularly imprinted polymers by epitope imprinting: bioinformatics resources." *Anal. Bioanal. Chem.* 2021;413:6101-6112. — **BLAST 고유성 검증**

### MIP 계산 설계 리뷰

9. **Rebelo P** et al. "Rational In Silico Design of MIPs: Current Challenges and Future Potential." *Int. J. Mol. Sci.* 2023;24:6785. — **GROMACS + gmx_MMPBSA 프로토콜**

---

## 14. 설치 및 환경

### Conda 환경

```bash
# 환경 생성 (또는 기존 GROMACS 환경 사용)
conda env create -f environment.yml
conda activate GROMACS

# AutoDock4 + AutoGrid4
conda install -c bioconda autodock autogrid

# ADFR Suite (prepare_receptor4, prepare_ligand4)
conda install -c hcc adfr-suite

# pip 패키지
pip install acpype gmx_MMPBSA meeko mdanalysis
```

### 필수 도구

| 도구 | 버전 | 용도 | 설치 |
|------|------|------|------|
| AutoDock4 | 4.2.6 | LGA 도킹 | `conda install -c bioconda autodock` |
| AutoGrid4 | 4.2.6 | 격자 맵 생성 | `conda install -c bioconda autogrid` |
| GROMACS | ≥ 2021 | MD 시뮬레이션 | `apt install gromacs` |
| ADFR Suite | 1.0 | PDBQT 변환 | `conda install -c hcc adfr-suite` |
| RDKit | ≥ 2025.03 | 분자 구조 생성 | `conda install rdkit` |
| acpype | 2023.10 | GAFF2 파라미터화 | `pip install acpype` |
| gmx_MMPBSA | 1.6 | MM-PBSA/GBSA | `pip install gmx_MMPBSA` |
| meeko | 0.7 | PDBQT 변환 | `pip install meeko` |
| BioPython | ≥ 1.86 | PDB 파싱, 서열 분석 | `conda install biopython` |
| MDAnalysis | ≥ 2.9 | 궤적 분석, contact frequency | `pip install mdanalysis` |

### 선택 도구

| 도구 | 용도 | 비고 |
|------|------|------|
| fpocket | 결합 부위 예측 | 미설치 시 geometric fallback |
| OpenBabel | 분자 형식 변환 | 미설치 시 RDKit fallback |

---

*이 파이프라인은 CD63/CD81/CD9 테트라스파닌 ECL2에 대한 에피토프-각인 MIP의 in silico 모노머 스크리닝을 위해 설계되었으며, Rajpal 2024의 MMSD 방법론을 핵심으로 Sullivan 2019, Sehit 2024, Teixeira 2021, Kowalczyk 2023의 계산/실험 전략을 통합한다.*
