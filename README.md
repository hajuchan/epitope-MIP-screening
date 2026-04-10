# Epitope-MIP Computational Screening Pipeline

엑소좀 테트라스파닌 ECL2(CD63 / CD81 / CD9)를 선택적으로 인식하는 에피토프-각인 MIP의 최적 모노머 조합을 계산화학적으로 스크리닝하는 6-Phase 파이프라인.

---

## 목차

1. [프로젝트 배경](#1-프로젝트-배경)
2. [파이프라인 전체 구조](#2-파이프라인-전체-구조)
3. [각 Phase 상세](#3-각-phase-상세)
4. [모노머 및 가교제 라이브러리](#4-모노머-및-가교제-라이브러리)
5. [실행 방법](#5-실행-방법)
6. [핵심 파라미터](#6-핵심-파라미터)
7. [참고 문헌](#7-참고-문헌)
8. [설치 및 환경](#8-설치-및-환경)

---

## 1. 프로젝트 배경

### 엑소좀과 테트라스파닌

엑소좀은 30–150 nm 세포외 소포체로, **CD9/CD63/CD81** 테트라스파닌이 가장 보편적인 바이오마커이다. 현재 항체 기반 검출은 비용이 높고 안정성이 낮다.

### MIP (Molecularly Imprinted Polymer)

"플라스틱 항체" — 합성 비용 낮고 안정성 우수. **에피토프 각인**은 표면 노출 펩타이드만 템플릿으로 사용하여 거대 분자 각인의 한계를 극복한다 [1].

### 타겟

| 단백질 | 구조 | ECL2 (도킹) | Head 16-mer (합성/MD) | N-Glycan |
|--------|------|------------|----------------------|----------|
| **CD63** | AlphaFold P08962 | 103-203 | 155-170 `EKIPSMSKNRVPDSCC` | **3개** |
| **CD81** | PDB 5TCX | 113-201 | 168-183 `SVLKNNLCPSGSNIIS` | **0개** |
| **CD9** | PDB 6K4J | 112-195 | 156-171 `AGGVEQFISDICPKKD` | **1개** |

---

## 2. 파이프라인 전체 구조

```
Phase 1: 에피토프 추출 + 구조 준비
  ├── PDB/AlphaFold 다운로드 → ECL2 전체 추출 (도킹 receptor)
  ├── Head 16-mer 추출 (합성 템플릿 / Phase 4,5 template)
  ├── PROPKA pH 7.4 protonation + BLAST 고유성 확인
  ├── 20ns MD 안정성 + Ensemble conformer 5개 추출
  └── → ECL2 receptor PDBQT (×6)

Phase 2: Single Monomer Docking (SMD) — AutoDock4/GPU
  ├── 24 monomer × 3 타겟 × 6 conformer 도킹
  ├── BE ≤ -2.0 kcal/mol 필터 → 상위 12개 선택
  └── → 타겟별 상위 12개 모노머

Phase 3: Greedy Forward Selection + MMSD — Rajpal 2024
  ├── Greedy forward selection: 모노머 순차 추가 (avg BE 개선까지)
  ├── Swap refinement: 각 위치에서 대안 교체 시도
  ├── 가교제 자동 선택: MMSD 마지막 step에서 호환 XL 전부 시도
  └── → 최적 모노머 조합 + 가교제 (종류 + 개수 자동 결정)

Phase 4: Pre-polymerization MD — Head 16-mer template
  ├── 25개 모노머 랜덤 배치 (PACKMOL 방식) + 100ns MD
  ├── Contact frequency (6Å) + residence time + pair distance
  ├── MM-GBSA 결합 자유 에너지
  └── → 합성 비율 (contact freq 역비례: 약한 결합 → 많이 넣기)

Phase 5: VIP Cavity Rebinding — Zink 2018
  ├── 균등 간격 5개 snapshot 선택 (cherry-picking 방지)
  ├── 모노머 position restraint → 중합 근사
  ├── Template removal test: 결합 강도 적정성 확인
  │   (너무 강하면 template 제거 불가 → MIP 성능 저하)
  ├── Rebinding MD: RMSD < 5Å → 성공
  ├── Selectivity: 다른 target head로 rebinding → own만 성공이면 selective
  └── → Cavity 검증 + selectivity + 재현성 (≥3/5)

Phase 6: 합성 레시피 (Phase 5 통과한 target만)
  ├── Phase 4 optimal ratio + Phase 5 rebinding 검증 반영
  ├── 합성 프로토콜 (sol-gel / solid-phase / free-radical)
  └── → 최종 레시피 JSON + 합성 프로토콜
```

---

## 3. 각 Phase 상세

### Phase 1: 에피토프 추출

**파일**: `code/pipeline/phase1_epitope_prep.py`

- 도킹에는 **ECL2 전체**(~90 잔기) 사용 — disulfide bond 유지
- MD/합성에는 **Head 16-mer** 사용 — 실제 합성 template과 동일
- Ensemble conformer: MD 궤적에서 5개 대표 구조 추출 → receptor 유연성 반영

### Phase 2: Single Monomer Docking

**파일**: `code/pipeline/phase2_smd.py`

- AutoDock4 LGA 도킹 [5] + AutoDock-GPU 가속 [6]
- Si/B 비표준 원자: UFF 기반 커스텀 파라미터 [8]
- Ensemble docking: 6개 receptor에 도킹 → best BE 선택
- 필터: BE ≤ -2.0, 상위 12개 → Phase 3 전달

### Phase 3: Multi-Monomer Optimization

**파일**: `code/pipeline/phase3_mmsd.py`

- **Greedy forward selection**: 모노머 순차 추가, avg BE 악화 시 중단 → 최적 크기 자동 결정 (2-6종)
- **MMSD sequential docking** [9]: 이전 결과를 receptor에 병합하여 다중 모노머 시너지/간섭 평가
- **가교제 자동 선택**: 6종 (TEOS, TMOS, MBAAm, EGDMA, DVB, TRIM) 중 호환 XL 전부 도킹 → best 선택
- 목적함수: `mmsd_per_monomer + 0.3 × max(0, delta_sum)` — 크기 정규화 + 간섭 페널티

### Phase 4: Pre-polymerization MD

**파일**: `code/pipeline/phase4_md_validation.py`

- **Head 16-mer를 template으로 사용** (문헌 표준, Rajpal 2024 [12])
- 25개 모노머 × 랜덤 배치 → 100ns all-atom MD (GROMACS GPU)
- 분석: contact frequency (6Å), mean min distance, residence time, RMSD, RMSF, H-bond, Rg, MM-GBSA
- **합성 비율**: contact freq 역비례 — 약한 결합 모노머를 더 많이 넣어 균등한 cavity 형성

### Phase 5: VIP Cavity Rebinding

**파일**: `code/pipeline/phase5_rebinding.py`

VIP (Virtually Imprinted Polymer) 방식 [11]:
1. Phase 4 trajectory 후반 50%에서 **균등 간격 5개 snapshot** 선택
2. 모노머 position restraint (1000 kJ/mol/nm²) → 중합 근사
3. **Template removal test** (10ns): template이 이탈하면 제거 가능 (moderate binding = good MIP)
4. **Rebinding MD** (20ns): template RMSD < 5Å → cavity 인식 성공
5. **Selectivity**: 다른 target head로 rebinding → own만 성공이면 selective
6. **5/5 성공률 ≥ 3** → 재현 가능

### Phase 6: 합성 레시피

**파일**: `code/pipeline/phase6_recipe.py`

- Phase 5 rebinding 실패(0/5) target은 **레시피 생성 제외**
- Phase 4 optimal ratio 적용 (MD 기반 비율)
- 합성 프로토콜: sol-gel (실란) / free-radical (비닐) / solid-phase
- CD63 이중 에피토프 전략: 펩타이드 + glycan [1]
- 실험 검증 계획: SPR, QCM-D [14]

---

## 4. 모노머 및 가교제 라이브러리

### Functional Monomers (24종)

**실란 (14종, sol-gel)**: PTES, APTES, APTMS, UPTMS, MPTMS, IBTES, MTMS, EDTMS, ICTES, VTMS, GPTMS, DIDMS, CETES, TTMS

**비닐/아크릴 (10종, free-radical)**: AA, MAA, AAm, NIPAm, 4VIm, HEMA, DA, NE, TBAm, APBA

### 가교제 (6종)

| 가교제 | 유형 | 관능기수 | 호환 |
|--------|------|---------|------|
| TEOS | 실란 | 4 | 실란 모노머 |
| TMOS | 실란 | 4 | 실란 모노머 |
| MBAAm | 비닐 | 2 | 비닐/아크릴 |
| EGDMA | 메타크릴 | 2 | 비닐/아크릴 |
| DVB | 스티렌 | 2 | 비닐 |
| TRIM | 메타크릴 | 3 | 비닐/아크릴 |

Si 포함 모노머는 **PolCA force field** (Jorge 2021 [13])로 파라미터화.

---

## 5. 실행 방법

```bash
conda activate GROMACS

# 전체 파이프라인
python run_pipeline.py

# 특정 타겟
python run_pipeline.py --target CD63 CD81

# 특정 Phase
python run_pipeline.py --phase 4 --target CD63
python run_pipeline.py --phase 5 --target CD63    # rebinding
python run_pipeline.py --phase 6                   # recipe

# Quick MD (디버깅)
python run_pipeline.py --phase 4 --quick-md --target CD63
```

### 디렉토리 구조

```
Monomer_screening_in_Bio/
├── run_pipeline.py                    # 엔트리포인트
├── environment.yml                    # Conda 환경
├── code/pipeline/
│   ├── config.py                      # 전역 설정
│   ├── run_pipeline.py                # Phase 오케스트레이터
│   ├── phase1_epitope_prep.py
│   ├── phase2_smd.py
│   ├── phase3_mmsd.py
│   ├── phase4_md_validation.py
│   ├── phase5_rebinding.py            # VIP cavity rebinding
│   ├── phase6_recipe.py               # 합성 레시피
│   ├── generate_report.py             # HTML 리포트
│   ├── utils_structure.py
│   ├── utils_autodock.py
│   ├── utils_gromacs.py
│   └── utils_analysis.py
└── results/
    ├── phase1/ ~ phase6/
    └── reports/pipeline_report.html
```

---

## 6. 핵심 파라미터

| Phase | 파라미터 | 값 | 근거 |
|-------|---------|-----|------|
| 1 | 에피토프 길이 | 16 잔기 | Teixeira 2021 [1] |
| 1 | 도킹 receptor | ECL2 전체 | Disulfide 유지 |
| 1 | Ensemble conformer | 5개 | 수용체 유연성 |
| 2 | BE threshold | ≤ -2.0 kcal/mol | 유의미한 결합 |
| 2 | Top N for Phase 3 | 12 | BE 상위 선택 |
| 3 | 조합 크기 | 2-6종 (자동) | Greedy selection |
| 3 | 가교제 선택 | MMSD 도킹 기반 | 마지막 step 분기 |
| 4 | Template | Head 16-mer | 합성 조건 동일 |
| 4 | 모노머 수 | 25개 (각 type 5개) | 통계 + 계산 효율 |
| 4 | MD 시간 | 100 ns | 평형 도달 |
| 4 | Contact cutoff | 6.0 Å | vdW 포함 (문헌 표준) |
| 5 | Snapshot 선택 | 균등 간격 5개 | Cherry-picking 방지 |
| 5 | Restraint 강도 | 1000 kJ/mol/nm² | Zink 2018 [11] |
| 5 | Rebinding 기준 | RMSD < 5 Å | Cavity 안정 |
| 5 | 성공률 기준 | ≥ 3/5 | 재현성 |

---

## 7. 참고 문헌

**MIP 에피토프 설계**

[1] Teixeira SPB et al. "Epitope-imprinted polymers: Design principles." *Science Advances* 2021;7:eabi9884.

[2] Bossi AM et al. "MIP by epitope imprinting: bioinformatics." *Anal. Bioanal. Chem.* 2021;413:6101-6112.

[3] Sehit E et al. "Computationally Designed Epitope-Mediated Imprinted Polymers." *ACS Sensors* 2024;9:1831-1841.

**도킹 + 모노머 스크리닝**

[4] Li H et al. "PROPKA pKa prediction." *Proteins* 2005;61:704-721.

[5] Morris GM et al. "AutoDock4." *J. Comput. Chem.* 2009;30:2785-2791.

[6] Santos-Martins D et al. "AutoDock-GPU." *J. Chem. Theory Comput.* 2021;17:1060-1073.

[7] Sullivan MV et al. "Rational Design of Selective MIPs for Proteins." *J. Phys. Chem. B* 2019;123:5432-5443.

[8] Rappe AK et al. "UFF force field." *JACS* 1992;114:10024-10035.

[9] Rajpal S et al. "Multi-monomer simultaneous docking for epitope imprinting." *Sci. Rep.* 2024;14:23057.

[10] Rajpal S, Mizaikoff B. "In silico multi-monomer combinations." *J. Mater. Chem. B* 2022;10:6618-6626.

**VIP + MD 시뮬레이션**

[11] Zink S et al. "Virtually imprinted polymers (VIPs)." *Phys. Chem. Chem. Phys.* 2018;20:13145-13152.

[12] Rajpal S et al. "MD in pre-polymerization mixtures." *J. Mol. Model.* 2024;30:247.

[13] Jorge M et al. "PolCA force field for organosilicon." *ACS Phys. Chem. Au* 2021;1:34-49.

**검증**

[14] Kowalczyk A et al. "SPR and QCM-D for CD9/CD63/CD81." *Anal. Chem.* 2023;95:9520-9530.

---

## 8. 설치 및 환경

```bash
# 1. 환경 생성
conda env create -f environment.yml
conda activate GROMACS

# 2. GROMACS GPU 빌드 (필수)
# https://manual.gromacs.org/current/download.html
# config.py에서 GMX_BIN 경로 설정

# 3. AutoDock-GPU (Phase 2/3 가속)
# https://github.com/ccsb-scripps/AutoDock-GPU

# 4. 설치 확인
python -c "from rdkit import Chem; print('RDKit OK')"
python -c "import MDAnalysis; print('MDAnalysis OK')"
gmx --version
autodock_gpu_128wi --version
```

### 테스트 환경

- OS: Ubuntu 22.04 (WSL2)
- Python: 3.13
- GROMACS: 2025.2 (GPU build)
- AutoDock-GPU: v1.6
- GPU: NVIDIA RTX 4070 Ti (12GB)
