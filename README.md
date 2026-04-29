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
  ├── EBN / HBNMax / H-bond lifetime (Yuan 2024)
  ├── Per-atom RDF (기능기별 결합 메커니즘)
  ├── Crosslinker proximity check (< 10Å)
  ├── MD convergence check (25ns 윈도우 비교)
  ├── MM-GBSA + per-residue decomposition (hotspot 분석)
  └── → 합성 비율 (contact freq 역비례: 약한 결합 → 많이 넣기)

Phase 5: VIP Cavity Rebinding — Zink 2018
  ├── 균등 간격 5개 snapshot 선택 (cherry-picking 방지)
  ├── 모노머 position restraint → 중합 근사
  ├── Template removal test: 결합 강도 적정성 확인
  │   (너무 강하면 template 제거 불가 → MIP 성능 저하)
  ├── Rebinding MD: RMSD + H-bond + contact + MM-GBSA
  ├── Selectivity Index (SI = RMSD_other/RMSD_own) + t-test
  └── → Cavity 검증 + 정량적 selectivity + 재현성 (≥3/5)

Phase 6: 합성 레시피 (Phase 5 통과한 target만)
  ├── Phase 4 optimal ratio + Phase 5 rebinding 검증 반영
  ├── 합성 프로토콜 (sol-gel / solid-phase / free-radical)
  └── → 최종 레시피 JSON + 합성 프로토콜
```

---

## 3. 각 Phase 상세

### Phase 1: 에피토프 추출 및 구조 준비

**파일**: `code/pipeline/phase1_epitope_prep.py`

구조 소스: CD63은 AlphaFold [25] DB API (UniProt P08962, pLDDT > 70 검증), CD81/CD9는 RCSB PDB (5TCX, 6K4J). pdbfixer [31]로 missing side chain 자동 수정.

**도킹 receptor (Phase 2-3용)**: ECL2 전체 (~90 잔기) 추출. Head만 사용하면 CCG 모티프의 disulfide bond가 끊어져 구조 불안정. ECL2 전체를 사용하되, grid center를 head 16-mer에 맞춤으로써 도킹이 head 부위를 향하도록 유도.

**합성 template (Phase 4-5용)**: Head 16-mer 추출. 실제 MIP 합성에서 주문하는 펩타이드와 동일. 에피토프 길이는 Teixeira et al. [1] 권장 범위(9-16 잔기) 내.

**Protonation**: PROPKA 3.5 [4]로 pH 7.4 (PBS 조건) protonation 상태 결정. His/Cys 양성자화 상태 자동 할당.

**고유성 검증**: NCBI BLAST로 16-mer가 인간 프로테옴에서 고유한지 확인 [2]. >70% identity 타 단백질 발견 시 경고.

**MD 안정성**: GROMACS [18] 20ns MD (amber99sb-ildn [20], TIP3P [19], 0.15M NaCl, 300K, NPT) → RMSD < 3.0 Å 확인 [3].

**Ensemble conformer**: 20ns MD 궤적에서 RMSD 기반 clustering → 5개 대표 구조 추출. 각각 OpenBabel [29]/ADFR로 receptor PDBQT 생성. Phase 2에서 6개 receptor (원본 + 5 conformer)에 도킹하여 receptor 유연성 반영.

### Phase 2: Single Monomer Docking (SMD)

**파일**: `code/pipeline/phase2_smd.py`

**도킹 엔진**: AutoDock4 [5] Lamarckian Genetic Algorithm (LGA). GPU 가속 시 AutoDock-GPU [6] 사용 (동일 force field + scoring function, ~100-350× 가속).

**도킹 파라미터**: GA runs = 50, population = 150, max evaluations = 2,500,000, grid spacing = 0.375 Å, grid points = 60×60×60. RMSD 기반 클러스터링 후 rank-1 클러스터 mean BE 추출.

**비표준 원자 처리 (Si, B)**: AutoDock4 기본 force field에 Si/B 미포함. (1) PDBQT 생성: Si→S, B→C proxy 치환 → meeko로 PDBQT 생성 → 원래 원자 타입 복원. (2) 커스텀 파라미터 파일 `AD4_parameters_custom.dat`: UFF [8] 기반 Si_3 (Rii=4.295Å, ε=0.402 kcal/mol), B_3 (Rii=4.083Å, ε=0.180 kcal/mol). (3) AutoDock-GPU: `--derivtype Si=S/B=C` + parameter_file.

**Ensemble docking**: 24 monomer × 3 target × 6 conformer = 최대 432 도킹. 모노머별로 6개 receptor 중 best BE 선택.

**필터링**: BE ≤ -2.0 kcal/mol (유의미한 결합), 상위 12개를 Phase 3으로 전달. ΔΔG 선택도는 계산하되 필터에 사용하지 않음 — SMD는 개별 상호작용만 평가하므로 multi-monomer selectivity는 Phase 3 이후에 판단.

### Phase 3: Multi-Monomer Simultaneous Docking (MMSD)

**파일**: `code/pipeline/phase3_mmsd.py`

**MMSD 프로토콜** [9,10]: Sequential docking — Step k에서 monomer k를 (에피토프 + 이전 k-1 모노머 pose 병합체)에 도킹. 이전 모노머가 차지한 공간을 피해 새로운 위치에 도킹되므로, multi-monomer synergy/interference를 평가 가능.

**Greedy forward selection**: (1) Phase 2 SMD BE 순으로 12개 모노머 정렬. (2) Round 1: 12개 각각 단독 MMSD → best 1종 선택 (12회). (3) Round 2: 나머지 11개를 각각 추가 → 모노머당 평균 BE (`mmsd_per_monomer = mmsd_sum / n_monomers`) 최소인 2종 선택 (11회). (4) 반복... avg BE 악화 시 중단 → 최적 크기 자동 결정 (2-6종). (5) Swap refinement: 선택된 각 위치에서 나머지 모노머로 교체 시도 → bo_objective 개선 시 교체. 총 ~70-90회 MMSD 평가.

**목적함수 (selectivity-aware)**:
```
bo_objective = mmsd_per_monomer + w_interfere × max(0, delta_sum) + w_sel × selectivity_penalty

selectivity_penalty = max(0, ΔΔG - threshold)
ΔΔG = BE_own - mean(BE_off_targets)    # negative = selective
```
- `mmsd_per_monomer = mmsd_sum / n_monomers`: 크기 정규화. 조합 크기가 다른 PC를 공정 비교 (Rajpal 2024 [9]는 4종 고정이라 정규화 불필요했으나, 본 파이프라인은 2-6종 가변).
- `delta_sum = mmsd_sum - smd_sum`: < 0이면 시너지 (cooperative binding), > 0이면 간섭 (steric clash) [9, Table 2].
- `w_interfere = 0.3`: 간섭 페널티 가중치. 시너지(delta < 0)는 페널티 없음 — MMSD에서 BE 증가로 자연 반영.
- **`selectivity_penalty`**: 각 모노머 조합을 own target뿐 아니라 **다른 target의 receptor에도 cross-docking**하여, ΔΔG가 threshold(-1.0 kcal/mol)보다 크면 (비선택적) 페널티 부과 [32,33]. `w_sel = 0.5`: selectivity와 affinity를 동등 가중.
- Rajpal 2024 [9]는 selectivity를 목적함수에 미포함 — 본 파이프라인의 contribution.

**가교제 자동 선택**: MMSD 마지막 step에서 호환 가교제 전부 도킹 → BE 최소 선택. 실란 조합: TEOS/TMOS (2종), 비닐 조합: MBAAm/EGDMA/DVB/TRIM (4종). 추가 비용: 비닐 조합당 +3 도킹.

### Phase 4: Pre-polymerization MD

**파일**: `code/pipeline/phase4_md_validation.py`

**시스템 구축**: Head 16-mer를 template으로 사용 (ECL2가 아닌 실제 합성 template) [12]. Phase 3 최적 조합의 functional monomer k종 × 5 copy + crosslinker × 5 copy = 25개 모노머. Protein 중심에서 반경 3.1-4.1 nm 구 껍질에 랜덤 배치 (min separation 1.0 nm, 겹침 방지). 문헌 표준 PACKMOL [27] 방식 [12].

**Force field**: protein — amber99sb-ildn [20]; vinyl monomers — GAFF2 (acpype [28]); Si-containing monomers — PolCA [13] (GAFF2 + Si LJ 파라미터). Si 원자의 bond equilibrium distances는 표준 공유결합 길이 사용 (Si-C: 0.186 nm, Si-O: 0.164 nm).

**MD 프로토콜**: 

| 단계 | 조건 |
|------|------|
| Solvation | Cubic box, 0.5 nm padding, TIP3P [19], 0.15M NaCl (PBS) |
| Energy minimization | Steepest descent, 50,000 steps, Fmax < 1000 kJ/mol/nm |
| NVT equilibration | 100 ps, V-rescale [21] 300K, dt=2fs, LINCS [24] h-bonds |
| NPT equilibration | 100 ps, Parrinello-Rahman [22] 1 bar, 300K |
| Production | 100 ns, dt=2fs, PME [23] (rcoulomb=1.0 nm, rvdw=1.0 nm), GPU 가속 |

**분석 (trajectory 후반 50%, stride 100)**:

| 지표 | 방법 | 근거 |
|------|------|------|
| Contact frequency | 각 모노머 type의 head 6Å 이내 접촉 프레임 비율 (MDAnalysis [26]) | 문헌 표준 cutoff [12] |
| Mean min distance | 모노머-head 최소 원자 간 거리 평균 | 접촉 품질 |
| Residence time | 연속 접촉 프레임 수 (안정적 결합 vs 스침) | 결합 안정성 |
| EBN (Effective Binding Number) | 프레임당 동시 접촉 모노머 최대 수 | Yuan et al. 2024 [14] |
| HBNMax | MDAnalysis HydrogenBondAnalysis (d-a 3.5Å, angle 150°) | Yuan et al. 2024 [14] |
| H-bond lifetime | 프레임당 평균 H-bond 수 (H-bond 안정성) | Yuan et al. 2024 [14] |
| Per-atom RDF | InterRDF — OH/NH/CO 기능기별 g(r) peak | Yuan et al. 2024 [14] |
| Monomer pair distance | 모노머 간 최소 원자 거리 | Cavity compactness |
| Crosslinker proximity | 가교제-모노머 min distance < 10Å 확인 | Rajpal 2023 [10] |
| RMSD/RMSF/H-bond/Rg | GROMACS gmx rms/rmsf/hbond/gyrate | 구조 안정성 |
| MM-GBSA + per-residue decomposition | gmx_MMPBSA (igb=5, saltcon=0.15, idecomp=2) [7,15,17] | 결합 자유 에너지 + hotspot |
| MD convergence | 25ns 윈도우 간 contact freq 차이 < 10% | Polania & Jiménez 2024 [12] |

**EBN** (Effective Binding Number): 프레임당 동시에 6Å 이내에 있는 동일 모노머 분자 수의 최대값. EBN > 1이면 해당 모노머가 에피토프의 여러 부위에 동시 결합 가능 → 합성 시 비율 조정 근거 [14].

**HBNMax** (Maximum H-bond Number): 에피토프-모노머 간 최대 H-bond 수. H-bond이 모노머-에피토프 결합의 주요 구동력이므로, HBNMax가 높은 모노머가 cavity 형성에 더 효과적 [14].

**Per-atom RDF**: 모노머 원자와 에피토프 기능기 (OH, NH, CO) 간 radial distribution function. g(r) peak > 1.5이면 해당 기능기 쌍이 특이적 상호작용을 형성함. 결합 *메커니즘*을 해석하는 데 핵심.

**Crosslinker proximity**: 가교제가 기능성 모노머와 10Å 이내에 배치되지 않으면, 실제 중합 시 가교 네트워크가 모노머를 포획하지 못함. 가교제-모노머 근접성이 모두 < 10Å이면 "well-positioned".

**합성 비율 결정**: EBN (Effective Binding Number) 기반 직접 비례 [14].
```
ratio_i = EBN_i / min(EBN_j for all j)
```
EBN = template에 동시 결합 가능한 최대 모노머 분자 수. EBN이 높은 모노머는 template 표면의 결합 site가 많으므로 더 많이 넣어 모든 site를 포화시킨다. Crosslinker는 functional monomer ratio 합과 동량.

### Phase 5: VIP Cavity Rebinding

**파일**: `code/pipeline/phase5_rebinding.py`

VIP (Virtually Imprinted Polymer) [11] 방식으로 중합을 근사하여 cavity 형성 및 rebinding 검증.

**Snapshot 선택**: Phase 4 trajectory 후반 50%에서 균등 간격 5개 frame 추출. Cherry-picking 방지 — 실제 중합은 UV/열 조사에 의해 랜덤 시점에 발생하므로, 특정 "최적 프레임"을 선택하면 과적합.

**중합 근사**: 모노머 heavy atoms에 harmonic position restraint (k = 1000 kJ mol⁻¹ nm⁻², GROMACS standard) 적용. "모노머가 현재 위치에서 cross-linking되어 polymer network에 잠긴" 상태를 근사.

**Template removal test** (10ns MD): 모노머 restrained + template(head) + 물 자유. Template이 cavity에서 이탈하면 (RMSD > 5Å) "removable" = template 세척 가능 = 적정 결합 강도. Template이 이탈 못 하면 "stuck" = 결합 너무 강함 = 실제 합성 시 template 제거 어려움 → IF 저하.

**Rebinding MD** (20ns): 동일 시스템에서 template이 cavity에 안정적으로 머무르는지 확인. Template backbone RMSD (후반 50% 평균, `gmx rms`로 PBC 보정) < 5Å → rebinding 성공. 5Å threshold: head 16-mer 크기(~1.5 nm = 15Å 직경)의 1/3 이내 변위.

**Rebinding 분석 지표**:

| 지표 | 방법 | 근거 |
|------|------|------|
| RMSD | `gmx rms` (PBC 보정) | Zink 2018 [11] |
| H-bond count | `gmx hbond` template-monomer 간 | Yuan 2024 [14] |
| Contact count | MDAnalysis 6Å cutoff | 문헌 표준 [12] |
| MM-GBSA ΔG | gmx_MMPBSA (후반 50% trajectory) | Kumar 2024 [15] |

**Selectivity 평가**: 같은 cavity에 다른 target head를 넣어 rebinding MD. 다른 head는 pdb2gmx로 새 topology 생성 → cavity의 restrained 모노머 + 물과 병합 → 20ns MD. 정량적 selectivity 지표:

```
Selectivity Index (SI) = RMSD_other / RMSD_own
  SI > 1.5 → selective
  SI 1.0-1.5 → weak selectivity  
  SI < 1.0 → cross-reactive
```
Welch's t-test로 own vs other RMSD 차이의 통계적 유의성 확인 (p < 0.05). H-bond count 및 MM-GBSA ΔG도 비교하여 selectivity의 물리적 근거 제시 [15,16].

**재현성**: 5개 snapshot 중 ≥ 3개 성공 → 재현 가능한 cavity.

### Phase 6: 합성 레시피

**파일**: `code/pipeline/phase6_recipe.py`

Phase 5 rebinding 검증을 통과한 target에 대해서만 합성 레시피 생성. Rebinding 실패 (0/5) target은 제외.

**비율**: Phase 4 MD contact frequency 기반 optimal ratio 적용.

**합성 프로토콜**: 실란 모노머 → sol-gel (TEOS/TMOS 가교, RT 16h) [9]; 비닐 모노머 → free-radical (APS/TEMED 또는 AIBN 개시제); 혼합 → solid-phase (glass bead) [3,30].

**CD63 이중 에피토프 전략** [1]: CD63의 3개 N-glycan을 활용한 펩타이드 + glycan 이중 각인. Layer 1: 펩타이드 에피토프 (기능성 모노머), Layer 2: N-acetylneuraminic acid (APBA 보론산).

**실험 검증 계획** [34]: SPR two-state reaction model fitting, QCM-D (6.1×10⁴ ~ 6.1×10⁷ particles/mL), 교차 반응성 (3개 에피토프 + HSA, BSA, lysozyme). 목표: IF > 3, KD < 50 nM [1].

### 검증 (Rajpal 2024 벤치마크)

SARS-CoV-2 spike protein 에피토프 (PDB 7JMO, 잔기 473-497)에 대해 Rajpal et al. [9] Table 1-2 재현:

| 검증 항목 | 결과 |
|----------|------|
| SMD 개별 BE 정확도 | 10/10 모노머 ±2.0 kcal/mol 이내 |
| PTES top-ranked | PASS |
| MMSD 시너지 방향 | 4/4 (APTMS/APTES/MPTMS/UPTMS) |
| MMSD sum vs IF Spearman ρ | 0.632 (기준 ≥ 0.6) |
| 비경쟁 결합 | 3/3 top PC uniform |

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
| 4 | H-bond cutoff | d-a 3.5Å, angle 150° | Yuan 2024 [14] |
| 4 | Convergence threshold | 윈도우 diff < 10% | Polania 2024 [12] |
| 4 | Crosslinker proximity | < 10 Å | Rajpal 2023 [10] |
| 4 | MM-GBSA decomposition | idecomp=2, within 6Å | Kumar 2024 [15] |
| 5 | Snapshot 선택 | 균등 간격 5개 | Cherry-picking 방지 |
| 5 | Restraint 강도 | 1000 kJ/mol/nm² | Zink 2018 [11] |
| 5 | Rebinding 기준 | RMSD < 5 Å | Cavity 안정 |
| 5 | Selectivity Index | SI > 1.5 = selective | Mohsenzadeh 2024 [16] |
| 5 | 통계 검정 | Welch's t-test, p < 0.05 | 표준 통계 |
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

[12] Polania LC, Jiménez VA. "Molecular dynamics simulations in pre-polymerization mixtures for peptide recognition." *J. Mol. Model.* 2024;30:266.

[13] Jorge M et al. "PolCA force field for organosilicon." *ACS Phys. Chem. Au* 2021;1:34-49.

**정량적 ��석**

[14] Yuan J et al. "Computational and Experimental Comparison of MIPs — Quantitative Parameters (EBN, HBNMax)." *Molecules* 2024;29:4236.

[15] Kumar MD et al. "Computational modelling and optimization studies of electropentamer for molecular imprinting — per-residue MMPBSA decomposition." *J. Mol. Graph. Model.* 2024;128:108715.

[16] Mohsenzadeh E et al. "Design of MIPs using computational methods: strategies and approaches." *WIREs Comput. Mol. Sci.* 2024;14:e1713.

[17] MIP-PhAC Dataset. "MD/MM-PBSA and DFT Resources for MIP Design." *Data* 2025;10:205.

**소프트웨어 + 방법론**

[18] Abraham MJ et al. "GROMACS: High performance molecular simulations." *SoftwareX* 2015;1-2:19-25.

[19] Jorgensen WL et al. "Comparison of simple potential functions for simulating liquid water (TIP3P)." *J. Chem. Phys.* 1983;79:926-935.

[20] Lindorff-Larsen K et al. "Improved side-chain torsion potentials for the Amber ff99SB-ILDN protein force field." *Proteins* 2010;78:1950-1958.

[21] Bussi G et al. "Canonical sampling through velocity rescaling (V-rescale thermostat)." *J. Chem. Phys.* 2007;127:014102.

[22] Parrinello M, Rahman A. "Polymorphic transitions in single crystals: A new molecular dynamics method." *J. Appl. Phys.* 1981;52:7182-7190.

[23] Darden T et al. "Particle mesh Ewald (PME)." *J. Chem. Phys.* 1993;98:10089-10092.

[24] Hess B et al. "LINCS: A linear constraint solver for molecular simulations." *J. Comput. Chem.* 1997;18:1463-1472.

[25] Jumper J et al. "Highly accurate protein structure prediction with AlphaFold." *Nature* 2021;596:583-589.

[26] Michaud-Agrawal N et al. "MDAnalysis: A toolkit for the analysis of molecular dynamics simulations." *J. Comput. Chem.* 2011;32:2319-2327.

[27] Martínez L et al. "PACKMOL: A package for building initial configurations for molecular dynamics simulations." *J. Comput. Chem.* 2009;30:2157-2164.

[28] Sousa da Silva AW, Vranken WF. "ACPYPE — AnteChamber PYthon Parser interfacE." *BMC Res. Notes* 2012;5:367.

[29] O'Boyle NM et al. "Open Babel: An open chemical toolbox." *J. Cheminform.* 2011;3:33.

[30] Poma A et al. "Solid-phase synthesis of molecularly imprinted polymer nanoparticles (nanoMIPs)." *Adv. Funct. Mater.* 2013;23:5537-5543.

[31] Eastman P et al. "OpenMM 7: Rapid development of high performance algorithms for molecular dynamics." *PLOS Comput. Biol.* 2017;13:e1005659.

**Selectivity 방법론**

[32] Garcia-Ortegon M et al. "DOCKSTRING: Easy Molecular Docking Yields Better Benchmarks for Ligand Design — selectivity scoring." *J. Chem. Inf. Model.* 2022;62:3486-3502.

[33] Mestres J et al. "The selectivity entropy as a single value to express inhibitor selectivity." *BMC Bioinformatics* 2011;12:94.

**검증**

[34] Kowalczyk A et al. "SPR and QCM-D for CD9/CD63/CD81." *Anal. Chem.* 2023;95:9520-9530.

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
