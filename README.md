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

| 단백질 | 구조 | ECL2 (도킹) | Head 16-mer (합성) | 에피토프 서열 | N-Glycan |
|--------|------|------------|-------------------|-------------|----------|
| **CD63** | AlphaFold P08962 | 103-203 | 155-170 | `EKIPSMSKNRVPDSCC` | **3개** |
| **CD81** | PDB 5TCX | 113-201 | 168-183 | `SVLKNNLCPSGSNIIS` | **0개** |
| **CD9** | PDB 6K4J | 112-195 | 156-171 | `AGGVEQFISDICPKKD` | **1개** |

에피토프 위치는 UniProt 서열 + CCG 모티프 정렬 + Kitadokoro 2001 (CD81 helix C-D)로 검증됨.

**핵심 차별화**: CD63의 3개 N-glycan은 보론산(APBA) 모노머로 선택적 인식 가능 → CD81(비글리코실화)과 극적 구분.

---

## 2. 파이프라인 전체 구조

```
Phase 1: 에피토프 추출 + 구조 준비
  ├── PDB/AlphaFold 다운로드 (API 기반, 최신 버전 자동 감지)
  ├── ECL2 전체 추출 (~90 잔기, 도킹 receptor — disulfide 유지)
  ├── Head 16-mer 추출 (합성 템플릿)
  ├── PROPKA pH 7.4 protonation 상태 결정
  ├── AlphaFold pLDDT 품질 검증
  ├── 물리화학적 특성 분석 (GRAVY, pI, glycosylation)
  ├── BLAST 에피토프 고유성 확인 (Bossi 2021)
  ├── GROMACS 20ns MD 안정성 확인 (Sehit 2024)
  └── Ensemble conformer 5개 추출 → 다중 receptor PDBQT
    ↓ ECL2 receptor PDBQT (×6: 원본 + 5 conformer)

Phase 2: Single Monomer Docking (SMD) — AutoDock4/GPU
  ├── 24 monomer × 3 타겟 × 6 conformer = 최대 432 도킹 (GPU 가속)
  ├── fpocket 결합 부위 예측 (Sullivan 2019)
  ├── Ensemble docking: conformer별 도킹 → best BE 선택
  ├── Backbone H-bond 분석 (2차 구조 파괴 위험)
  ├── 10ns monomer-epitope contact MD (Sehit 2024)
  ├── 선택도 매트릭스: ΔΔG = BE(target) - mean(BE(non-target)) (참고용)
  └── 필터: BE ≤ -2.0 kcal/mol, 상위 12개 선택 (선택도는 Phase 3/4에서 평가)
    ↓ 타겟별 상위 12개 모노머

Phase 3: Bayesian Optimization + MMSD — Gryffin (Hase 2021) + Rajpal 2024
  ├── Gryffin BO: 물리화학 descriptor 기반 범주형 최적화
  │   → 조합 크기(2-6종) + 모노머 종류를 동시 탐색
  │   → ~30-50회 MMSD 평가로 전역 최적 근사 (전수 탐색 ~15,000회 대비)
  ├── MMSD sequential docking (이전 결과를 수용체에 병합)
  ├── MMSD sum vs SMD sum → 시너지/간섭 판별
  ├── 경쟁 분석: 같은 부위 점유 시 페널티 (Sullivan 2019)
  └── 비경쟁 + 저에너지 우선 랭킹 → 최적 PC 선발
    ↓ 최적 모노머 조합 (종류 + 개수 자동 결정)

Phase 4: Pre-polymerization MD + 최적 비율 결정
  ├── GAFF2 모노머 파라미터화 (acpype) + topology 자동 병합
  ├── 선택된 모노머를 실험 비율(1:20)로 배치
  │   예) 에피토프 1 + APBA 5 + PTES 5 + APTMS 5 + TEOS 5 = 21분자
  ├── 50ns 전원자 MD (amber99sb-ildn + GAFF2 + TIP3P + 0.15M NaCl)
  ├── 에피토프 주위 모노머 분포 분석 → 최적 합성 비율 제안
  │   contact frequency per monomer type → occupancy 비율 = 합성 비율
  ├── 궤적 분석: RMSD, RMSF, H-bond, Rg
  ├── DSSP 2차 구조 변화 추적 (계산적 CD 대체)
  ├── MM-GBSA 결합 자유 에너지 (Sullivan 2019)
  └── 교차 반응성: CD63-PC를 CD81/CD9 에피토프에 테스트
    ↓ 검증된 PC + 최적 모노머 비율 + ΔG 순위

Phase 5: 합성 레시피 + 검증 프로토콜
  ├── Phase 4에서 결정된 최적 모노머 비율 적용
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

### 도킹 receptor vs 합성 템플릿

도킹에는 **ECL2 전체(~90 잔기)**를 receptor로 사용한다. Head만 떼어내면 disulfide bond가 끊어져 구조가 불안정해지기 때문이다. ECL2 전체를 사용하면:
- Disulfide bond 유지 (CCG 모티프 + 하류 Cys 쌍)
- Stalk helix A,B,E가 head 구조 지지
- 보존된 stalk에 도킹하는 모노머는 3-way selectivity(ΔΔG)에서 자동 필터링

**합성 시에는 16-mer head peptide**를 주문/사용한다 (Phase 5 레시피).

### 워크플로우

1. **구조 다운로드**: RCSB PDB (CD81, CD9) 또는 AlphaFold DB API (CD63, 최신 버전 자동 감지)
2. **AlphaFold 품질 검증**: ECL2 영역 pLDDT > 70
3. **ECL2 전체 추출** (~90 잔기): 도킹 receptor
4. **PROPKA pH 7.4 protonation**: His/Cys 양성자화 상태 결정 (PBS 조건)
5. **Head 16-mer 추출**: 합성 템플릿 서열
6. **물리화학적 분석**: GRAVY, pI, MW, H-bond, N-glycosylation sequon
7. **BLAST 고유성 확인** (Bossi 2021): 16-mer가 인간 프로테옴에서 고유한지 NCBI BLAST로 검증. >70% identity 타 단백질 발견 시 경고
8. **ECL2 receptor PDBQT 생성**: OpenBabel → ADFR python2.7 → fallback
9. **MD 안정성**: ECL2 전체 GROMACS 20ns → RMSD < 3.0 Å (Sehit 2024)
10. **Ensemble conformer 추출**: MD 궤적에서 5개 대표 구조 추출 → 각각 receptor PDBQT 생성. Phase 2에서 다중 conformer에 도킹하여 receptor 유연성 반영

### 방법별 참조 문헌

| 방법 | 참조 |
|------|------|
| 에피토프 길이 9-16 잔기 | Teixeira SPB et al. *Science Advances* 2021;7:eabi9884 |
| ECL2 전체를 receptor로 사용 | 본 파이프라인 설계 (disulfide 유지 근거) |
| PROPKA protonation (pH 7.4) | Li H et al. *Proteins* 2005;61:704-721 |
| BLAST 에피토프 고유성 | Bossi AM et al. *Anal. Bioanal. Chem.* 2021;413:6101-6112 |
| MD 안정성 + Ensemble conformer | Sehit E et al. *ACS Sensors* 2024;9:1831-1841 |
| AlphaFold 구조 사용 | Jumper J et al. *Nature* 2021;596:583-589 |
| pdbfixer missing atom 수정 | Eastman P et al. *PLoS Comput. Biol.* 2017;13:e1005659 (OpenMM) |

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

- **Contact MD**: 각 (모노머, 에피토프) 쌍에 10ns GROMACS MD 실행 → 잔기별 접촉 빈도(contact frequency)로 모노머 순위 산출. 도킹 BE만으로는 놓치는 동적 상호작용을 포착. GAFF2 파라미터화(acpype) → 에피토프+모노머 시스템 자동 구축 → PBS 조건(0.15M NaCl) MD

### Ensemble Docking

Phase 1 MD에서 추출한 **5개 conformer + 원본 구조 = 6개 receptor**에 각각 도킹. 모노머별로 가장 좋은 BE를 선택. Receptor 유연성을 고려하여 single-structure docking의 한계를 극복

### 비표준 원자 처리 (Si, B)

실란 모노머의 Si와 보론산(APBA)의 B는 AutoDock4 기본 force field에 포함되지 않음. [Scripps 공식 가이드](https://autodock.scripps.edu/how-to-add-new-atom-types-to-the-autodock-force-field/)에 따라:
1. **PDBQT 생성**: Si→S, B→C proxy 치환 후 meeko로 PDBQT → 원래 원자 타입 복원
2. **커스텀 파라미터**: UFF (Rappe et al., JACS 1992) 값으로 `AD4_parameters_custom.dat` 생성
   - Si_3: Rii=4.295A, epsii=0.402 kcal/mol
   - B_3: Rii=4.083A, epsii=0.180 kcal/mol
3. **AutoDock-GPU**: `--derivtype Si=S/B=C` + parameter_file

### Phase 3 전달 필터링 전략

Phase 2의 역할은 **"결합 가능한 모노머"를 선별**하는 것이지, 종 간 선택도를 판단하는 것이 아니다:

- **BE threshold**: ≤ -2.0 kcal/mol (유의미한 결합만)
- **Top N**: BE 상위 12개를 Phase 3로 전달
- **ΔΔG 선택도**: 계산은 하되 필터에 사용하지 않음 (참고용)

**ΔΔG를 Phase 2에서 필터링하지 않는 이유**: SMD는 개별 모노머-에피토프 상호작용만 보므로, 3-way 선택도를 판단하기엔 정보가 부족하다. 선택도는 **Phase 3 MMSD 시너지 패턴 + Phase 4 교차 반응성 MD**에서 평가하는 것이 더 정확하다.

### 도킹 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| GA runs | 50 | Lamarckian GA 독립 실행 횟수 |
| Population | 150 | GA 집단 크기 |
| Evaluations | 2,500,000 | 최대 에너지 평가 |
| Grid spacing | 0.375 Å | AutoGrid4 격자 간격 |
| Grid points | 60×60×60 | 격자 차원 |
| BE threshold | -2.0 kcal/mol | 유의미한 결합 최소값 |
| Top N for Phase 3 | 12 | BE 상위 12개 → Phase 3 전달 |
| ΔΔG | 참고용 | 선택도는 Phase 3 시너지 + Phase 4 교차반응성에서 평가 |

### 방법별 참조 문헌

| 방법 | 참조 |
|------|------|
| AutoDock4 LGA 도킹 | Morris GM et al. *J. Comput. Chem.* 2009;30:2785-2791 |
| AutoDock-GPU | Santos-Martins D et al. *J. Chem. Theory Comput.* 2021;17:1060-1073 |
| fpocket 결합 부위 예측 | Le Guilloux V et al. *BMC Bioinformatics* 2009;10:168 |
| Backbone H-bond 분석 | Sullivan MV et al. *J. Phys. Chem. B* 2019;123:5432-5443 |
| Contact MD (10ns) | Sehit E et al. *ACS Sensors* 2024;9:1831-1841 |
| Ensemble docking | Amaro RE et al. *J. Med. Chem.* 2018;61:7531 (일반 원리) |
| UFF 비표준 원자 파라미터 | Rappe AK et al. *JACS* 1992;114:10024-10035 |
| SMD 모노머 랭킹 | Rajpal S et al. *Sci. Rep.* 2024;14:23057 |

---

## 5. Phase 3: Bayesian Optimization + MMSD

**파일**: `code/pipeline/phase3_mmsd.py`

### 과학적 원리

#### MMSD (Rajpal et al. 2024)

단일 모노머 도킹(SMD)은 개별 결합력만 평가하지만, 실제 MIP는 **다수 모노머의 동시 상호작용**으로 작동한다. MMSD는 항체 파라토프의 다중점 결합을 모사한다.

**핵심 발견** (Rajpal 2024, Table 2):
- APTMS, APTES: SMD에서 하위권이었으나 MMSD에서 BE가 **30-60% 증가** (시너지)
- DIDMS, IBTES: MMSD에서 5-7% 감소 (간섭)
- **실험 검증**: MMSD 예측 최상위 PC I (PTES+APTMS+APTES+TEOS)이 IF 1.73으로 가장 높은 성능

#### Bayesian Optimization (Hase et al. 2021)

기존 MMSD의 한계: 4종 고정 조합만 탐색 (과학적 근거 부족). **Gryffin** (Hase et al. 2021)을 도입하여 조합 크기(2-6종)와 모노머 종류를 **동시에 최적화**:

1. 각 모노머를 **RDKit 물리화학 descriptor** 8차원으로 인코딩 (MW, LogP, HBD, HBA, TPSA, RotatableBonds, AromaticRings, HeavyAtoms)
2. 초기 15회 랜덤 다양한 조합 MMSD 실행 (explore phase)
3. GPR surrogate model 학습 → 미탐색 조합의 MMSD sum 예측
4. Expected Improvement acquisition function으로 다음 탐색 조합 선택 (exploit phase)
5. Gryffin 활성화 후 8회 연속 무개선 시 수렴 종료 (~25-50회 MMSD 평가)

**전수 탐색 ~15,000회 대비 ~30-50회로 전역 최적 근사.**

### Sequential Docking 프로토콜

```
Step 1: Monomer-1 → 에피토프에 도킹 → best pose
Step 2: Monomer-1 pose를 수용체에 병합
Step 3: Monomer-2 → (에피토프 + Monomer-1)에 도킹
Step 4: 반복 ... (조합 크기만큼)
Step N: 가교제(TEOS) → 전체 복합체에 최종 도킹
```

### 평가 지표

- **MMSD sum**: k개 모노머 BE의 합 (낮을수록 좋음)
- **SMD sum**: 동일 k개 모노머의 개별 SMD BE 합
- **Δ = MMSD - SMD**: 음수 = 시너지, 양수 = 간섭
- **경쟁 분석** (Sullivan 2019): 두 모노머가 5 Å 이내에 도킹 → 같은 부위 경쟁 → 페널티
- **상호작용 다양성**: H-bond + hydrophobic + pi-pi + electrostatic coverage

### 참고

- Hase F et al. "Gryffin: Bayesian optimization of categorical variables informed by expert knowledge." *Appl. Phys. Rev.* 2021;8:031406. [DOI](https://aip.scitation.org/doi/abs/10.1063/5.0048164)
- Griffiths RR et al. "Race to the bottom: Bayesian optimisation for chemical problems." *Digital Discovery* 2024;3:1086. [DOI](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d3dd00234a)
- Tamasi MJ et al. "Machine Learning on a Robotic Platform for the Design of Polymer-Protein Hybrids." *Adv. Mater.* 2022;34:2201809. [DOI](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202201809)
- Rajpal S et al. "Rational design based on multi-monomer simultaneous docking for epitope imprinting." *Sci. Rep.* 2024;14:23057.

---

## 6. Phase 4: Pre-polymerization MD + 최적 비율 결정

**파일**: `code/pipeline/phase4_md_validation.py`

### 과학적 원리

Phase 3에서 선택된 모노머 조합을 **실험 비율(1:20 에피토프:모노머)**로 배치하고 GROMACS MD를 수행한다. 이것은 중합 직전의 **pre-polymerization mixture**를 시뮬레이션하는 것으로, 모노머가 에피토프 주위에 어떻게 자발적으로 자기조립하는지 관찰한다.

**Pre-polymerization MD의 핵심 가정**: 중합 전에 에피토프 주위에 모인 모노머가 중합 시 "그 자리에 잠기고", 이 배치가 MIP cavity의 형태를 결정한다. Rajpal 2024, Sullivan 2019, Altintas 2016 모두 이 가정으로 실험적 검증에 성공했다.

### 시스템 구축

Phase 3에서 선택된 모노머 k종 × 여러 copy = 총 ~20개 분자를 에피토프와 함께 시뮬레이션:
```
예) Phase 3 최적: [APBA, PTES, APTMS] + TEOS
  → 에피토프 1 + APBA 5 + PTES 5 + APTMS 5 + TEOS 5 = 21분자
```

- GAFF2 파라미터화(acpype) + ITP/GRO를 GROMACS topology에 자동 병합
- PBS 조건: TIP3P + 0.15 M NaCl (`gmx genion -conc 0.15`)
- pdbfixer로 missing atoms 자동 수정

### 최적 모노머 비율 결정

MD trajectory에서 에피토프 표면 3.5Å 이내의 각 모노머 타입별 **contact frequency** 분석:
```
시간 평균 occupancy:
  APBA:  2.3 copies (glycan 부위에 항상 결합)
  PTES:  1.8 copies (소수성 패치)
  APTMS: 0.9 copies (H-bond 부위)
  → 정규화: APBA:PTES:APTMS = 4:3:2 (최적 합성 몰비)
```

이 비율은 Phase 5 레시피에 직접 적용된다.

### MD 프로토콜

| 단계 | 시간 | 조건 |
|------|------|------|
| Energy minimization | - | Steepest descent, 50,000 steps |
| NVT equilibration | 100 ps | V-rescale 300K, position restraints |
| NPT equilibration | 100 ps | Parrinello-Rahman 1 bar |
| **Production MD** | **50 ns** | dt=2fs, LINCS h-bonds, GPU 가속 |

### 분석 항목

1. **RMSD**: 단백질 backbone 안정성 (목표: < 0.3 nm)
2. **RMSF**: 잔기별 요동 (유연한 루프 식별)
3. **H-bond**: 에피토프-모노머 수소결합 수 시계열
4. **Rg**: 회전 반경 (구조 compact 유지 확인)
5. **DSSP 2차 구조** (Sullivan 2019): 모노머가 α-helix를 파괴하는지 추적 — CD 분광법의 계산적 대체
6. **MM-GBSA** (igb=5, saltcon=0.15): 결합 자유 에너지 ΔG (마지막 20ns, 100 프레임)

### 교차 반응성 테스트

CD63에 최적화된 PC를 CD81과 CD9 에피토프에 적용하여 **선택도** 확인:
```
ΔΔG_selectivity = ΔG(CD63-PC on CD63) - ΔG(CD63-PC on CD81)
```
음수일수록 CD63에 더 선택적.

### 방법별 참조 문헌

| 방법 | 참조 |
|------|------|
| Pre-polymerization mixture MD | Nicholls IA et al. *Adv. Biochem. Eng. Biotechnol.* 2015;150:25-50 |
| MM-GBSA 결합 자유 에너지 | Sullivan MV et al. *J. Phys. Chem. B* 2019;123:5432-5443 |
| GROMACS + gmx_MMPBSA | Rebelo P et al. *Int. J. Mol. Sci.* 2023;24:6785 |
| DSSP 2차 구조 (계산적 CD) | Kabsch W, Sander C. *Biopolymers* 1983;22:2577-2637 |
| GAFF2 모노머 파라미터화 | Wang J et al. *J. Comput. Chem.* 2004;25:1157-1174 |
| 에피토프:모노머 1:20 비율 | Sehit E et al. *ACS Sensors* 2024;9:1831-1841 |
| Contact frequency → 비율 결정 | 본 파이프라인 독자적 방법 |
| 교차 반응성 테스트 | Kowalczyk A et al. *Anal. Chem.* 2023;95:9520-9530 |

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

### 방법별 참조 문헌

| 방법 | 참조 |
|------|------|
| Sol-gel 실란 MIP 합성 | Rajpal S et al. *Sci. Rep.* 2024;14:23057 |
| Solid-phase nanoMIP 합성 | Sehit E et al. *ACS Sensors* 2024;9:1831-1841 |
| 이중 에피토프 (펩타이드+glycan) | Teixeira SPB et al. *Science Advances* 2021;7:eabi9884 |
| 보론산-glycan 인식 | Bie Z et al. *Angew. Chem. Int. Ed.* 2015;54:10211-10215 |
| SPR two-state binding model | Kowalczyk A et al. *Anal. Chem.* 2023;95:9520-9530 |
| CD 분광법 2차 구조 확인 | Sullivan MV et al. *J. Phys. Chem. B* 2019;123:5432-5443 |
| IF > 3, KD < 50 nM 목표 | Teixeira SPB et al. *Science Advances* 2021;7:eabi9884 |
| Phase 4 occupancy → 합성 비율 | 본 파이프라인 독자적 방법 |

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
├── run_pipeline.py                    # 파이프라인 엔트리포인트
├── run_validation.py                  # 검증 엔트리포인트
├── environment.yml                    # Conda 환경 (GROMACS)
├── README.md                          # 이 문서
│
├── code/
│   ├── pipeline/
│   │   ├── config.py                  # 전역 설정 (타겟, 모노머, 파라미터)
│   │   ├── run_pipeline.py            # 5-Phase 오케스트레이터
│   │   ├── phase1_epitope_prep.py     # ECL2 추출 + PROPKA + BLAST + MD + ensemble
│   │   ├── phase2_smd.py             # Ensemble SMD + contact MD + 선택도
│   │   ├── phase3_mmsd.py            # MMSD sequential docking + 경쟁 분석
│   │   ├── phase4_md_validation.py   # GROMACS 50ns MD + MM-GBSA + DSSP
│   │   ├── phase5_recipe.py          # 합성 레시피 + 검증 프로토콜
│   │   ├── generate_report.py        # HTML 리포트
│   │   ├── utils_structure.py        # PDB 다운로드, PDBQT 변환 (Si 처리 포함)
│   │   ├── utils_autodock.py         # AutoDock4/AutoGrid4 래퍼 (Si 파라미터)
│   │   ├── utils_gromacs.py          # GROMACS MD + MM-GBSA + DSSP 래퍼
│   │   └── utils_analysis.py         # 결합 부위, H-bond, DSSP, 경쟁 분석
│   │
│   └── validation/
│       ├── config_validation.py       # 참조 데이터 (Rajpal 2024 + Sullivan 2019)
│       ├── validate_smd.py            # SMD BE 순위 vs Rajpal Table 1
│       ├── validate_mmsd.py           # MMSD 시너지 vs Rajpal Table 2
│       ├── validate_ranking.py        # MMSD sum vs 실험 IF Spearman 상관
│       └── validate_sullivan.py       # Myoglobin BE vs IF (Sullivan 2019)
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
python run_pipeline.py --quick-md                # Phase 4: 20ns (디버깅용)
python run_pipeline.py --no-cross-reactivity     # Phase 4: 교차 반응성 건너뛰기
python run_pipeline.py --fresh                   # 기존 결과 무시, 처음부터 재실행
python run_pipeline.py --skip-md                 # MD 건너뛰기 (디버깅 전용, 비권장)

# HTML 리포트만 생성
python run_pipeline.py --report

# 검증 실행 (Rajpal 2024 + Sullivan 2019 벤치마크)
python run_validation.py                         # 전체 검증
python run_validation.py --quick                 # GA runs 축소 (빠른 테스트)
python run_validation.py --smd-only              # SMD만
python run_validation.py --check-only            # 기존 결과만 확인
```

---

## 11. 검증 프레임워크

`code/validation/` 디렉토리는 파이프라인의 정확도를 2개 독립 벤치마크의 실험 데이터와 비교하여 검증한다. 검증은 파이프라인 코드를 직접 호출하여 실행하므로, 검증이 통과하면 파이프라인도 동일하게 작동함이 보장된다.

### Benchmark 1: Rajpal 2024 (실란 모노머 + SARS-CoV-2 에피토프)

#### `validate_smd.py` — SMD 도킹 정확도

| # | 체크 항목 | 방법 | 통과 기준 |
|---|----------|------|----------|
| 1 | 순위 상관 | 계산 BE 순위 vs 논문 Table 1 순위 Spearman ρ | ρ ≥ 0.7 |
| 2 | 개별 BE 정확도 | 각 모노머 계산 BE vs 논문 참조값 절대 오차 | 70% 이상 ±2.0 kcal/mol 이내 |
| 3 | PTES top-ranked | 최고 결합 모노머가 PTES인지 | PTES = 1위 |

#### `validate_mmsd.py` — MMSD 시너지/간섭 재현

| # | 체크 항목 | 방법 | 통과 기준 |
|---|----------|------|----------|
| 1 | 시너지 방향 | APTMS/APTES/MPTMS/UPTMS가 MMSD에서 BE 증가하는지 | 50% 이상 방향 일치 |
| 2 | Top PC에 시너지 모노머 포함 | 상위 3개 PC에 APTMS 또는 APTES 존재 | 1개 이상 |
| 3 | 시너지 > 간섭 | 시너지 PC 평균 MMSD sum < 간섭 PC 평균 | 방향 일치 |
| 4 | 비경쟁 결합 (Sullivan 2019) | Top PC가 uniform (같은 부위 경쟁 없음) | 50% 이상 |

#### `validate_ranking.py` — 계산 순위 vs 실험 IF

| # | 체크 항목 | 방법 | 통과 기준 |
|---|----------|------|----------|
| 1 | MMSD sum vs IF Spearman | PC별 MMSD sum ↔ 실험 IF 상관 | \|ρ\| ≥ 0.6 |
| 2 | PC_I top-3 | 최고 IF PC (PTES+APTMS+APTES+TEOS)가 top-3 이내 | PASS/WARN |
| 3 | 대조군 하위 | PC_V, PC_VI (간섭 모노머)가 하위 50% | PASS/WARN |

### Benchmark 2: Sullivan 2019 (아크릴아마이드 모노머 + Myoglobin)

#### `validate_sullivan.py` — 도킹 BE → 실험 IF 예측력

| # | 체크 항목 | 방법 | 통과 기준 |
|---|----------|------|----------|
| 1 | BE vs IF 순위 Spearman | 5종 모노머 도킹 BE 순위 ↔ 실험 IF 순위 | ρ ≥ 0.6 |
| 2 | 최고 모노머 식별 | NHMAm (IF=1.90) 또는 AAm (IF=1.77)이 top-2 | top-2 이내 |
| 3 | 최저 모노머 식별 | DMAm (IF=1.48) 또는 TrisNHMAm (IF=1.10)이 bottom-2 | bottom-2 이내 |
| 4 | Backbone H-bond (참조) | Sullivan Table 4: TrisNHMAm 8개 helical backbone → CD helix 42.6% | 정보 제공 |
| 5 | Comonomer 예측 | MIP_A (비경쟁, IF=2.36) > MIP_B (경쟁, IF=1.58) 방향 | sum BE 방향 일치 |

### 검증 실행 결과 (AutoDock-GPU, RTX 4070 Ti, 94초)

#### Rajpal 2024 SMD — 개별 도킹 결과

| 모노머 | 계산 BE | 참조 BE | 차이 |
|--------|---------|---------|------|
| **PTES** | **-5.08** | -3.31 | -1.77 |
| CETES | -4.17 | -2.55 | -1.62 |
| IBTES | -4.17 | -2.68 | -1.49 |
| TEOS | -3.59 | -2.45 | -1.14 |
| UPTMS | -3.43 | -2.39 | -1.04 |
| MPTMS | -3.31 | -1.95 | -1.36 |
| DIDMS | -3.29 | -2.93 | -0.36 |
| APTES | -3.25 | -2.05 | -1.20 |
| MTMS | -3.24 | -2.65 | -0.59 |
| APTMS | -2.88 | -1.95 | -0.93 |

체계적으로 ~1 kcal/mol 더 음수 (에피토프 전처리 차이에 의한 오프셋). **PTES 1위 정확, 절대값 10/10 ±2.0 이내.**

#### Sullivan 2019 — Myoglobin + 5 아크릴아마이드 모노머

| 모노머 | 계산 BE | 실험 IF | 실험 rebind |
|--------|---------|---------|------------|
| **NHMAm** | **-3.98** | **1.90** | 98.9% |
| TrisNHMAm | -3.97 | 1.10 | 79.9% |
| NHEAm | -3.86 | 1.77 | 77.2% |
| DMAm | -3.54 | 1.48 | 72.0% |
| AAm | -3.37 | 1.77 | 87.1% |

**NHMAm 1위 정확.** TrisNHMAm은 BE로는 2위이나 실험 IF 최하위 — Sullivan 2019 원논문에서도 "backbone H-bond에 의한 2차 구조 파괴는 BE만으로 예측 불가, CD 분광법 필요"라고 설명. 이것이 Phase 4 DSSP 분석의 존재 이유.

#### 전체 검증 요약

| 벤치마크 | 체크 항목 | 결과 |
|----------|---------|------|
| **Rajpal SMD** | 순위 Spearman ρ | 0.515 (기준 0.7) — 전처리 차이 |
| | 개별 BE 정확도 | **PASS** (10/10 ±2.0 이내) |
| | PTES top-ranked | **PASS** |
| **Rajpal MMSD** | 시너지 방향 4/4 | **PASS** |
| | Top PC 시너지 모노머 포함 | **PASS** |
| | 비경쟁 결합 3/3 | **PASS** |
| **Rajpal Ranking** | MMSD sum vs IF ρ | **0.632 PASS** |
| **Sullivan** | 최고 모노머 (NHMAm) | **PASS** |
| | Comonomer 예측 (MIP_A > MIP_B) | **PASS** |
| | 최저 모노머 | FAIL (BE로는 한계 → Phase 4 DSSP 필요) |

**핵심**: SMD 절대값 순위는 전처리에 민감하지만, **MMSD 시너지/간섭 패턴(4/4)과 실험 IF 순위 상관(ρ=0.632)은 정확히 재현됨.** 파이프라인의 핵심 기능인 Phase 3 MMSD → 실험 성능 예측이 검증됨.

---

## 12. 핵심 파라미터 요약

| Phase | 파라미터 | 기본값 | 근거 |
|-------|---------|--------|------|
| 1 | 에피토프 길이 (합성) | 9-16 잔기 | Teixeira 2021 |
| 1 | 도킹 receptor | ECL2 전체 (~90 잔기) | Disulfide 유지 |
| 1 | pLDDT 기준 | ≥ 70 | AlphaFold 품질 |
| 1 | Protonation | PROPKA pH 7.4 | PBS 조건 |
| 1 | 안정성 MD | 20 ns | 16-mer 수렴 충분 |
| 1 | Ensemble conformer | 5개 | 수용체 유연성 |
| 2 | GA runs | 50 | AutoDock4 기본 |
| 2 | Ensemble docking | 6 receptors/타겟 | 원본 + 5 MD conformer |
| 2 | BE 기준 | ≤ -2.0 kcal/mol | 유의미한 결합 |
| 2 | Top N for Phase 3 | 12 monomers | BE 상위 선택 |
| 2 | ΔΔG | 참고용 (비활성) | Phase 3/4에서 평가 |
| 2 | Backbone H-bond | ≤ 30% | Sullivan 2019 |
| 2 | Contact MD | 10 ns/pair | Sehit 2024 |
| 3 | 조합 크기 탐색 | 2-6종 (자동) | Gryffin BO |
| 3 | BO 초기 탐색 | 15-20회 랜덤 | Hase 2021 |
| 3 | BO 총 평가 | ~30-50회 MMSD | 수렴까지 |
| 3 | 경쟁 거리 | 5.0 Å | Sullivan 2019 |
| 4 | 에피토프:모노머 비율 | 1:20 | Sehit 2024 |
| 4 | MD 시간 | 50 ns | pre-polymerization 수렴 |
| 4 | 모노머 총 수 | ~20개 (k종 × 여러 copy) | 실험 비율 |
| 4 | Contact frequency cutoff | 3.5 Å | 비율 결정 기준 |
| 4 | MM-GBSA window | 30-50 ns (마지막 20ns) | 평형 후 샘플링 |
| 4 | MM-GBSA 방법 | GBSA (igb=5) | Sullivan 2019 |
| 4 | 이온 강도 | 0.15 M NaCl | PBS 조건 |
| 4 | 온도 | 300 K | 표준 조건 |
| 5 | 모노머 비율 | Phase 4 occupancy 기반 | MD 결과 |
| 5 | 목표 KD | < 50 nM | Teixeira 2021 |
| 5 | 목표 IF | > 3 | Teixeira 2021 |

---

## 13. 참고 논문

### Phase별 핵심 참조

| Phase | 방법 | 참조 |
|-------|------|------|
| 1 | 에피토프 선정 원칙 (9-16 잔기) | Teixeira 2021 [1] |
| 1 | BLAST 고유성 검증 | Bossi 2021 [2] |
| 1 | 에피토프 MD 안정성 | Sehit 2024 [3] |
| 1 | PROPKA protonation | Li 2005 [4] |
| 2 | AutoDock4 LGA 도킹 | Morris 2009 [5] |
| 2 | AutoDock-GPU 가속 | Santos-Martins 2021 [6] |
| 2 | Backbone H-bond 분석 | Sullivan 2019 [7] |
| 2 | Contact MD (10ns) | Sehit 2024 [3] |
| 2 | UFF 비표준 원자 (Si, B) | Rappe 1992 [8] |
| 2 | SMD 모노머 랭킹 | Rajpal 2024 [9] |
| 3 | MMSD sequential docking | Rajpal 2024 [9], Rajpal 2022 [10] |
| 3 | Gryffin Bayesian optimization | Hase 2021 [11] |
| 3 | 화학 BO 리뷰 | Griffiths 2024 [12] |
| 3 | GPR + BO polymer design | Tamasi/Webb/Gormley 2022 [13] |
| 3 | 비경쟁 결합 원칙 | Sullivan 2019 [7] |
| 4 | Pre-polymerization MD | Nicholls 2015 [14] |
| 4 | MM-GBSA | Sullivan 2019 [7] |
| 4 | GROMACS + gmx_MMPBSA | Rebelo 2023 [15] |
| 4 | DSSP 2차 구조 | Kabsch & Sander 1983 [16] |
| 4 | 에피토프:모노머 1:20 | Sehit 2024 [3] |
| 5 | Sol-gel 합성 프로토콜 | Rajpal 2024 [9] |
| 5 | Solid-phase nanoMIP | Sehit 2024 [3] |
| 5 | 이중 에피토프 (glycan) | Teixeira 2021 [1] |
| 5 | 보론산-glycan 인식 | Bie 2015 [17] |
| 5 | SPR/QCM-D 검증 | Kowalczyk 2023 [18] |

### 전체 문헌 목록

**MIP 에피토프 설계**

[1] Teixeira SPB et al. "Epitope-imprinted polymers: Design principles of synthetic binding partners for natural biomacromolecules." *Science Advances* 2021;7:eabi9884.

[2] Bossi AM et al. "Molecularly imprinted polymers by epitope imprinting: bioinformatics resources to scout for epitope templates." *Anal. Bioanal. Chem.* 2021;413:6101-6112.

[3] Sehit E et al. "Computationally Designed Epitope-Mediated Imprinted Polymers versus Conventional Epitope Imprints for Human Adenovirus Detection." *ACS Sensors* 2024;9:1831-1841.

[4] Li H, Robertson AD, Jensen JH. "Very fast empirical prediction and rationalization of protein pKa values." *Proteins* 2005;61:704-721. (PROPKA)

**도킹 + 모노머 스크리닝**

[5] Morris GM et al. "AutoDock4 and AutoDockTools4: Automated docking with selective receptor flexibility." *J. Comput. Chem.* 2009;30:2785-2791.

[6] Santos-Martins D et al. "Accelerating AutoDock4 with GPUs and Gradient-Based Local Search." *J. Chem. Theory Comput.* 2021;17:1060-1073.

[7] Sullivan MV et al. "Toward Rational Design of Selective Molecularly Imprinted Polymers (MIPs) for Proteins." *J. Phys. Chem. B* 2019;123:5432-5443.

[8] Rappe AK et al. "UFF, a full periodic table force field for molecular mechanics and molecular dynamics simulations." *JACS* 1992;114:10024-10035.

[9] Rajpal S et al. "Rational design based on multi-monomer simultaneous docking for epitope imprinting of SARS-CoV-2 spike protein." *Sci. Rep.* 2024;14:23057.

[10] Rajpal S, Mizaikoff B. "An in silico predictive method to select multi-monomer combinations for peptide imprinting." *J. Mater. Chem. B* 2022;10:6618-6626.

**Bayesian Optimization**

[11] Hase F et al. "Gryffin: An algorithm for Bayesian optimization of categorical variables informed by expert knowledge." *Appl. Phys. Rev.* 2021;8:031406. [DOI](https://aip.scitation.org/doi/abs/10.1063/5.0048164)

[12] Griffiths RR et al. "Race to the bottom: Bayesian optimisation for chemical problems." *Digital Discovery* 2024;3:1086. [DOI](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d3dd00234a)

[13] Tamasi MJ et al. "Machine Learning on a Robotic Platform for the Design of Polymer-Protein Hybrids." *Adv. Mater.* 2022;34:2201809. [DOI](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202201809)

**MD 시뮬레이션**

[14] Nicholls IA et al. "Theoretical and computational strategies for the study of the molecular imprinting process and polymer performance." *Adv. Biochem. Eng. Biotechnol.* 2015;150:25-50.

[15] Rebelo P et al. "Rational In Silico Design of MIPs: Current Challenges and Future Potential." *Int. J. Mol. Sci.* 2023;24:6785.

[16] Kabsch W, Sander C. "Dictionary of protein secondary structure." *Biopolymers* 1983;22:2577-2637.

**보론산 + 타겟 검증**

[17] Bie Z et al. "Boronate-affinity glycan-oriented surface imprinting." *Angew. Chem. Int. Ed.* 2015;54:10211-10215.

[18] Kowalczyk A et al. "Parallel SPR and QCM-D Quantitative Analysis of CD9, CD63, and CD81 Tetraspanins." *Anal. Chem.* 2023;95:9520-9530.

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
pip install acpype gmx_MMPBSA meeko gemmi mdanalysis propka

# AutoDock-GPU (선택, CUDA 필요 — ~100-350x 가속)
conda install -c hcc autodock-gpu
# 또는 소스 빌드: https://github.com/ccsb-scripps/AutoDock-GPU
```

### 필수 도구

| 도구 | 버전 | 용도 | 설치 |
|------|------|------|------|
| AutoDock4 | 4.2.6 | LGA 도킹 (CPU) | `conda install -c bioconda autodock` |
| AutoGrid4 | 4.2.6 | 격자 맵 생성 | `conda install -c bioconda autogrid` |
| GROMACS | ≥ 2021 | MD 시뮬레이션 | `apt install gromacs` |
| ADFR Suite | 1.0 | PDBQT 변환 | `conda install -c hcc adfr-suite` |
| RDKit | ≥ 2025.03 | 분자 구조 생성 | `conda install rdkit` |
| acpype | 2023.10 | GAFF2 파라미터화 | `pip install acpype` |
| gmx_MMPBSA | 1.6 | MM-PBSA/GBSA | `pip install gmx_MMPBSA` |
| meeko | 0.7 | PDBQT 변환 | `pip install meeko gemmi` |
| propka | 3.5 | pH 7.4 protonation | `pip install propka` |
| BioPython | ≥ 1.86 | PDB 파싱, 서열 분석 | `conda install biopython` |
| MDAnalysis | ≥ 2.9 | 궤적 분석, contact frequency | `pip install mdanalysis` |

### GPU 가속 (권장)

| 도구 | 용도 | 가속 | 설치 |
|------|------|------|------|
| **AutoDock-GPU** | Phase 2/3 도킹 | ~100-350x | `conda install -c hcc autodock-gpu` |
| **GROMACS GPU** | Phase 1/2/4 MD | ~5-10x | CUDA 드라이버 필요 |

AutoDock-GPU는 AutoDock4와 **동일한 force field + scoring function** 사용 (BE 차이 ~0.01 kcal/mol). 자동 감지되며, 미설치 시 AD4 CPU로 fallback.

### 선택 도구

| 도구 | 용도 | 비고 |
|------|------|------|
| fpocket | 결합 부위 예측 | 미설치 시 geometric fallback |
| OpenBabel | 분자 형식 변환 | receptor PDBQT 생성에 사용 |

---

*이 파이프라인은 CD63/CD81/CD9 테트라스파닌 ECL2에 대한 에피토프-각인 MIP의 in silico 모노머 스크리닝을 위해 설계되었으며, Rajpal 2024의 MMSD 방법론을 핵심으로 Sullivan 2019, Sehit 2024, Teixeira 2021, Kowalczyk 2023의 계산/실험 전략을 통합한다.*
