# Bacterial Membrane Composition References

`run_membrane_bacteria.py`에서 사용하는 모든 설정의 문헌 근거를 체계적으로 정리한 문서.

---

## 1. PDB 구조 선택

### 선택 원칙

1. **최신 고해상도 구조 우선**: 동일 단백질의 여러 PDB 중 가장 최근 + 최고 해상도
2. **Wild-type 우선**: 변이체보다 야생형 구조 선택
3. **X-ray 우선**: NMR보다 X-ray (도킹 receptor로 적합)
4. **1990년대 구조는 최신 대안이 없는 경우에만 사용**: 예) OmpA 1BXW (1998)은 여전히 유일한 E.coli OmpA TM 구조

### PDB 목록

| 성분 | PDB | 해상도 | Method | Year | 출처 | 선택 이유 |
|------|-----|--------|--------|------|------|----------|
| OmpF (E.coli) | **4GCP** | 2.4 A | X-ray | 2013 | Ziervogel BK, Roux B. *Structure* 2013;21(1):76-87. | 2OMF(1992, 3.0A) 대비 고해상도. Ampicillin 결합 포즈로 loop 배향이 기능적 상태 반영 |
| OmpA (E.coli) | **1BXW** | 2.5 A | X-ray | 1998 | Pautsch A, Schulz GE. *Nat. Struct. Biol.* 1998;5:1013-1017. [PMID: 9808047](https://pubmed.ncbi.nlm.nih.gov/9808047/) | 유일한 E.coli OmpA TM X-ray 구조. 2025년 현재까지 full-length 실험 구조 없음. AF3 대안 가능하나, beta-barrel 도메인은 X-ray가 정확 |
| OmpC (E.coli) | **2J1N** | 2.0 A | X-ray | 2006 | Basle A et al. *J. Mol. Biol.* 2006;362:933-942. [PMID: 16949612](https://pubmed.ncbi.nlm.nih.gov/16949612/) | OmpC 최고해상도 구조. OmpF와 60% identity, 16-strand barrel trimer |
| OprF (P.aeruginosa) | **4RLC** | 1.6 A | X-ray | 2014 | Kefala G et al. 2014 (RCSB deposited). | OprF N-terminal beta-barrel domain 최고해상도 |
| OprD (P.aeruginosa) | **3SY7** | 2.0 A | X-ray | 2012 | Eren E et al. 2012. | Improved crystal structure. 18-strand monomeric barrel. Carbapenem entry pathway |
| OmpK36 (K.pneumoniae) | **6RD3** | **1.92 A** | X-ray | **2019** | Wong JLC et al. *Nat. Commun.* 2019;10:3733. | **Wild-type** 최고해상도. 5O79(3.2A, Q235R mutant) 대비 우수 |
| Protein A (S.aureus) | **4WWI** | **2.3 A** | X-ray | **2015** | O'Seaghdha M, van Schooten CJ. 2015 (RCSB deposited). | 1BDD(NMR, 1997) 대비 **X-ray crystal**. C-domain + human IgG Fc 복합체 → MIP가 인식할 실제 표면 |
| FnBPA (S.aureus) | **2RKZ** | 2.1 A | X-ray | 2008 | Meenan NAG et al. *PNAS* 2008;105(34):12254-12258. [PMC2518095](https://pmc.ncbi.nlm.nih.gov/articles/PMC2518095/) | 유일한 FnBPA-fibronectin F1 module 복합체 구조 |

### AlphaFold3 vs 1990s X-ray 참고

OmpA 1BXW (1998)에 대해: beta-barrel 도메인은 X-ray가 정확하지만, **extracellular loop** (MIP 인식 부위)은 crystal packing 영향이 있을 수 있어 AF3가 더 나을 수 있다. 단, 현재 파이프라인에서 Phase 1 MD로 구조를 평형화하고 ensemble conformer를 추출하므로, 초기 구조가 X-ray든 AF3든 **MD 이후에는 수렴**한다.

---

## 2. SMILES 검증 (RDKit)

| 성분 | PubChem CID | MW (Da) | Formula | 구조 설명 | RDKit 검증 |
|------|------------|---------|---------|----------|----------|
| **Lipid IVA** | **10329124** | 1405.7 | C68H130N2O23P2 | Tetra-acylated Lipid A 전구체. Bisphosphorylated glucosamine disaccharide + 4x C14 acyl chains | ✅ |
| PE headgroup | - | 215.1 | C5H14NO6P | Glycerophosphoethanolamine. Acyl chain 제거한 headgroup만 | ✅ |
| PG headgroup | - | 246.2 | C6H15O8P | Glycerophosphoglycerol. 음전하 headgroup | ✅ |
| LTA repeating unit | - | 326.1 | C6H16O11P2 | Polyglycerol-phosphate 2개 반복 단위. S.aureus LTA 구조 | ✅ |
| Lys-PG | - | 374.3 | C12H27N2O9P | Lysyl-phosphatidylglycerol. Lysine ester 결합으로 양전하 | ✅ |
| CPS mannose unit | - | 180.2 | C6H12O6 | D-mannose. K.pneumoniae K-antigen 반복 단위의 주요 당 | ✅ |

---

## 3. Lipid IVA 사용 근거

### 3.1 왜 Lipid IVA (CID 10329124)인가

Lipid IVA는 E. coli LPS 생합성 경로의 **핵심 중간체**로, **4개 acyl chain + bisphosphorylated glucosamine disaccharide** 구조이다. TLR4/MD-2 면역 인식 연구에서 **가장 많이 사용되는 표준 Lipid A 형태**.

**선행 문헌:**

1. **Park BS et al. "The structural basis of lipopolysaccharide recognition by the TLR4-MD-2 complex." *Nature* 2009;458:1191-1195.**
   - PDB 3FXI: TLR4/MD-2/LPS 복합체 co-crystal 구조에 Lipid A가 결합

2. **Ryu JK et al. "Conformationally Constrained Lipid A Mimetics for Exploration of TLR4/MD-2 Activation." *ACS Chem. Biol.* 2014;9:2237-2246.**
   - Lipid A 구조 기반 TLR4 작용제 설계. 3FXI의 Lipid A를 scaffold으로 사용

3. **Cochet F, Peri F. "The Role of Carbohydrates in the Lipopolysaccharide (LPS)/Toll-Like Receptor 4 (TLR4) Signalling." *Int. J. Mol. Sci.* 2017;18:2318.**
   - Lipid A/IVA 구조가 TLR4 도킹 연구에서 표준으로 사용됨

### 3.2 Full Lipid A 대비 Lipid IVA 선택 이유

| 형태 | PubChem CID | MW | Acyl chains | Rotatable bonds | AutoDock4 적합 |
|------|------------|-----|-------------|-----------------|---------------|
| Full Lipid A (hexa-acylated) | 9877306 | 1798 Da | 6 | >60 | ❌ torsion 한계 초과 |
| **Lipid IVA (tetra-acylated)** | **10329124** | **1406 Da** | **4** | **~40** | **✅** (receptor로 사용) |

Lipid IVA를 **rigid receptor PDBQT**로 사용하면 torsion 제한이 적용되지 않는다 (ligand인 모노머만 유연).

### 3.3 MIP 도킹에서의 Lipid A 선행 문헌

**Altintas Z et al. "Ultrasensitive detection of endotoxins using computationally designed nanoMIPs." *Anal. Chim. Acta* 2016;902:77-86.**

- E. coli 0111:B4 endotoxin(LPS)을 템플릿으로 사용
- **AutoDock Vina로 21종 모노머를 Lipid A moiety의 phosphate 부분에 도킹**
- Itaconic acid, methacrylic acid, acrylamide가 최적 모노머로 선정
- KD = 4.4-5.3 × 10^-10 M, LOD = 0.44 ng/mL (SPR)
- **핵심**: LPS의 Lipid A 부분이 MIP 모노머 도킹의 타겟 → 우리 접근과 동일

### 3.4 Headgroup-only grid 제한 근거

Gram-negative 박테리아 외막은 **비대칭 이중층**:

```
세포 외부 (수용액, MIP 접근 가능)
──────────────────────────────────
  Phosphate + Sugar headgroup     ← 표면 노출, MIP 인식 가능
  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  Acyl chains (소수성 내부)        ← 막 내부에 매몰, MIP 접근 불가
──────────────────────────────────
  Phospholipid inner leaflet
──────────────────────────────────
세포 내부 (periplasm)
```

**Acyl chain 매몰 근거 문헌:**

1. **Henderson JC et al. "The Power and Promise of Lipopolysaccharide." *Annu. Rev. Microbiol.* 2016;70:255-278.**
   - "Lipid A anchors LPS into the outer leaflet via its acyl chains, which are **buried within the hydrophobic core** of the membrane, while the core oligosaccharide and O-antigen extend into the extracellular milieu."

2. **May KL, Grabowicz M. "The bacterial outer membrane is an evolving antibiotic barrier." *Proc. Natl. Acad. Sci.* 2018;115(36):8852-8854.** [PMC7981291](https://pmc.ncbi.nlm.nih.gov/articles/PMC7981291/)
   - "The outer membrane is an **asymmetric bilayer** with an inner leaflet of glycerophospholipids and an outer leaflet mainly composed of lipopolysaccharide (LPS) molecules."
   - "The **fatty acyl chains** attached to lipopolysaccharide comprising the **hydrophobic portion** of the outer leaflet."

3. **Snyder S et al. "Bilayer Properties of Lipid A from Various Gram-Negative Bacteria." *Biophys. J.* 2016;111:1750-1760.** [PMC5071556](https://pmc.ncbi.nlm.nih.gov/articles/PMC5071556/)
   - "Lipid A is **the lipid anchor** of lipopolysaccharide in the outer leaflet of the outer membrane."
   - Acyl chain packing density와 headgroup 배향의 정량적 분석

4. **Clifton LA et al. "Asymmetric phospholipid: lipopolysaccharide bilayers; a Gram-negative bacterial outer membrane mimic." *J. R. Soc. Interface* 2013;10:20130810.** [PMC3808558](https://pmc.ncbi.nlm.nih.gov/articles/PMC3808558/)
   - 실험적 비대칭 이중층 재구성 → LPS headgroup이 수용액 쪽으로 노출 확인 (neutron reflectometry)

**결론**: Acyl chain은 막의 소수성 내부에 매몰되어 MIP 모노머가 접근할 수 없다. 도킹 grid를 **phosphate + sugar headgroup 영역에만 제한**하는 것이 물리적으로 정당.

### 3.5 Grid 제한 구현

Lipid IVA를 rigid receptor PDBQT로 생성한 뒤:
1. **Grid center**: P(인) 및 N(질소) 원자의 geometric center → headgroup 영역
2. **Grid size**: 40x40x40 points (0.375 A spacing) = ~15x15x15 A box
3. Acyl chain은 grid 밖 → 탐색 대상에서 자동 제외

---

## 3.6 조성 비율 가중합 방법론 근거

본 파이프라인의 핵심 방법 — **여러 막 성분에 대한 도킹 BE를 조성 비율로 가중합하여 전체 막 친화도를 계산** — 의 선행 문헌.

### 직접 선행: 도킹 BE 가중합으로 혼합물 효과 예측

**Yao J et al. "Using molecular docking-based binding energy to predict toxicity of binary mixture with different binding sites." *Chemosphere* 2013;92(9):1169-1176.** [PMID: 23484458](https://pubmed.ncbi.nlm.nih.gov/23484458/)

이 논문이 우리 접근과 **동일한 개념**을 사용:
- **여러 타겟 단백질**에 대한 개별 도킹 결합 에너지(E_binding)를 계산
- **조성 비율로 가중**하여 혼합물의 전체 효과(미생물 독성) 예측
- 3가지 시나리오 검증: (1) 같은 결합 부위, (2) 같은 단백질 다른 부위, (3) **다른 타겟 단백질**
- "binding energy를 사용하여 개별 화학물질이 다른 결합 부위에서 작용하는 방식을 기술함으로써, 미생물에 대한 혼합물 독성을 예측하는 **일반적이고 단순한 모델**"
- AutoDock 사용

| Yao et al. 2013 | 본 파이프라인 |
|-----------------|-------------|
| 여러 타겟 단백질에 화학물질 도킹 | 여러 막 성분에 MIP 모노머 도킹 |
| 조성 비율로 가중합 → 혼합물 독성 예측 | 조성 비율로 가중합 → 막 친화도 예측 |
| 미생물에 대한 독성 선택성 | 미생물 표면에 대한 MIP 모노머 선택성 |
| AutoDock | AutoDock4/GPU |

### 간접 선행: 막 조성이 결합 선택성을 결정

**Matsuzaki K et al. "Molecular basis for membrane selectivity of an antimicrobial peptide, magainin 2." *Biochemistry* 1995;34(10):3423-3429.** [PMID: 7533538](https://pubmed.ncbi.nlm.nih.gov/7533538/)

- 항균 펩타이드의 **Gram-negative vs Gram-positive 막 선택성**을 막 조성(PE/PG/CL 비율) 차이로 설명
- "Acidic phospholipids(PG)가 있으면 결합력 증가, zwitterionic(PE)만 있으면 약함"
- **막 조성이 리간드 결합 선택성을 결정한다는 원리** — 우리의 가중합 모델의 과학적 기반

### 개별 성분 도킹 선행

**Altintas Z et al. "Ultrasensitive detection of endotoxins using computationally designed nanoMIPs." *Anal. Chim. Acta* 2016;902:77-86.** [PMID: 27543033](https://pubmed.ncbi.nlm.nih.gov/27543033/)
- LPS Lipid A에 21종 모노머 도킹 → MIP 모노머 선정. **Gram-negative 성분 도킹 검증.**

**Narula K, Rajpal S, Bhakta S, Kulanthaivel S, Mishra P. "Rationally designed protein A surface molecularly imprinted magnetic nanoparticles for the capture and detection of Staphylococcus aureus." *J. Mater. Chem. B* 2024;12:5699-5710.**
- **Protein A에 AutoDock4로 모노머 도킹** (PTES, UPTMS, APTES 포함). **S.aureus 성분 도킹 검증.**

---

## 4. E. coli K-12 Outer Membrane 조성

### 외막 구조 개요

E. coli 외막은 **비대칭 이중층**: 외엽은 LPS (~75% 면적), 내엽은 glycerophospholipid (PE/PG/CL). 단백질은 양 엽에 걸쳐 삽입됨. 단백질:지질 질량비 ~2:1 (Silhavy et al. 2010).

### 각 성분별 weight 근거

#### LPS_LipidA: weight = 0.35

- OM 외엽의 **~75%**가 LPS이다 (Nikaido H. *Microbiol. Mol. Biol. Rev.* 2003;67:593-656).
- 그러나 OMP trimer들이 상당한 면적을 차지한다. Lipid/protein mass ratio(LPR)는 0.2-0.4로, 단백질이 ~70% 이상의 질량을 차지 (May & Grabowicz 2018).
- LPS 중 MIP가 접촉하는 **headgroup 부분만** 고려하면 실질 표면 접근 면적은 ~35%.
- **보수적 추정: 0.35**

#### OmpF: weight = 0.15

- **~70,000-80,000 copies/cell**. Trimer로 존재 → ~25,000 trimers.
- 각 trimer의 extracellular surface area ~1,200 A^2.
- 저삼투압 조건(LB 배지)에서 OmpC보다 우세하게 발현.
- Cowan SW et al. "Crystal structures explain functional properties of two E. coli porins." *Nature* 1992;358:727-733.
- **표면 비율 추정: 0.15**

#### OmpA: weight = 0.15

- **~100,000-200,000 copies/cell**. 가장 풍부한 OMP 중 하나.
- 8-strand beta-barrel TM domain. Surface에 4개 short extracellular loops 노출.
- Copy number는 OmpF보다 높지만, extracellular loop 면적이 작아 (porin trimer의 vestibule보다 작음) 비슷한 weight.
- Koebnik R, Locher KP, Van Gelder P. *Mol. Microbiol.* 2000;37(2):239-253. [PMID: 10931321](https://pubmed.ncbi.nlm.nih.gov/10931321/)
- **표면 비율 추정: 0.15**

#### OmpC: weight = 0.10

- OmpF와 상보적 발현: **고삼투압(장내 환경)에서 우세**, 저삼투압(LB)에서 OmpF 우세.
- 표준 배양 조건(LB, 37C)에서는 OmpF보다 약간 적음.
- 임상 분리주에서는 OmpC가 우세한 경우가 흔함 — 장내 osmolarity가 높기 때문.
- Basle A et al. *J. Mol. Biol.* 2006;362:933-942.
- **OmpF보다 한 단계 낮게 설정: 0.10**

#### PE: weight = 0.15

- E. coli 인지질 중 **~75%** (Raetz & Dowhan 1990).
- 그러나 대부분 OM **내엽(inner leaflet)**에 위치.
- 외엽으로 flip하는 PE는 전체의 ~10-20% (Doerrler WT. *Mol. Microbiol.* 2006;60:542-552).
- MIP 접근 가능한 양으로 **0.15**.

#### PG: weight = 0.10

- 인지질 중 **~20%** (Raetz & Dowhan 1990).
- PE보다 적고, 역시 내엽에 주로 존재.
- CL(cardiolipin)은 ~5%이고 주로 inner membrane이므로 포함하지 않음.
- **0.10**

#### 포함하지 않은 성분

- **Lpp (Braun's lipoprotein)**: ~500,000 copies로 가장 많지만, **공유결합으로 PG에 부착** → surface에 미노출 (Braun V. *Biochim. Biophys. Acta* 1975;415:335-377). Weight = 0.
- **CL (cardiolipin)**: ~5% of phospholipids, 주로 inner membrane에 위치. Weight = 0.

| 성분 | Weight | Copy number | Surface type |
|------|--------|-------------|-------------|
| LPS_LipidA | **0.35** | >10^6 | 외엽 주성분 |
| OmpF | **0.15** | ~70-80K | Trimer, extracellular loops |
| OmpA | **0.15** | ~100-200K | Monomer, 4 short loops |
| OmpC | **0.10** | ~50-70K | Trimer, OmpF와 상보적 |
| PE | **0.15** | - | 내엽 주성분, 일부 외엽 flip |
| PG | **0.10** | - | 인지질 중 20% |
| **합계** | **1.00** | | |

---

## 5. P. aeruginosa PAO1 Outer Membrane 조성

### 외막 특이점

- E. coli 대비 **OM 투과성이 12-100배 낮음** — porin 수가 적고 구조가 다름.
  - Hancock REW. "Resistance mechanisms in Pseudomonas aeruginosa and other nonfermentative gram-negative bacteria." *Clin. Infect. Dis.* 1998;27:S93-S99.

### 각 성분별 weight 근거

#### OprF: weight = 0.20

- **가장 풍부한 non-lipoprotein OMP**. E. coli OmpA의 homologue이지만 pore 기능도 가짐.
- 세포 표면에서 가장 접근 가능한 단백질.
- Rawling EG, Brinkman FSL, Hancock REW. "Roles of the carboxy-terminal half of Pseudomonas aeruginosa major outer membrane protein OprF." *Infect. Immun.* 1998;66:1228-1233.
- E. coli에서 OmpF+OmpA를 합친 역할 → **weight을 E.coli보다 높게: 0.20**

#### OprD: weight = 0.10

- **Carbapenem(imipenem) 진입 통로**. 18-strand monomeric barrel.
- OprF보다 적지만, 항생제 내성에서 핵심적 역할 — **임상적 중요성**으로 포함.
- Huang H, Hancock REW. "Structure, function and regulation of Pseudomonas aeruginosa porins." *FEMS Microbiol. Rev.* 2016;41(5):698-722. [DOI](https://academic.oup.com/femsre/article/41/5/698/3959603)
- **0.10**

#### LPS_LipidA: weight = 0.35

- Gram-negative 공통. P. aeruginosa LPS는 E. coli와 Lipid A 구조가 유사하지만, **penta-acylated** (5개 acyl chain) variant가 흔함.
- Lipid IVA(4-acyl)를 사용하므로 P. aeruginosa 특이적 차이는 약간 과소평가될 수 있으나, headgroup 구조(bisphosphorylated glucosamine)는 동일.
- **E. coli와 동일: 0.35**

#### PE: weight = 0.20

- P. aeruginosa는 E. coli보다 PE 비율이 약간 높음 (~80% of phospholipids).
- Touw DS et al. "OprF membrane topology." *J. Biol. Chem.* 2010;285:3753-3762.
- **0.20**

#### PG: weight = 0.15

- E. coli보다 약간 높은 비율 (~15-25% of phospholipids).
- **0.15**

| 성분 | Weight | 근거 |
|------|--------|------|
| OprF | **0.20** | 가장 풍부한 surface OMP |
| OprD | **0.10** | carbapenem entry, 임상 중요 |
| LPS_LipidA | **0.35** | Gram-negative 공통 |
| PE | **0.20** | E.coli보다 약간 높은 PE 비율 |
| PG | **0.15** | |
| **합계** | **1.00** | |

---

## 6. K. pneumoniae Outer Membrane 조성

### 외막 특이점

- **두꺼운 capsule (K-antigen)**: 가장 큰 차이. 캡슐이 박테리아 표면을 완전히 덮어 MIP가 가장 먼저 만나는 구조.
- OmpK35/OmpK36: E. coli OmpF/OmpC의 homologue.

### 각 성분별 weight 근거

#### OmpK36: weight = 0.15

- E. coli OmpC와 **87% sequence identity**. 16-strand barrel trimer.
- **OmpK35:OmpK36 발현 비율 = 1:9** (고삼투압 조건) → OmpK36이 dominant.
- Densitometric 분석: Kaczmarek FM et al. *J. Antimicrob. Chemother.* 2006.
- Wong JLC et al. "OmpK36-mediated Carbapenem resistance attenuates ST258 K. pneumoniae in vivo." *Nat. Commun.* 2019;10:3733.
- **0.15**

#### OmpA_Kp: weight = 0.10

- Enterobacteriaceae에서 **보존적**. E. coli OmpA와 거의 동일한 구조.
- 동일 PDB (1BXW) 사용 가능.
- **E. coli와 동일: 0.10** (OmpK36에 우선순위)

#### LPS_LipidA: weight = 0.35

- Gram-negative 공통. K. pneumoniae LPS의 Lipid A는 E. coli와 구조적으로 유사.
- **0.35**

#### CPS (capsular polysaccharide, K-antigen): weight = 0.20

- K. pneumoniae의 **핵심 독성 인자**. 캡슐이 표면을 완전히 덮음.
- 77개 이상의 K-type이 확인됨 — 각각 다른 다당류 구조.
- MIP가 세포에 접근할 때 **가장 먼저 만나는 층** → 높은 weight.
- 단순화: mannose unit를 대표 당으로 사용 (많은 K-type에서 mannose가 주요 성분).
- Paczosa MK, Mecsas J. "Klebsiella pneumoniae: Going on the Offense with a Strong Defense." *Microbiol. Mol. Biol. Rev.* 2016;80(3):629-661. [PMC5320592](https://pmc.ncbi.nlm.nih.gov/articles/PMC5320592/)
- **0.20**

#### PE/PG: weight = 0.10/0.10

- Enterobacteriaceae 공통. CPS가 surface를 차지하므로 인지질의 상대적 weight 감소.
- **각각 0.10**

| 성분 | Weight | 근거 |
|------|--------|------|
| OmpK36 | **0.15** | Major porin (9:1 vs OmpK35) |
| OmpA_Kp | **0.10** | Enterobacteriaceae 보존 |
| LPS_LipidA | **0.35** | Gram-negative 공통 |
| CPS | **0.20** | 캡슐이 표면 최외곽 |
| PE | **0.10** | CPS 때문에 상대 비중 감소 |
| PG | **0.10** | |
| **합계** | **1.00** | |

---

## 7. S. aureus Cell Surface 조성

### 세포 표면 특이점 (Gram-positive)

- **외막(outer membrane) 없음** — Gram-negative와 근본적으로 다름.
- 세포막(cytoplasmic membrane) 위에 **두꺼운 peptidoglycan + teichoic acid** 층.
- 표면 단백질은 **sortase A**에 의해 peptidoglycan에 공유 고정.
- 이 구조적 차이가 Gram-negative와 Gram-positive를 MIP로 구분 가능하게 하는 핵심.

### 각 성분별 weight 근거

#### Protein A (SpA): weight = 0.25

- S. aureus의 **가장 중요한 표면 마커**.
- **~100,000 copies/cell**. Sortase A가 cell wall peptidoglycan에 공유 고정.
- 5개 IgG-binding domain (E, D, A, B, C) — 면역 회피에 핵심.
- 다른 Staphylococcus 종에는 없는 **S. aureus 특이적 마커** → MIP 선택도의 핵심.
- DeDent AC et al. "Signal peptides direct surface proteins to two distinct envelope locations of Staphylococcus aureus." *EMBO J.* 2008;27:2656-2668.
- Becker S et al. "Release of protein A from the cell wall of S. aureus." *PNAS* 2014;111:1574-1579. [DOI](https://www.pnas.org/doi/10.1073/pnas.1317181111)
- **가장 높은 단백질 weight: 0.25**

#### FnBPA: weight = 0.15

- Fibronectin-binding protein A. 세포 부착, biofilm 형성에 관여.
- Protein A보다 적지만 표면에 노출됨.
- 여러 tandem beta-zipper repeats로 fibronectin에 결합.
- Meenan NAG et al. *PNAS* 2008;105(34):12254-12258.
- **0.15**

#### LTA (Lipoteichoic acid): weight = 0.30

- **Teichoic acid = cell wall mass의 ~50-60%** (Neuhaus & Baddiley 2003).
- LTA는 cell membrane에 glycolipid anchor로 고정, polyglycerol-phosphate chain이 cell wall을 관통하여 표면에 노출.
- **음전하** polymer → 양전하 모노머(APTES 등)와 정전기 상호작용.
- Gram-positive 박테리아의 **공통 표면 성분** → 종 간 구분에는 제한적이나, Gram-neg와의 구분에 핵심.
- Neuhaus FC, Baddiley J. "A continuum of anionic charge: structures and functions of D-alanyl-teichoic acids in gram-positive bacteria." *Microbiol. Mol. Biol. Rev.* 2003;67(4):686-723. [PMID: 14665680](https://pubmed.ncbi.nlm.nih.gov/14665680/)
- **가장 높은 지질 weight: 0.30**

#### PG: weight = 0.20

- S. aureus 세포막의 주요 인지질 (~55% of phospholipids).
- Gram-positive는 외막이 없으므로, 세포막 지질이 cell wall을 통해 부분적으로 접근 가능.
- Tsai M et al. "Staphylococcus aureus requires cardiolipin for survival." *BMC Microbiol.* 2011;11:13.
- **0.20**

#### Lys-PG (Lysyl-phosphatidylglycerol): weight = 0.10

- PG의 **15-38%**가 Lys-PG로 존재. **S. aureus 특이적** 양전하 지질.
- MprF(multipeptide resistance factor)에 의해 합성.
- **양전하** → 양이온 항생제(daptomycin) 및 양전하 MIP 모노머에 대한 반발.
- Daptomycin 내성 메커니즘에 관여.
- Kilelee E et al. "Lysyl-phosphatidylglycerol attenuates membrane perturbation rather than surface association of the cationic antimicrobial peptide 6W-RP-1 in a model membrane system." *J. Biol. Chem.* 2010;285:14823-14828.
- **S. aureus 특이적이므로 포함: 0.10**

#### 포함하지 않은 성분

- **WTA (Wall teichoic acid)**: Ribitol-phosphate polymer. Cell wall에 공유 결합. LTA보다 표면 접근성 낮음 (peptidoglycan에 매몰). WTA와 LTA 중 MIP가 더 접근 가능한 LTA만 포함.
- **Peptidoglycan**: 구조적 scaffold이지만, 그 자체로 MIP가 인식할 만한 반복 단위가 아님 (crosslinked network).

| 성분 | Weight | 근거 |
|------|--------|------|
| ProteinA | **0.25** | ~100K copies, S.aureus 핵심 마커 |
| FnBPA | **0.15** | 부착/biofilm 단백질 |
| LTA | **0.30** | Cell wall mass의 ~50-60% |
| PG | **0.20** | 주요 막 인지질 (~55%) |
| Lys_PG | **0.10** | S.aureus 특이적 양전하 지질 |
| **합계** | **1.00** | |

---

## 8. Weight 결정 원칙 요약

1. **표면 접근성 기반**: MIP는 세포 표면에 접근 → 표면 노출 성분에 높은 weight
2. **Copy number 비례**: 단백질은 copy number에 대략 비례
3. **면적 비례**: 지질은 외엽 면적 비율 기준
4. **캡슐 보정**: K. pneumoniae CPS가 표면을 덮음 → 다른 성분 weight 감소
5. **Gram 분류 반영**: Gram-neg는 LPS 공통 0.35, Gram-pos는 LTA 0.30
6. **종 특이적 마커 강조**: Protein A(S.aureus), Lys-PG(S.aureus), CPS(K.pneumoniae)
7. **모든 종의 weight sum = 1.00**

### Sensitivity Analysis 권장

Weight는 문헌 정량 데이터 + 표면 접근성의 **추정치**이다. 다음 요인에 따라 변동 가능:

- **배양 조건**: LB vs minimal media (OmpF/OmpC 비율 역전)
- **성장 단계**: exponential vs stationary (LPS 양 변화)
- **pH**: acidic pH에서 OMP 발현 변화
- **삼투압**: 고삼투압에서 OmpC >> OmpF

**권장**: weight를 ±50% 변경하여 결과 안정성을 확인. 핵심 결론(어떤 모노머가 어떤 종에 선택적인지)이 weight 변동에 robust한지 확인.

---

## 9. 참고 문헌 전체 목록

1. Park BS et al. *Nature* 2009;458:1191 — TLR4/MD-2/LPS co-crystal (PDB 3FXI)
2. Altintas Z et al. *Anal. Chim. Acta* 2016;902:77 — endotoxin nanoMIP computational design
3. Henderson JC et al. *Annu. Rev. Microbiol.* 2016;70:255 — LPS acyl chain burial
4. May KL, Grabowicz M. *PNAS* 2018;115:8852 — OM asymmetric bilayer
5. Snyder S et al. *Biophys. J.* 2016;111:1750 — Lipid A bilayer properties
6. Clifton LA et al. *J. R. Soc. Interface* 2013;10:20130810 — asymmetric bilayer mimic
7. Raetz CRH, Dowhan W. *J. Biol. Chem.* 1990;265:1235 — E.coli PE:PG:CL = 75:20:5
8. Sohlenkamp C, Geiger O. *FEMS Microbiol. Rev.* 2016;40:133 — bacterial membrane lipids
9. Silhavy TJ et al. *Cold Spring Harb. Perspect. Biol.* 2010;2:a000414 — bacterial cell envelope
10. Nikaido H. *Microbiol. Mol. Biol. Rev.* 2003;67:593 — OM permeability
11. Koebnik R et al. *Mol. Microbiol.* 2000;37:239 — OMP structure (OmpA 100K copies)
12. Hancock REW. *Clin. Infect. Dis.* 1998;27:S93 — P.aeruginosa OM permeability
13. Huang H, Hancock REW. *FEMS Microbiol. Rev.* 2016;41:698 — P.aeruginosa porins
14. Wong JLC et al. *Nat. Commun.* 2019;10:3733 — OmpK36 WT structure (PDB 6RD3)
15. Paczosa MK, Mecsas J. *Microbiol. Mol. Biol. Rev.* 2016;80:629 — K.pneumoniae capsule
16. Neuhaus FC, Baddiley J. *Microbiol. Mol. Biol. Rev.* 2003;67:686 — teichoic acids 50-60% cell wall
17. Becker S et al. *PNAS* 2014;111:1574 — Protein A release
18. Kilelee E et al. *J. Biol. Chem.* 2010;285:18302 — Lys-PG 15-38%
19. Meenan NAG et al. *PNAS* 2008;105:12254 — FnBPA structure (PDB 2RKZ)
20. Basle A et al. *J. Mol. Biol.* 2006;362:933 — OmpC 2.0A (PDB 2J1N)
21. **Yao J et al. *Chemosphere* 2013;92:1169 — 도킹 BE 가중합으로 혼합물 효과 예측 (본 파이프라인 방법론 선행)**
22. **Matsuzaki K et al. *Biochemistry* 1995;34:3423 — 막 조성이 항균 펩타이드 선택성 결정**
23. **Narula K et al. *J. Mater. Chem. B* 2024;12:5699 — Protein A에 AutoDock4 모노머 도킹 (S.aureus MIP)**
