# Phase 1-3만 먼저 (도킹까지, ~30분)
python run_pipeline.py --phase 1
python run_pipeline.py --phase 2
python run_pipeline.py --phase 3

# 결과 확인 후 Phase 4 (MD, ~수일)
python run_pipeline.py --phase 4 --quick-md    # 50ns 먼저 테스트
python run_pipeline.py --phase 4               # 200ns 풀 실행

# 레시피
python run_pipeline.py --phase 5