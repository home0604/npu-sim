#!/bin/bash
# =============================================================================
# NPU-Sim 명령어 모음 (복사해서 사용)
# =============================================================================

# -----------------------------------------------------------------------------
# 1. 단일 시뮬레이션 (run_sim.py)
# -----------------------------------------------------------------------------

# MatMul 기본 실행
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256

# Dataflow 지정 (OS / WS / IS)
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --dataflow OS
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --dataflow WS
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --dataflow IS

# Double Buffer 활성화
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --dataflow WS --use-db

# JSON 출력
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --json

# 연산 정합성 검증 (PyTorch 결과와 비교)
python3 scripts/run_sim.py --workload matmul --M 64 --N 64 --K 64 --check-correctness

# DRAMSim3 명시 사용 (default.yaml에서 이미 true면 생략 가능)
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --use-dramsim3

# 커스텀 config 파일 사용
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --config configs/my_config.yaml

# Attention 워크로드
python3 scripts/run_sim.py --workload attention --seq-len 128 --hidden-dim 768 --num-heads 12

# Transformer 워크로드
python3 scripts/run_sim.py --workload transformer --seq-len 512 --hidden-dim 768 --num-heads 12

# 옵션 조합 예시
python3 scripts/run_sim.py --workload transformer --seq-len 128 --hidden-dim 768 --num-heads 12 --dataflow WS --use-db --json

# -----------------------------------------------------------------------------
# 2. VQ 시뮬레이션
# -----------------------------------------------------------------------------

# VQ 기본 실행 (vq.yaml 사용)
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --config configs/vq.yaml

# VQ + JSON 출력
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --config configs/vq.yaml --json

# VQ + Double Buffer
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --config configs/vq.yaml --use-db

# VQ + Dataflow 지정
python3 scripts/run_sim.py --workload matmul --M 256 --N 256 --K 256 --config configs/vq.yaml --dataflow WS

# VQ 정합성 검증
python3 scripts/run_sim.py --workload matmul --M 64 --N 64 --K 64 --config configs/vq.yaml --check-correctness

# Dequantization 사이클 파라미터 스윕 (vec_dim × stages × cb_banks × bank_width)
python3 scripts/run_dequant_sweep.py

# -----------------------------------------------------------------------------
# 3. 벤치마크 (run_benchmark.py)
# -----------------------------------------------------------------------------

# 전체 벤치마크 실행 (결과: reports/benchmark/ 에 txt + json 저장)
python3 scripts/run_benchmark.py

# -----------------------------------------------------------------------------
# 4. 테스트
# -----------------------------------------------------------------------------

# 전체 테스트
pytest tests/

# 특정 테스트 파일
pytest tests/test_e2e.py

# 특정 테스트 케이스
pytest tests/test_e2e.py::TestE2E::test_small_matmul -v

# VQ 테스트
pytest tests/test_vq.py -v

# -----------------------------------------------------------------------------
# 5. DRAMSim3 빌드 (최초 1회)
# -----------------------------------------------------------------------------

bash scripts/setup_dramsim3.sh
