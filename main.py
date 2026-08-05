# -*- coding: utf-8 -*-
"""
Mini NPU Simulator
- MAC(Multiply-Accumulate) 연산의 핵심 원리를 반복문으로 직접 구현한 콘솔 애플리케이션
- 외부 라이브러리(NumPy 등) 사용 금지, 표준 라이브러리(json, time, re)만 사용
"""

import json
import time
import re
import os
import statistics

# ------------------------------------------------------------------
# 전역 설정
# ------------------------------------------------------------------
EPSILON = 1e-9              # 점수 비교 허용오차(동점 판정 기준)
DATA_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
PERF_SIZES = [3, 5, 13, 25]  # 성능 분석 대상 크기
PERF_REPEATS = 10            # 성능 분석 반복 횟수


# ------------------------------------------------------------------
# 1. 데이터 구조: NxN 2차원 패턴/필터 저장 및 위치별 읽기/쓰기
# ------------------------------------------------------------------
def create_matrix(n, fill=0.0):
    """N x N 크기의 2차원 배열을 생성한다 (최소 3,5,13,25 지원, N은 임의의 양의 정수)."""
    return [[fill for _ in range(n)] for _ in range(n)]


def set_value(matrix, row, col, value):
    """특정 위치(row, col)에 값을 저장한다."""
    matrix[row][col] = value


def get_value(matrix, row, col):
    """특정 위치(row, col)의 값을 읽어온다."""
    return matrix[row][col]


# ---- 패턴 생성기 (보너스: 십자가·X 패턴 자동 생성) ----
def generate_cross(n):
    """NxN 십자가(+) 패턴을 생성한다. set_value()로 위치별 값을 기록한다."""
    mat = create_matrix(n)
    mid = n // 2
    for i in range(n):
        set_value(mat, i, mid, 1.0)
        set_value(mat, mid, i, 1.0)
    return mat


def generate_x(n):
    """NxN X 패턴을 생성한다. set_value()로 위치별 값을 기록한다."""
    mat = create_matrix(n)
    for i in range(n):
        set_value(mat, i, i, 1.0)
        set_value(mat, i, n - 1 - i, 1.0)
    return mat


# ------------------------------------------------------------------
# 2. MAC 연산 (반복문으로 직접 구현, 외부 라이브러리 금지)
# ------------------------------------------------------------------
def mac_operation(pattern, filt):
    """
    입력 패턴과 필터를 위치별로 곱하고 모두 더한다.
    연산 흐름: 입력(pattern,filt) -> 위치별 곱셈(Multiply) -> 누적 합산(Accumulate) -> 점수 반환
    Multiply(곱하기) + Accumulate(누적 더하기) = MAC 연산
    get_value()로 각 위치의 값을 읽어와 반복문으로 직접 계산한다(외부 라이브러리 미사용).
    """
    n = len(pattern)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += get_value(pattern, i, j) * get_value(filt, i, j)
    return total


# ------------------------------------------------------------------
# 3. 라벨 정규화 (표준 라벨: "Cross", "X")
# ------------------------------------------------------------------
def normalize_expected(raw_label):
    """
    data.json 의 expected 값을 표준 라벨로 정규화한다.

    공식 규칙: expected 값 '+' -> Cross, 'x' -> X
    (과제 자료 초기 버전에는 '+'가 't'로 잘못 표기되어 있었으나, 최신 요구사항
    문서와 실제 data.json 모두 '+'를 사용함을 확인했다. 하위 호환을 위해
    't'/'true'/'cross' 표기도 함께 Cross로 인식하도록 방어적으로 넓혀두었다.)

    우선순위(충돌 처리): Cross 그룹("+","t","true","cross")과 X 그룹("x")은 서로
    겹치는 표기가 없어 현재는 충돌이 발생하지 않는다. 만약 향후 두 그룹에 동시에
    속할 수 있는 표기가 추가된다면, if문 순서상 Cross 그룹이 먼저 검사되므로
    Cross가 우선 적용된다(첫 매칭 우선).
    """
    if raw_label is None:
        return None
    key = str(raw_label).strip().lower()
    if key in ("+", "t", "true", "cross"):
        return "Cross"
    if key == "x":
        return "X"
    return None  # 알 수 없는 라벨 -> 스키마 문제로 처리


def normalize_filter_key(raw_key):
    """필터 JSON의 키('cross', 'x')를 표준 라벨로 정규화한다."""
    if raw_key is None:
        return None
    key = str(raw_key).strip().lower()
    if key == "cross":
        return "Cross"
    if key == "x":
        return "X"
    return None


# ------------------------------------------------------------------
# 4. 모드 1: 사용자 입력(3x3) - 입력 검증 포함
# ------------------------------------------------------------------
def read_matrix(label, size):
    """
    size x size 행렬을 한 줄씩(공백 구분) 입력받는다.
    행/열 개수 불일치, 숫자 파싱 실패 시 안내 문구를 출력하고 재입력을 유도한다.
    """
    while True:
        print(f"\n{label} ({size}줄 입력, 공백 구분)")
        matrix = create_matrix(size)
        error_occurred = False
        for i in range(size):
            line = input(f"  {i + 1}행: ").strip()
            parts = line.split()

            if len(parts) != size:
                print(f"입력 형식 오류 ({i + 1}번째 줄): 각 줄에 {size}개의 숫자를 "
                      f"공백으로 구분해 입력하세요. (입력값: {len(parts)}개)")
                error_occurred = True
                break

            try:
                values = [float(p) for p in parts]
            except ValueError:
                print(f"입력 형식 오류 ({i + 1}번째 줄): 숫자로 변환할 수 없는 값이 "
                      f"포함되어 있습니다. 다시 입력하세요.")
                error_occurred = True
                break

            for j, value in enumerate(values):
                set_value(matrix, i, j, value)

        if error_occurred:
            print("→ 처음부터 다시 입력해 주세요.")
            continue

        return matrix


def judge_scores(score_a, score_b):
    """A/B 점수를 epsilon 기준으로 비교하여 판정한다."""
    if abs(score_a - score_b) < EPSILON:
        return "UNDECIDED"
    return "A" if score_a > score_b else "B"


def run_mode1():
    print("\n" + "-" * 40)
    print("# [1] 필터 입력")
    print("-" * 40)
    filter_a = read_matrix("필터 A", 3)
    filter_b = read_matrix("필터 B", 3)
    print("\n필터 A, B 저장 완료.")

    print("\n" + "-" * 40)
    print("# [2] 패턴 입력")
    print("-" * 40)
    pattern = read_matrix("패턴", 3)

    # MAC 연산 + 시간 측정 (평균/10회)
    start = time.perf_counter()
    for _ in range(PERF_REPEATS):
        score_a = mac_operation(pattern, filter_a)
        score_b = mac_operation(pattern, filter_b)
    elapsed_ms = (time.perf_counter() - start) / PERF_REPEATS * 1000

    judge = judge_scores(score_a, score_b)

    print("\n" + "-" * 40)
    if judge == "UNDECIDED":
        print("# [3] MAC 결과 (판정 불가)")
    else:
        print("# [3] MAC 결과")
    print("-" * 40)
    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")
    print(f"연산 시간(평균/{PERF_REPEATS}회): {elapsed_ms:.3f} ms")
    if judge == "UNDECIDED":
        print(f"판정: 판정 불가 (|A-B| < {EPSILON}, 부동소수점 오차 허용 범위 내 동점)")
    else:
        print(f"판정: {judge}")


# ------------------------------------------------------------------
# 5. 모드 2: JSON 로드 및 스키마 검증(data.json)
# ------------------------------------------------------------------
def load_data_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, f"data.json 파일을 찾을 수 없습니다: {path}"
    except json.JSONDecodeError as e:
        return None, f"data.json 파싱 오류: {e}"


def run_mode2():
    data, err = load_data_json(DATA_JSON_PATH)
    if err:
        print(f"\n[오류] {err}")
        return 0, 0, 0, []

    filters = data.get("filters", {})
    patterns = data.get("patterns", {})

    print("\n" + "-" * 40)
    print("# [1] 필터 로드")
    print("-" * 40)
    for key in sorted(filters.keys()):
        f = filters[key]
        # filter 키('cross', 'x')를 표준 라벨로 정규화하여 출력
        labels = sorted({normalize_filter_key(k) for k in f.keys() if normalize_filter_key(k)})
        print(f"√ {key} 필터 로드 완료 ({', '.join(labels) if labels else '라벨 없음'})")

    print("\n" + "-" * 40)
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("-" * 40)

    total = 0
    passed = 0
    failed = 0
    fail_cases = []

    for key in sorted(patterns.keys()):
        total += 1
        entry = patterns[key]
        print(f"\n--- {key} ---")

        # 2-1. 키에서 크기(N) 추출
        m = re.search(r"size_(\d+)", key)
        if not m:
            reason = "패턴 키에서 크기(N)를 추출할 수 없음 (키 규칙 위반)"
            print(f"판정: FAIL ({reason})")
            failed += 1
            fail_cases.append((key, reason))
            continue

        n = int(m.group(1))
        filter_key = f"size_{n}"

        # 2-2. 해당 크기의 필터 존재 여부 검증
        if filter_key not in filters:
            reason = f"{filter_key} 필터를 찾을 수 없음 (스키마 불일치)"
            print(f"판정: FAIL ({reason})")
            failed += 1
            fail_cases.append((key, reason))
            continue

        filt_set = filters[filter_key]
        # filter 키('cross','x', 대소문자 등)를 표준 라벨로 정규화한 뒤 조회
        normalized_filters = {}
        for raw_key, matrix in filt_set.items():
            norm_key = normalize_filter_key(raw_key)
            if norm_key:
                normalized_filters[norm_key] = matrix

        cross_filter = normalized_filters.get("Cross")
        x_filter = normalized_filters.get("X")
        pattern_input = entry.get("input")

        if cross_filter is None or x_filter is None or pattern_input is None:
            reason = "필터(cross/x) 또는 패턴(input) 데이터 누락"
            print(f"판정: FAIL ({reason})")
            failed += 1
            fail_cases.append((key, reason))
            continue

        # 2-3. 크기 일치 검증
        sizes = {len(pattern_input), len(cross_filter), len(x_filter)}
        row_len_ok = all(len(row) == n for row in pattern_input) and \
                     all(len(row) == n for row in cross_filter) and \
                     all(len(row) == n for row in x_filter)
        if sizes != {n} or not row_len_ok:
            reason = f"크기 불일치 (기대: {n}x{n})"
            print(f"판정: FAIL ({reason})")
            failed += 1
            fail_cases.append((key, reason))
            continue

        # 2-4. MAC 연산
        score_cross = mac_operation(pattern_input, cross_filter)
        score_x = mac_operation(pattern_input, x_filter)

        if abs(score_cross - score_x) < EPSILON:
            judge = "UNDECIDED"
        elif score_cross > score_x:
            judge = "Cross"
        else:
            judge = "X"

        # 2-5. expected 라벨 정규화 및 비교
        expected_raw = entry.get("expected")
        expected_label = normalize_expected(expected_raw)

        print(f"Cross 점수: {score_cross}")
        print(f"X 점수: {score_x}")

        if expected_label is None:
            reason = f"알 수 없는 expected 라벨: '{expected_raw}' (라벨 정규화 실패)"
            print(f"판정: {judge} | expected: '{expected_raw}' (정규화 불가) | FAIL")
            failed += 1
            fail_cases.append((key, reason))
            continue

        if judge == expected_label:
            print(f"판정: {judge} | expected: {expected_label} | PASS")
            passed += 1
        else:
            reason_tag = "동점 규칙" if judge == "UNDECIDED" else "판정 불일치"
            print(f"판정: {judge} | expected: {expected_label} | FAIL ({reason_tag})")
            failed += 1
            fail_cases.append((key, f"판정={judge}, expected={expected_label} ({reason_tag})"))

    return total, passed, failed, fail_cases


# ------------------------------------------------------------------
# 6. 성능 분석 (크기별 MAC 연산 시간 측정)
# ------------------------------------------------------------------
def measure_performance(sizes, repeats=PERF_REPEATS):
    """
    크기별로 십자가 패턴 vs 십자가 필터의 MAC 연산을 repeats회 반복 측정하여
    평균 시간(ms)과 표준편차(ms)를 함께 반환한다.
    (I/O 시간은 제외하고 연산 함수 호출 구간만 측정 / 반복마다 개별 시간을 기록해
    측정값의 흔들림(신뢰도)까지 표준편차로 보여준다)
    """
    results = []
    for n in sizes:
        pattern = generate_cross(n)
        filt = generate_cross(n)

        sample_times_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            mac_operation(pattern, filt)
            sample_times_ms.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(sample_times_ms) / repeats
        stdev_ms = statistics.stdev(sample_times_ms) if repeats > 1 else 0.0

        results.append((n, avg_ms, stdev_ms, n * n))
    return results


def print_performance_table(results):
    print("\n" + "-" * 40)
    print(f"# [성능 분석] (평균±표준편차 / {PERF_REPEATS}회)")
    print("-" * 40)
    print(f"{'크기':<10}{'평균 시간(ms)':<16}{'표준편차(ms)':<14}{'연산 횟수(N^2)'}")
    print("-" * 40)
    for n, avg_ms, stdev_ms, ops in results:
        print(f"{n}x{n:<8}{avg_ms:<16.4f}{stdev_ms:<14.4f}{ops}")


# ------------------------------------------------------------------
# 7. 결과 요약 출력
# ------------------------------------------------------------------
def print_summary(total, passed, failed, fail_cases):
    print("\n" + "-" * 40)
    print("# [결과 요약]")
    print("-" * 40)
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {failed}개")

    if fail_cases:
        print("\n실패 케이스:")
        for case_id, reason in fail_cases:
            print(f" - {case_id}: {reason}")
    elif total == 0:
        print("\n분석된 케이스가 없습니다 (data.json 로드 실패 또는 patterns가 비어 있음).")
    else:
        print("\n실패 케이스 없음 (0 FAIL): 라벨 정규화('+' -> Cross, 'x' -> X)와")
        print("epsilon(1e-9) 기반 동점 판정 정책이 모든 케이스에 정상 적용되었습니다.")


# ------------------------------------------------------------------
# 8. 메인 실행 흐름
# ------------------------------------------------------------------
def main():
    print("=== Mini NPU Simulator ===")
    print("\n[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")

    while True:
        choice = input("선택: ").strip()
        if choice in ("1", "2"):
            break
        print("1 또는 2를 입력해 주세요.")

    total = passed = failed = 0
    fail_cases = []

    if choice == "1":
        run_mode1()
        perf_results = measure_performance(PERF_SIZES)
        print_performance_table(perf_results)
        # 모드 1은 expected 값이 없으므로 PASS/FAIL 집계 대상이 아님
    else:
        total, passed, failed, fail_cases = run_mode2()
        perf_results = measure_performance(PERF_SIZES)
        print_performance_table(perf_results)
        print_summary(total, passed, failed, fail_cases)


if __name__ == "__main__":
    main()