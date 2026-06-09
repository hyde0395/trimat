class TrimatError(Exception):
    """trimat 기반 예외 — 모든 trimat 에러의 상위 클래스."""


class PackError(TrimatError):
    """패킹·양자화 입력이 잘못됐을 때."""


class KernelError(TrimatError):
    """GEMV/GEMM 연산 인자가 잘못됐을 때."""


class LoaderError(TrimatError):
    """HuggingFace 모델 로딩 실패 시 (P3에서 사용)."""
