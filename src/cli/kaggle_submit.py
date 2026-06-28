"""src/cli/kaggle_submit.py
Kaggle限速提交引擎 — 指数退避(1s→2s→4s) + 最多3次重试

κ-Phase: Kaggle限速 = κ-Snap对Kaggle平台的限速策略软件模拟,
指数退避确保不触发Kaggle API限速, 最多3次重试确保不浪费配额。

退避公式:
  backoff_time = min(2**retry_count, 4.0)
  retry序列: 1s → 2s → 4s (最多3次)

API密钥:
  从环境变量 KAGGLE_API_KEY 读取 (不硬编码)

Version: v1.0.0 — ARC-AGI主动探测+IDO/TOMAS复盘框架
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    'KaggleSubmission',
    'KaggleSubmitter',
    'KaggleBatchSubmission',
    'MAX_RETRIES',
    'BACKOFF_BASE',
    'BACKOFF_MAX',
    # v4.0 — Kaggle提交常量 (与__init__.py兼容)
    'SubmissionStatus',
    'SubmissionBatch',
    'BACKOFF_INITIAL',
    'BACKOFF_FACTOR',
    'MAX_SUBMISSIONS_PER_HOUR',
    'SUBMISSION_COOLDOWN',
    'KAGGLE_COMPETITION_SLUG',
    'SUBMISSION_TIMEOUT',
    'STATUS_QUERY_TIMEOUT',
]

# 常量定义
MAX_RETRIES: int = 3  # 最多3次重试
BACKOFF_BASE: float = 1.0  # 退避基数 (2**0 = 1s)
BACKOFF_MAX: float = 4.0  # 退避上限 (max 4s)
COMPETITION_NAME: str = "arc-agi-3"  # Kaggle竞赛名

# v4.0 — Kaggle提交常量 (与src/cli/__init__.py兼容)
SubmissionStatus: str = "pending"  # 提交状态默认值 (别名: 用字符串表示状态类型)
BACKOFF_INITIAL: float = BACKOFF_BASE  # 退避初始时间 (别名)
BACKOFF_FACTOR: float = 2.0  # 退避因子 (2**retry_count)
MAX_SUBMISSIONS_PER_HOUR: int = 60  # 每小时最大提交数 (Kaggle限速)
SUBMISSION_COOLDOWN: float = 1.0  # 提交间冷却时间 (秒)
KAGGLE_COMPETITION_SLUG: str = COMPETITION_NAME  # Kaggle竞赛slug (别名)
SUBMISSION_TIMEOUT: float = 30.0  # 单次提交超时 (秒)
STATUS_QUERY_TIMEOUT: float = 10.0  # 状态查询超时 (秒)


# ============================================================================
# §1. 数据结构
# ============================================================================

@dataclass
class KaggleSubmission:
    """Kaggle提交结果 — 单次提交的状态记录。

    κ-Phase: KaggleSubmission = κ-Snap对Kaggle API提交的状态跟踪,
    记录task_id、answer、submission_id、status等。

    Attributes:
        task_id: ARC-AGI任务ID.
        answer: 提交的答案 (变换结果或DSL序列).
        submission_id: Kaggle submission ID (UUID格式).
        status: 提交状态 ('pending'/'accepted'/'rejected'/'error').
        score: 提交评分 (0.0~1.0).
        timestamp: 提交时间戳 (Unix epoch).
        retry_count: 重试次数 (0~3).
        backoff_time: 最后一次重试的退避时间 (秒).
        error_message: 错误信息 (status='error'时).
    """

    task_id: str = ""
    answer: Any = None
    submission_id: str = ""
    status: str = "pending"  # pending/accepted/rejected/error
    score: float = 0.0
    timestamp: float = 0.0
    retry_count: int = 0
    backoff_time: float = 0.0
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式.

        Returns:
            包含所有KaggleSubmission字段的字典.
        """
        return {
            'task_id': self.task_id,
            'answer': self.answer,
            'submission_id': self.submission_id,
            'status': self.status,
            'score': self.score,
            'timestamp': self.timestamp,
            'retry_count': self.retry_count,
            'backoff_time': self.backoff_time,
            'error_message': self.error_message,
        }

    def is_accepted(self) -> bool:
        """判断提交是否被accept.

        Returns:
            True if status='accepted', False otherwise.
        """
        return self.status == "accepted"

    def is_error(self) -> bool:
        """判断提交是否出错.

        Returns:
            True if status='error', False otherwise.
        """
        return self.status == "error"

    def is_final(self) -> bool:
        """判断提交是否为终态 (不再需要重试).

        Returns:
            True if status in ('accepted', 'rejected'), False otherwise.
        """
        return self.status in ("accepted", "rejected")


@dataclass
class KaggleBatchSubmission:
    """批量提交结果 — 多任务的KaggleSubmission集合。

    Attributes:
        submissions: 各任务的KaggleSubmission列表.
        n_total: 总提交数.
        n_accepted: accept数.
        n_rejected: reject数.
        n_error: error数.
        total_duration: 批量提交总耗时 (秒).
    """

    submissions: List[KaggleSubmission] = field(default_factory=list)
    n_total: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    n_error: int = 0
    total_duration: float = 0.0

    def compute_summary(self) -> None:
        """计算汇总统计."""
        self.n_total = len(self.submissions)
        self.n_accepted = sum(1 for s in self.submissions if s.is_accepted())
        self.n_rejected = sum(1 for s in self.submissions if s.status == "rejected")
        self.n_error = sum(1 for s in self.submissions if s.is_error())

    def get_accepted(self) -> List[KaggleSubmission]:
        """获取所有accepted的提交.

        Returns:
            accepted的KaggleSubmission列表.
        """
        return [s for s in self.submissions if s.is_accepted()]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式.

        Returns:
            包含所有KaggleBatchSubmission字段的字典.
        """
        return {
            'submissions': [s.to_dict() for s in self.submissions],
            'n_total': self.n_total,
            'n_accepted': self.n_accepted,
            'n_rejected': self.n_rejected,
            'n_error': self.n_error,
            'total_duration': self.total_duration,
        }


# v4.0 — KaggleBatchSubmission别名 (与src/cli/__init__.py兼容)
SubmissionBatch: type = KaggleBatchSubmission


# ============================================================================
# §2. Kaggle限速提交引擎
# ============================================================================

class KaggleSubmitter:
    """Kaggle限速提交引擎 — 指数退避(1s→2s→4s) + 最多3次重试。

    κ-Phase: KaggleSubmitter = κ-Snap对Kaggle平台的限速策略软件模拟,
    指数退避确保不触发Kaggle API限速, 最多3次重试确保不浪费配额。

    退避公式:
      backoff_time = min(2**retry_count, 4.0)
      retry序列: 1s → 2s → 4s

    API密钥管理:
      - 优先从构造参数 api_key 读取
      - 其次从环境变量 KAGGLE_API_KEY 读取
      - 再次从 ~/.kaggle/kaggle.json 读取
      - 不硬编码任何密钥

    Attributes:
        api_key: Kaggle API密钥 (从环境变量或参数获取).
        max_retries: 最大重试次数, 默认3.
        competition: Kaggle竞赛名, 默认'arc-agi-3'.
        backoff_base: 退避基数, 默认1.0s.
        backoff_max: 退避上限, 默认4.0s.
        dry_run: 是否为模拟模式 (不实际提交), 默认False.
        submissions: 已完成的KaggleSubmission列表.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = MAX_RETRIES,
        competition: str = COMPETITION_NAME,
        backoff_base: float = BACKOFF_BASE,
        backoff_max: float = BACKOFF_MAX,
        dry_run: bool = False,
    ) -> None:
        """初始化Kaggle限速提交引擎.

        Args:
            api_key: Kaggle API密钥 (None=从环境变量读取).
            max_retries: 最大重试次数, 默认3.
            competition: Kaggle竞赛名, 默认'arc-agi-3'.
            backoff_base: 退避基数, 默认1.0s (2**0).
            backoff_max: 退避上限, 默认4.0s.
            dry_run: 模拟模式 (不实际提交), 默认False.
        """
        # API密钥: 优先参数 → 环境变量 → ~/.kaggle/kaggle.json
        self.api_key: str = self._resolve_api_key(api_key)
        self.max_retries: int = max_retries
        self.competition: str = competition
        self.backoff_base: float = backoff_base
        self.backoff_max: float = backoff_max
        self.dry_run: bool = dry_run
        self.submissions: List[KaggleSubmission] = []
        self._kaggle_api: Optional[Any] = None  # KaggleApi实例 (延迟初始化)

    def _resolve_api_key(
        self,
        explicit_key: Optional[str] = None,
    ) -> str:
        """解析API密钥 — 优先参数 → 环境变量 → ~/.kaggle/kaggle.json。

        κ-Phase: 密钥解析 = κ-Snap对API密钥的安全获取策略,
        不硬编码任何密钥, 从多层来源读取。

        Args:
            explicit_key: 显式传入的API密钥.

        Returns:
            解析后的API密钥字符串 (空字符串=未找到).
        """
        # 优先级1: 显式传入
        if explicit_key is not None and len(explicit_key) > 0:
            return explicit_key

        # 优先级2: 环境变量 KAGGLE_API_KEY
        env_key: str = os.environ.get('KAGGLE_API_KEY', '')
        if len(env_key) > 0:
            return env_key

        # 优先级3: ~/.kaggle/kaggle.json
        kaggle_json_path: Path = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json_path.exists():
            try:
                with open(kaggle_json_path, 'r', encoding='utf-8') as f:
                    creds: Dict[str, str] = json.load(f)
                    key: str = creds.get('key', '')
                    if len(key) > 0:
                        return key
            except (json.JSONDecodeError, IOError):
                pass

        # 未找到密钥 → 返回空字符串 (dry_run模式可正常工作)
        return ""

    def exponential_backoff(
        self,
        retry_count: int,
    ) -> float:
        """指数退避时间计算 — min(2**retry_count, max_backoff)。

        κ-Phase: 退避 = κ-Snap对Kaggle限速的指数退避策略,
        2**retry_count确保不触发API限速。

        退避序列: retry 0→1s, retry 1→2s, retry 2→4s

        Args:
            retry_count: 重试次数 (0, 1, 2).

        Returns:
            退避等待时间 (秒), 最多4.0s.
        """
        backoff: float = min(
            self.backoff_base * (2 ** retry_count),
            self.backoff_max,
        )
        return backoff

    def submit(
        self,
        task_id: str,
        answer: Any,
    ) -> KaggleSubmission:
        """指数退避提交 — κ-Snap限速策略提交单个任务。

        κ-Phase: submit = κ-Snap对Kaggle平台的限速提交,
        指数退避(1s→2s→4s) + 最多3次重试。

        算法流程:
          1. 生成submission_id (UUID)
          2. 尝试提交到Kaggle API
          3. 成功 → 返回accepted KaggleSubmission
          4. 失败 → 指数退避等待 → 重试
          5. 超过max_retries → 返回error KaggleSubmission

        Args:
            task_id: ARC-AGI任务ID.
            answer: 提交的答案 (变换结果或DSL序列).

        Returns:
            KaggleSubmission (含最终status和retry_count).
        """
        submission_id: str = f"sub_{uuid.uuid4().hex[:12]}"
        timestamp: float = time.time()

        # dry_run模式: 模拟提交 (不实际调用API)
        if self.dry_run:
            status: str = self._simulate_status(answer)
            score: float = self._simulate_score(answer, status)
            submission: KaggleSubmission = KaggleSubmission(
                task_id=task_id,
                answer=answer,
                submission_id=submission_id,
                status=status,
                score=score,
                timestamp=timestamp,
                retry_count=0,
                backoff_time=0.0,
            )
            self.submissions.append(submission)
            return submission

        # 实际提交: 指数退避重试
        last_error: str = ""
        for retry in range(self.max_retries):
            try:
                result: KaggleSubmission = self._do_submit(
                    task_id, answer, submission_id, timestamp, retry
                )
                self.submissions.append(result)
                return result
            except Exception as exc:
                last_error = str(exc)
                backoff_time: float = self.exponential_backoff(retry)
                if retry < self.max_retries - 1:
                    time.sleep(backoff_time)

        # 所有重试失败 → 返回error
        error_submission: KaggleSubmission = KaggleSubmission(
            task_id=task_id,
            answer=answer,
            submission_id=submission_id,
            status="error",
            score=0.0,
            timestamp=timestamp,
            retry_count=self.max_retries,
            backoff_time=self.exponential_backoff(self.max_retries - 1),
            error_message=last_error,
        )
        self.submissions.append(error_submission)
        return error_submission

    def check_status(
        self,
        submission_id: str,
    ) -> str:
        """查询提交状态 — 检查Kaggle API的submission状态。

        κ-Phase: check_status = κ-Snap对Kaggle平台的submission状态查询,
        支持指数退避重试。

        Args:
            submission_id: Kaggle submission ID.

        Returns:
            状态字符串 ('pending'/'accepted'/'rejected'/'error'/'unknown').
        """
        # dry_run模式: 从本地记录查找
        if self.dry_run:
            for s in self.submissions:
                if s.submission_id == submission_id:
                    return s.status
            return "unknown"

        # 实际查询: 尝试Kaggle API
        for retry in range(self.max_retries):
            try:
                api: Any = self._get_kaggle_api()
                if api is None:
                    return "error"

                # 查询submission状态
                submissions_list: List[Any] = api.competitions_submissions_list(
                    self.competition
                )
                for sub in submissions_list:
                    if hasattr(sub, 'id') and str(sub.id) == submission_id:
                        status_str: str = getattr(sub, 'status', 'unknown')
                        return status_str

                return "pending"  # 未找到 → 可能还在处理中

            except Exception:
                if retry < self.max_retries - 1:
                    time.sleep(self.exponential_backoff(retry))

        return "error"

    def batch_submit(
        self,
        submissions: List[Tuple[str, Any]],
    ) -> KaggleBatchSubmission:
        """批量提交 — 对多个任务依次执行限速提交。

        κ-Phase: batch_submit = κ-Snap对多个ARC-AGI任务的批量限速提交,
        每个任务独立执行指数退避重试。

        Args:
            submissions: [(task_id, answer), ...] 列表.

        Returns:
            KaggleBatchSubmission批量提交结果.
        """
        start_time: float = time.time()
        results: List[KaggleSubmission] = []

        for task_id, answer in submissions:
            result: KaggleSubmission = self.submit(task_id, answer)
            results.append(result)

            # 批量提交间加1s间隔 (避免触发限速)
            if not self.dry_run and len(submissions) > 1:
                time.sleep(1.0)

        batch: KaggleBatchSubmission = KaggleBatchSubmission(
            submissions=results,
            total_duration=time.time() - start_time,
        )
        batch.compute_summary()
        return batch

    def get_submission_by_task(
        self,
        task_id: str,
    ) -> Optional[KaggleSubmission]:
        """按task_id查找最近的提交记录.

        Args:
            task_id: ARC-AGI任务ID.

        Returns:
            最近的KaggleSubmission, 或None.
        """
        for s in reversed(self.submissions):
            if s.task_id == task_id:
                return s
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """获取提交统计信息.

        Returns:
            包含总提交数、accept率、平均重试次数等的统计字典.
        """
        n_total: int = len(self.submissions)
        n_accepted: int = sum(1 for s in self.submissions if s.is_accepted())
        n_error: int = sum(1 for s in self.submissions if s.is_error())
        avg_retries: float = (
            sum(s.retry_count for s in self.submissions) / n_total
            if n_total > 0 else 0.0
        )
        avg_backoff: float = (
            sum(s.backoff_time for s in self.submissions) / n_total
            if n_total > 0 else 0.0
        )

        return {
            'n_total': n_total,
            'n_accepted': n_accepted,
            'n_error': n_error,
            'accept_rate': n_accepted / n_total if n_total > 0 else 0.0,
            'avg_retries': avg_retries,
            'avg_backoff_time': avg_backoff,
            'has_api_key': len(self.api_key) > 0,
            'dry_run': self.dry_run,
        }

    # ========================================================================
    # 内部辅助函数
    # ========================================================================

    def _get_kaggle_api(self) -> Optional[Any]:
        """获取KaggleApi实例 (延迟初始化).

        Returns:
            KaggleApi实例, 或None (kaggle包未安装).
        """
        if self._kaggle_api is not None:
            return self._kaggle_api

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api: KaggleApi = KaggleApi()
            api.authenticate()
            self._kaggle_api = api
            return api
        except ImportError:
            return None
        except Exception:
            return None

    def _do_submit(
        self,
        task_id: str,
        answer: Any,
        submission_id: str,
        timestamp: float,
        retry_count: int,
    ) -> KaggleSubmission:
        """实际执行一次Kaggle API提交.

        Args:
            task_id: ARC-AGI任务ID.
            answer: 提交答案.
            submission_id: 预生成的submission ID.
            timestamp: 提交时间戳.
            retry_count: 当前重试次数.

        Returns:
            KaggleSubmission.

        Raises:
            Exception: API提交失败.
        """
        api: Any = self._get_kaggle_api()
        if api is None:
            raise RuntimeError("Kaggle API not available")

        # 格式化提交答案
        formatted_answer: Dict[str, Any] = self._format_answer(task_id, answer)

        # 提交到Kaggle (通过API或文件上传)
        # ARC-AGI-3: 通过notebook输出提交, 此处为简化版
        result_obj: Any = api.competitions_submissions_submit(
            file_path=None,
            message=f"TOMAS probe: {task_id}",
            competition_id=self.competition,
        )

        # 解析结果
        status: str = "pending"
        score: float = 0.0

        if hasattr(result_obj, 'status'):
            status = str(result_obj.status)
        if hasattr(result_obj, 'score'):
            score = float(result_obj.score)

        backoff_time: float = self.exponential_backoff(retry_count) if retry_count > 0 else 0.0

        return KaggleSubmission(
            task_id=task_id,
            answer=answer,
            submission_id=submission_id,
            status=status,
            score=score,
            timestamp=timestamp,
            retry_count=retry_count,
            backoff_time=backoff_time,
        )

    def _format_answer(
        self,
        task_id: str,
        answer: Any,
    ) -> Dict[str, Any]:
        """格式化提交答案为ARC-AGI-3要求格式.

        Args:
            task_id: ARC-AGI任务ID.
            answer: 原始答案 (numpy array或DSL变换列表).

        Returns:
            格式化后的答案字典.
        """
        if isinstance(answer, dict):
            return answer

        if isinstance(answer, np.ndarray):
            # numpy array → 转为列表格式
            return {
                'task_id': task_id,
                'output': answer.tolist(),
            }

        # 其他格式 → 包装为字典
        return {
            'task_id': task_id,
            'answer': answer,
        }

    def _simulate_status(
        self,
        answer: Any,
    ) -> str:
        """模拟提交状态 (dry_run模式).

        κ-Phase: 模拟 = κ-Snap在dry_run模式下的提交状态模拟,
        根据answer质量判定accepted/rejected。

        Args:
            answer: 提交答案.

        Returns:
            模拟状态字符串.
        """
        # 简化判定: 有answer → pending (待进一步评估)
        if answer is None:
            return "rejected"

        if isinstance(answer, np.ndarray):
            # 有实际输出 → 模拟为accepted (简化)
            if answer.size > 0:
                return "accepted"
            return "rejected"

        if isinstance(answer, list):
            if len(answer) > 0:
                return "accepted"
            return "rejected"

        if isinstance(answer, dict):
            if 'output' in answer or 'answer' in answer:
                return "accepted"
            return "rejected"

        return "pending"

    def _simulate_score(
        self,
        answer: Any,
        status: str,
    ) -> float:
        """模拟提交评分 (dry_run模式).

        Args:
            answer: 提交答案.
            status: 模拟状态.

        Returns:
            模拟评分 (0.0~1.0).
        """
        if status == "accepted":
            return 0.85  # 模拟高分
        elif status == "rejected":
            return 0.0
        else:
            return 0.5  # pending


# ============================================================================
# §3. 自测函数
# ============================================================================

def _self_test() -> bool:
    """KaggleSubmitter自测: 验证指数退避、限速提交、批量提交。

    Returns:
        True if all tests pass, False otherwise.
    """
    # 测试1: exponential_backoff公式
    submitter: KaggleSubmitter = KaggleSubmitter(dry_run=True)

    assert submitter.exponential_backoff(0) == 1.0, f"retry 0 → 1s, got {submitter.exponential_backoff(0)}"
    assert submitter.exponential_backoff(1) == 2.0, f"retry 1 → 2s, got {submitter.exponential_backoff(1)}"
    assert submitter.exponential_backoff(2) == 4.0, f"retry 2 → 4s, got {submitter.exponential_backoff(2)}"
    assert submitter.exponential_backoff(3) == 4.0, f"retry 3 → capped at 4s, got {submitter.exponential_backoff(3)}"
    assert submitter.exponential_backoff(10) == 4.0, f"retry 10 → capped at 4s, got {submitter.exponential_backoff(10)}"

    # 测试2: KaggleSubmission数据结构
    sub: KaggleSubmission = KaggleSubmission(
        task_id="test_001",
        answer={"output": [[1, 2], [3, 4]]},
        submission_id="sub_abc123",
        status="accepted",
        score=0.85,
        timestamp=time.time(),
    )
    assert sub.is_accepted(), "KaggleSubmission.is_accepted() should return True"
    assert not sub.is_error(), "KaggleSubmission.is_error() should return False"
    assert sub.is_final(), "accepted → is_final=True"

    sub_dict: Dict[str, Any] = sub.to_dict()
    assert sub_dict['task_id'] == "test_001", "to_dict should preserve task_id"
    assert sub_dict['status'] == "accepted", "to_dict should preserve status"

    # 测试3: dry_run模式提交
    result: KaggleSubmission = submitter.submit("test_dry", [[1, 2], [3, 4]])
    assert result.task_id == "test_dry", "dry_run submit should have correct task_id"
    assert result.submission_id.startswith("sub_"), "dry_run should generate submission_id"
    assert result.status in ("accepted", "pending"), f"dry_run status should be accepted/pending, got {result.status}"

    # 测试4: dry_run模式 — None answer → rejected
    result_none: KaggleSubmission = submitter.submit("test_none", None)
    assert result_none.status == "rejected", f"None answer → rejected, got {result_none.status}"

    # 测试5: 批量提交 (dry_run)
    batch_inputs: List[Tuple[str, Any]] = [
        ("task_a", [[1, 2], [3, 4]]),
        ("task_b", [[5, 6], [7, 8]]),
        ("task_c", None),
    ]
    batch_result: KaggleBatchSubmission = submitter.batch_submit(batch_inputs)
    assert batch_result.n_total == 3, f"batch should have 3 submissions, got {batch_result.n_total}"
    assert batch_result.n_accepted >= 1, "batch should have at least 1 accepted"
    assert batch_result.n_rejected >= 1, "batch should have at least 1 rejected (None answer)"

    # 测试6: check_status (dry_run)
    status: str = submitter.check_status(result.submission_id)
    assert status == "accepted", f"check_status should find accepted, got {status}"

    status_unknown: str = submitter.check_status("nonexistent_id")
    assert status_unknown == "unknown", f"nonexistent → unknown, got {status_unknown}"

    # 测试7: API密钥解析 — 从环境变量
    os.environ['KAGGLE_API_KEY'] = 'test_key_12345'
    submitter_env: KaggleSubmitter = KaggleSubmitter(dry_run=True)
    assert submitter_env.api_key == 'test_key_12345', f"Should read from env, got {submitter_env.api_key}"

    # 清理环境变量
    os.environ.pop('KAGGLE_API_KEY', None)

    # 测试8: API密钥解析 — 显式参数优先
    os.environ['KAGGLE_API_KEY'] = 'env_key'
    submitter_explicit: KaggleSubmitter = KaggleSubmitter(
        api_key='explicit_key', dry_run=True
    )
    assert submitter_explicit.api_key == 'explicit_key', "Explicit key should override env"

    os.environ.pop('KAGGLE_API_KEY', None)

    # 测试9: API密钥解析 — 无密钥
    submitter_no_key: KaggleSubmitter = KaggleSubmitter(dry_run=True)
    # 无密钥但dry_run → 应能正常工作
    result_no_key: KaggleSubmission = submitter_no_key.submit("test_no_key", [[1, 2]])
    assert result_no_key.task_id == "test_no_key", "dry_run should work without API key"

    # 测试10: get_submission_by_task
    found: Optional[KaggleSubmission] = submitter.get_submission_by_task("test_dry")
    assert found is not None, "Should find submission by task_id"
    assert found.task_id == "test_dry", "Found submission should have correct task_id"

    not_found: Optional[KaggleSubmission] = submitter.get_submission_by_task("nonexistent")
    assert not_found is None, "Nonexistent task → None"

    # 测试11: get_statistics
    stats: Dict[str, Any] = submitter.get_statistics()
    assert 'n_total' in stats, "Statistics should have n_total"
    assert 'accept_rate' in stats, "Statistics should have accept_rate"
    assert 'avg_retries' in stats, "Statistics should have avg_retries"

    # 测试12: KaggleBatchSubmission.to_dict
    batch_dict: Dict[str, Any] = batch_result.to_dict()
    assert 'submissions' in batch_dict, "batch.to_dict should have submissions"
    assert 'n_total' in batch_dict, "batch.to_dict should have n_total"

    # 测试13: _format_answer — numpy array
    import numpy as np
    np_answer: np.ndarray = np.array([[1, 2], [3, 4]])
    formatted: Dict[str, Any] = submitter._format_answer("np_test", np_answer)
    assert 'task_id' in formatted, "Formatted should have task_id"
    assert 'output' in formatted, "Formatted should have output"

    # 测试14: exponential_backoff自定义参数
    custom_submitter: KaggleSubmitter = KaggleSubmitter(
        dry_run=True, backoff_base=2.0, backoff_max=8.0
    )
    assert custom_submitter.exponential_backoff(0) == 2.0, "Custom base=2 → retry 0: 2s"
    assert custom_submitter.exponential_backoff(1) == 4.0, "Custom base=2 → retry 1: 4s"
    assert custom_submitter.exponential_backoff(2) == 8.0, "Custom max=8 → retry 2: 8s"
    assert custom_submitter.exponential_backoff(3) == 8.0, "Custom max=8 → retry 3: capped 8s"

    print("[PASS] kaggle_submit _self_test passed")
    return True


if __name__ == "__main__":
    _self_test()
