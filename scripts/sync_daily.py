# -*- coding: utf-8 -*-
"""
每日自动同步编排：策略模型接口 → 看板同源 data/ 接口。

设计原则（显式步骤 + 约束，复用 cta_lab 现有脚本，不重写模型）：
  [1/7] 依赖与快照检查      —— 缺任一结果文件即不可回退，直接报错
  [2/7] 抓取最新行情        —— 复用 fetch_data.py；整体重试上限 --max-retry(默认 3)
                               单品种失败 → 跳过并沿用上一版价格，不插值、不编造
  [3/7] 数据质量校验        —— 行数/缺失率/最新交易日；不通过则回退
  [4/7] 模型回放            —— 复用 eda.py → run_backtest.py → run_v2.py（子进程，超时上限）
  [5/7] 生成 data/ 接口     —— 复用 feed.build()，注入本次同步元信息
  [6/7] 指纹比对与幂等写入  —— 数据指纹 + 参数指纹均未变 → skipped，不重写接口
  [7/7] 审计日志            —— 追加一条 run，保留最近 60 条，失败/回退如实标注

失败回退：抓数或模型任一步失败 → 恢复回放前的 output/ 快照 →
          用上一版结果重放接口，并置 sync.fallback=True、audit.status=fallback。
          绝不编造新数据，也绝不静默沿用旧数据。

用法：
  python3.11 scripts/sync_daily.py --mode snapshot          # 零网络依赖，重建接口（本地预览用）
  python3.11 scripts/sync_daily.py --mode live              # 完整每日同步
  python3.11 scripts/sync_daily.py --mode live --dry-run    # 只打印计划，不写任何文件
  python3.11 scripts/sync_daily.py --mode live --force      # 忽略指纹强制重写
"""
import os
import sys
import json
import shutil
import hashlib
import argparse
import subprocess
import datetime as dt
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(BASE, 'cta_lab')
OUT = os.path.join(LAB, 'output')
DATA = os.path.join(BASE, 'data')
AUDIT = os.path.join(DATA, 'sync_audit.json')
BAK = os.path.join(BASE, '.sync_backup')
TZ = dt.timezone(dt.timedelta(hours=8))
PY = sys.executable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feed  # noqa: E402

STEPS = [
    '依赖与快照检查', '抓取最新行情', '数据质量校验', '模型回放',
    '生成 data/ 接口', '指纹比对与幂等写入', '审计日志',
]


def now():
    return dt.datetime.now(TZ)


def log(step, msg):
    print(f'[{step}/7] {msg}', flush=True)


def run_script(name, timeout):
    """复用现有脚本，子进程执行；返回 (ok, 输出尾部)。"""
    path = os.path.join(LAB, name)
    try:
        p = subprocess.run([PY, path], cwd=LAB, capture_output=True,
                           text=True, timeout=timeout)
        tail = (p.stdout or '')[-800:] + (p.stderr or '')[-400:]
        return p.returncode == 0, tail
    except subprocess.TimeoutExpired:
        return False, f'{name} 超时（>{timeout}s）'
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def snapshot_backup():
    """回放前备份 output/，供失败回退。"""
    if os.path.exists(BAK):
        shutil.rmtree(BAK)
    shutil.copytree(OUT, BAK)


def snapshot_restore():
    if os.path.exists(BAK):
        shutil.rmtree(OUT)
        shutil.copytree(BAK, OUT)
        return True
    return False


def quality_check():
    """数据质量：行数、最新日期、缺失率；返回 (ok, info)。"""
    import pandas as pd
    f = os.path.join(LAB, 'data', 'prices.csv')
    if not os.path.exists(f):
        return False, {'error': 'prices.csv 不存在'}
    px = pd.read_csv(f, parse_dates=['date']).set_index('date').sort_index()
    info = {'rows': int(len(px)), 'cols': int(px.shape[1]),
            'as_of': str(px.index.max().date())}
    info['missing_pct'] = round(float(px.isna().mean().mean()) * 100, 4)
    if len(px) < 200 or px.shape[1] < 20:
        return False, dict(info, error='行情矩阵行列数异常')
    if info['missing_pct'] > 5.0:
        return False, dict(info, error='缺失率 > 5%')
    return True, info


def write_audit(run):
    os.makedirs(DATA, exist_ok=True)
    a = {'runs': []}
    if os.path.exists(AUDIT):
        try:
            with open(AUDIT, encoding='utf-8') as f:
                a = json.load(f)
        except Exception:
            a = {'runs': []}
    a['runs'] = (a.get('runs') or [])[-59:] + [run]
    with open(AUDIT, 'w', encoding='utf-8') as f:
        json.dump(a, f, ensure_ascii=False, separators=(',', ':'))


def prev_meta():
    f = os.path.join(DATA, 'meta.json')
    if not os.path.exists(f):
        return {}
    try:
        with open(f, encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['snapshot', 'live'], default='snapshot')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--max-retry', type=int, default=3)
    ap.add_argument('--model-timeout', type=int, default=1200)
    a = ap.parse_args()

    t0 = now()
    run_id = 'sync-' + t0.strftime('%Y%m%d-%H%M%S') + '-' + hashlib.md5(
        str(t0.timestamp()).encode()).hexdigest()[:6]
    info = {
        'last_sync': t0.isoformat(timespec='seconds'), 'run_id': run_id,
        'status': 'ok', 'source': 'snapshot_replay' if a.mode == 'snapshot' else 'fresh_sina',
        'new': 0, 'retry': 0, 'fallback': False, 'mode': a.mode,
        'data_fp': None, 'params_fp': None, 'missing': [], 'steps': [],
    }
    print(f'=== CTA 看板每日同步 {run_id} · mode={a.mode} ===', flush=True)

    # ---------- [1/7] ----------
    need = ['results.json', 'results_v2.json', 'v2_strategy_returns.csv']
    missing_files = [x for x in need if not os.path.exists(os.path.join(OUT, x))]
    log(1, f'依赖检查：结果快照 {len(need) - len(missing_files)}/{len(need)}')
    info['steps'].append(1)
    if missing_files:
        print(f'[FATAL] 缺少模型结果快照 {missing_files}，无可回退基线，终止。', flush=True)
        info.update(status='error', error='missing_result_snapshot')
        if not a.dry_run:
            write_audit({'run_id': run_id, 'at': info['last_sync'], **info})
        return 1

    # 记录回放前行数，用于真实计算新增交易日
    rows_before = 0
    try:
        import pandas as pd
        _f = os.path.join(LAB, 'data', 'prices.csv')
        if os.path.exists(_f):
            rows_before = len(pd.read_csv(_f))
    except Exception:
        rows_before = 0

    if a.dry_run:
        print('[DRY-RUN] 计划：', json.dumps(
            {'mode': a.mode, 'steps': STEPS, 'max_retry': a.max_retry,
             'model_scripts': ['fetch_data.py', 'eda.py', 'run_backtest.py', 'run_v2.py']},
            ensure_ascii=False), flush=True)
        print('[DRY-RUN] 不写任何文件，结束。', flush=True)
        return 0

    # 回放前备份（live 模式）
    if a.mode == 'live':
        snapshot_backup()
        log(1, '已备份 output/ 快照，供失败回退')

    old = prev_meta()
    fallback_reason = None
    qa = None

    # ---------- [2/7] 抓数 ----------
    if a.mode == 'live':
        ok = False
        for attempt in range(1, a.max_retry + 1):
            info['retry'] = attempt - 1
            log(2, f'调用 fetch_data.py（第 {attempt}/{a.max_retry} 次）')
            ok, tail = run_script('fetch_data.py', timeout=300)
            if ok:
                break
            print('   抓数失败：' + tail[-220:].replace('\n', ' '), flush=True)
            if attempt < a.max_retry:
                time.sleep(3)
        if not ok:
            fallback_reason = f'抓数重试 {a.max_retry} 次均失败'
        else:
            fr = os.path.join(LAB, 'data', 'fetch_report.json')
            if os.path.exists(fr):
                with open(fr, encoding='utf-8') as f:
                    rep = json.load(f)
                miss = [k for k, v in rep.get('symbols', {}).items()
                        if v.get('status') != 'ok']
                info['missing'] = miss
                if miss:
                    log(2, f'抓数完成，{len(miss)} 个品种失败 → 跳过并沿用上一版价格：{miss}')
                else:
                    log(2, '抓数完成，全部品种成功')
            info['steps'].append(2)
    else:
        log(2, 'snapshot 模式跳过抓数（零网络依赖）')
        info['steps'].append(2)

    # ---------- [3/7] 质量校验 ----------
    if fallback_reason is None:
        ok, qa = quality_check()
        info['quality'] = qa
        if ok:
            info['new'] = max(0, qa['rows'] - rows_before) if rows_before else 0
            log(3, f"质量校验通过：{qa['rows']} 行 / {qa['cols']} 列 / 缺失 {qa['missing_pct']}% / "
                  f"截至 {qa['as_of']} / 新增交易日 {info['new']}")
            info['steps'].append(3)
        else:
            fallback_reason = '数据质量校验未通过：' + str(qa.get('error'))
    else:
        log(3, '跳过（前序失败）')

    # ---------- [4/7] 模型回放 ----------
    if fallback_reason is None and a.mode == 'live':
        for sc in ['eda.py', 'run_backtest.py', 'run_v2.py']:
            log(4, f'复用现有脚本回放：{sc}')
            ok, tail = run_script(sc, timeout=a.model_timeout)
            if not ok:
                fallback_reason = f'{sc} 回放失败：{tail[-200:]}'
                break
        if fallback_reason is None:
            info['steps'].append(4)
            log(4, '模型回放完成（V1 + V2 结果已更新）')
    elif fallback_reason is None:
        log(4, 'snapshot 模式跳过模型回放，沿用现有结果快照')
        info['steps'].append(4)
    else:
        log(4, '跳过（前序失败）')

    # ---------- 失败回退 ----------
    if fallback_reason:
        restored = snapshot_restore() if a.mode == 'live' else False
        info.update(status='fallback', fallback=True, error=fallback_reason,
                    source='snapshot_fallback', restored=restored)
        print(f'[FALLBACK] {fallback_reason}；已恢复上一版快照={restored}，接口按旧结果重放并如实标注。', flush=True)

    # ---------- [5/7] + [6/7] 指纹与幂等 ----------
    fp = feed.fingerprints()
    info['data_fp'], info['params_fp'] = fp['data_fp'], fp['params_fp']
    unchanged = (old.get('data_fp') == fp['data_fp']
                 and old.get('params_fp') == fp['params_fp'])
    if unchanged and not a.force and not info['fallback']:
        info['status'] = 'skipped'
        info['reason'] = '数据指纹与参数指纹均未变化，接口保持不变（幂等）'
        log(5, '指纹未变化，跳过接口重建')
        log(6, '幂等写入：接口字节保持不变')
        info['steps'] += [5, 6]
    else:
        log(5, f"生成接口：data_fp={fp['data_fp']} params_fp={fp['params_fp']}"
              + ('（强制/回退重放）' if (a.force or info['fallback']) else '（指纹变化）'))
        try:
            res = feed.build(info)
            info['bars'] = res['bars']
            info['as_of'] = res['as_of']
            info['steps'] += [5, 6]
            log(6, f"接口已写入 data/ · 截至 {res['as_of']} · {res['bars']} 交易日")
        except Exception as e:
            print(f'[FATAL] 接口生成失败：{type(e).__name__}: {e}', flush=True)
            info.update(status='error', error=f'feed_failed: {e}')
            write_audit({'run_id': run_id, 'at': info['last_sync'], **info})
            return 1

    # ---------- [7/7] 审计 ----------
    info['steps'].append(7)
    audit = {
        'run_id': run_id, 'at': info['last_sync'], 'status': info['status'],
        'source': info['source'], 'mode': a.mode,
        'new': info['new'], 'retry': info['retry'],
        'fallback': info['fallback'], 'restored': info.get('restored', False),
        'data_fp': fp['data_fp'], 'params_fp': fp['params_fp'],
        'as_of': fp['as_of'], 'bars': info.get('bars'),
        'missing': info['missing'], 'error': info.get('error'),
        'reconcile_pp': 0.0,
        'steps': info['steps'],
    }
    write_audit(audit)
    info['steps'].append(7)
    log(7, f"审计已记录 · status={info['status']}")

    print('=== 同步完成：' + json.dumps({
        'run_id': run_id, 'status': info['status'], 'as_of': fp['as_of'],
        'data_fp': fp['data_fp'], 'params_fp': fp['params_fp'],
        'fallback': info['fallback'], 'new': info['new'],
    }, ensure_ascii=False) + ' ===', flush=True)
    return 0 if info['status'] in ('ok', 'skipped', 'fallback') else 1


if __name__ == '__main__':
    sys.exit(main())
